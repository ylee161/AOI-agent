"""Optuna Bayesian-optimization block for the `optimizer/lr-schedule` target.

`get_optuna_block` returns a self-contained Python code block (as a string) that
the Phase 2 refinement coder splices into a generated training script, just
BEFORE the main training loop. The block runs a short TPE study over the
optimizer / lr-schedule hyperparameters, then builds the FINAL `optimizer` and
`scheduler` from the best trial so the main loop can consume them directly.

The block deliberately references names the surrounding script already defines —
`build_model`, `train_loader`, `val_loader`, `pos_weight`, `device` — and leaves
`optimizer`, `scheduler`, and `best_params` in scope for the main loop. When
fewer than OPTUNA_MIN_TRIALS trials complete (timeout / failures), it sets
`best_params = None` so the script can fall back to the LLM-written optimizer.
"""


def get_optuna_block(n_trials: int, timeout_seconds: int, search_epochs: int) -> str:
    """Return the Optuna search-and-build code block as a string.

    Args:
        n_trials:        max number of TPE trials for the study.
        timeout_seconds: wall-clock cap for `study.optimize` (timed_out flag).
        search_epochs:   epochs trained per trial during the search.

    The returned block defines `_optuna_objective(trial)`, runs the study with a
    seeded `TPESampler` (seed=42, direction="minimize" on val_loss), and — when
    `len(study.trials) >= OPTUNA_MIN_TRIALS` — assigns `best_params` and builds
    the final `optimizer`/`scheduler`; otherwise `best_params = None`. It prints
    one `OPTUNA_RESULT: {...}` JSON line with best_params, trial_count, timed_out.
    """
    block = '''# ─── Optuna Bayesian hyperparameter search (optimizer/lr-schedule) ──────────
import optuna
from optuna.samplers import TPESampler

OPTUNA_N_TRIALS = __N_TRIALS__
OPTUNA_TIMEOUT_SECONDS = __TIMEOUT_SECONDS__
OPTUNA_SEARCH_EPOCHS = __SEARCH_EPOCHS__
OPTUNA_MIN_TRIALS = __MIN_TRIALS__


def _build_optimizer_from_params(model, params):
    """Construct an optimizer from a suggested-params dict (search and final)."""
    opt_name = params["optimizer"]
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    if opt_name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # SGD
    return torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=params.get("momentum", 0.9),
        nesterov=True,
        weight_decay=weight_decay,
    )


def _build_scheduler_from_params(optimizer, params):
    """Construct an lr scheduler from a suggested-params dict (search and final)."""
    sched_name = params["scheduler"]
    if sched_name == "CosineWarm":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=params.get("T_0", 5),
            T_mult=params.get("T_mult", 2),
            eta_min=params.get("eta_min", 1e-6),
        )
    # ReducePlateau
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=params.get("factor", 0.5),
        patience=params.get("plateau_patience", 2),
        min_lr=params.get("min_lr", 1e-6),
    )


def _optuna_suggest_params(trial):
    """Suggest the full optimizer/lr-schedule hyperparameter set for one trial."""
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["AdamW", "SGD"]),
        "scheduler": trial.suggest_categorical("scheduler", ["CosineWarm", "ReducePlateau"]),
    }
    if params["optimizer"] == "SGD":
        params["momentum"] = trial.suggest_float("momentum", 0.8, 0.99)
    if params["scheduler"] == "CosineWarm":
        params["T_0"] = trial.suggest_int("T_0", 3, 10)
        params["T_mult"] = trial.suggest_int("T_mult", 1, 2)
        params["eta_min"] = trial.suggest_float("eta_min", 1e-7, 1e-4, log=True)
    else:  # ReducePlateau
        params["factor"] = trial.suggest_float("factor", 0.1, 0.7)
        params["plateau_patience"] = trial.suggest_int("plateau_patience", 1, 3)
        params["min_lr"] = trial.suggest_float("min_lr", 1e-7, 1e-4, log=True)
    return params


def _optuna_objective(trial):
    """Train a fresh model for OPTUNA_SEARCH_EPOCHS and return final val_loss."""
    params = _optuna_suggest_params(trial)
    model = build_model().to(device)
    optimizer = _build_optimizer_from_params(model, params)
    scheduler = _build_scheduler_from_params(optimizer, params)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    val_loss = float("inf")
    for _epoch in range(OPTUNA_SEARCH_EPOCHS):
        model.train()
        for batch in train_loader:
            inputs, targets = batch[0].to(device), batch[1].to(device).float()
            optimizer.zero_grad()
            logits = model(inputs).squeeze(-1)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                scheduler.step(_epoch + 1)

        model.eval()
        _losses = []
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch[0].to(device), batch[1].to(device).float()
                logits = model(inputs).squeeze(-1)
                _losses.append(criterion(logits, targets).item())
        val_loss = float(np.mean(_losses)) if _losses else float("inf")
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        trial.report(val_loss, _epoch)
    return val_loss


_optuna_timed_out = False
_optuna_study = optuna.create_study(
    direction="minimize",
    sampler=TPESampler(seed=42),
)
try:
    _optuna_t0 = time.time()
    _optuna_study.optimize(
        _optuna_objective,
        n_trials=OPTUNA_N_TRIALS,
        timeout=OPTUNA_TIMEOUT_SECONDS,
    )
    _optuna_timed_out = (time.time() - _optuna_t0) >= OPTUNA_TIMEOUT_SECONDS
except Exception as _optuna_err:  # search must never crash the training run
    print("OPTUNA_WARNING:", repr(_optuna_err))

_optuna_trial_count = len(_optuna_study.trials)
if _optuna_trial_count >= OPTUNA_MIN_TRIALS:
    best_params = _optuna_study.best_params
    OPTIMIZER_LR_SCHEDULE_VARIANT = "optuna_" + best_params["optimizer"].lower() + "_" + best_params["scheduler"].lower()
    optimizer = _build_optimizer_from_params(model, best_params)
    scheduler = _build_scheduler_from_params(optimizer, best_params)
else:
    best_params = None  # too few trials — main loop falls back to LLM-written params

print("OPTUNA_RESULT:", json.dumps({
    "best_params": best_params,
    "trial_count": _optuna_trial_count,
    "timed_out": bool(_optuna_timed_out),
}))
# ─── end Optuna block ───────────────────────────────────────────────────────
'''
    return (
        block
        .replace("__N_TRIALS__", str(n_trials))
        .replace("__TIMEOUT_SECONDS__", str(timeout_seconds))
        .replace("__SEARCH_EPOCHS__", str(search_epochs))
        .replace("__MIN_TRIALS__", "5")
    )
