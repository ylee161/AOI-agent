"""Validation for the feature-level shared-weight Siamese difference candidate family.

This module verifies that a candidate model implements a *feature-level* stereo
difference, NOT the pixel-level 9-channel ``abs(L - R)`` input that the baseline
coder prompt already describes. The two are different and must not be conflated:

  * Pixel-level (existing 9-channel input): ``abs(img_l - img_r)`` is computed on
    the raw images and concatenated to the 3+3 channels BEFORE the encoder. The
    encoder sees a 9-channel tensor. There is no per-branch feature extraction.

  * Feature-level Siamese difference (this module): a single SHARED encoder
    processes the left image -> ``f_L`` and the same shared encoder (same weights)
    processes the right image -> ``f_R``. The absolute feature difference
    ``|f_L - f_R|`` is computed, and the classifier head receives the concatenation
    ``[f_L, f_R, |f_L - f_R|]`` (width == 3 x feature_dim).

Two complementary checks are provided so validation does not rely on comments or
string matching:

  1. ``validate_difference_feature_source(source)`` — static AST dataflow analysis
     over the model's ``forward`` method. Confirms a shared encoder is applied to
     two distinct inputs, an absolute difference of the two encoder outputs is
     computed, and both features plus the difference flow into a single concat.

  2. ``forward_pass_structural_check(model, dummy_left, dummy_right, ...)`` — a live
     forward pass with dummy tensors. Confirms (via per-instance call counting and
     traced ``torch.cat`` / ``torch.abs``) that a single encoder instance is invoked
     twice (shared weights, not two independent backbones), that an absolute
     difference equal to ``|f_L - f_R|`` is actually placed into the concatenation,
     and that the concatenation / classifier-head input width equals ``3 * feat_dim``.

Both return plain dicts so they can be surfaced through an ADK ``FunctionTool``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# ---------------------------------------------------------------------------
# 1. Static AST analysis
# ---------------------------------------------------------------------------


@dataclass
class DiffFeatureSourceReport:
    ok: bool = False
    model_class: Optional[str] = None
    shared_encoder: bool = False
    independent_backbones: bool = False
    abs_diff_of_features: bool = False
    diff_and_both_features_in_concat: bool = False
    concat_passed_to_head: bool = False
    encoder_attr: Optional[str] = None
    feature_vars: list = field(default_factory=list)
    diff_var: Optional[str] = None
    reasons: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _is_module_base(base: ast.expr) -> bool:
    """True for `nn.Module`, `torch.nn.Module`, or a bare `Module` base class."""
    if isinstance(base, ast.Attribute):
        return base.attr == "Module"
    if isinstance(base, ast.Name):
        return base.id == "Module"
    return False


def _self_attr_call(node: ast.AST) -> Optional[tuple[str, ast.expr]]:
    """If `node` is `self.<attr>(<single arg>)`, return (attr, arg_node)."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and len(node.args) == 1
    ):
        return node.func.attr, node.args[0]
    return None


def _abs_sub_operands(node: ast.AST) -> Optional[tuple[ast.expr, ast.expr]]:
    """Return (a, b) if `node` is abs(a - b) in any of these forms:
    torch.abs(a - b), abs(a - b), (a - b).abs(), torch.absolute(a - b).
    """
    # torch.abs(...) / abs(...) / torch.absolute(...)
    if isinstance(node, ast.Call) and len(node.args) >= 1:
        fname = None
        if isinstance(node.func, ast.Attribute):
            fname = node.func.attr
        elif isinstance(node.func, ast.Name):
            fname = node.func.id
        if fname in ("abs", "absolute"):
            inner = node.args[0]
            if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Sub):
                return inner.left, inner.right
        # (a - b).abs()
        if (
            fname == "abs"
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.BinOp)
            and isinstance(node.func.value.op, ast.Sub)
        ):
            return node.func.value.left, node.func.value.right
    return None


def _names_in(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _cat_elements(node: ast.AST) -> Optional[list[ast.expr]]:
    """If `node` is torch.cat([...], ...) or x.cat((...)), return the element list."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "cat":
        if node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
            return list(node.args[0].elts)
    return None


def validate_difference_feature_source(source: str) -> dict:
    """Static AST check that `source` implements a feature-level Siamese difference.

    Looks for the first nn.Module subclass with a `forward` method that:
      - applies the SAME `self.<attr>` encoder to >= 2 distinct inputs (shared weights),
      - computes abs(f_L - f_R) where f_L, f_R are that encoder's outputs,
      - concatenates f_L, f_R and the difference together,
      - feeds the concatenation to a head.

    Returns DiffFeatureSourceReport.as_dict(). `ok` is True only when shared encoder,
    feature-level abs difference, and the three-way concat are all present and no
    independent (two-backbone) pattern is detected.
    """
    report = DiffFeatureSourceReport()
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        report.reasons.append(f"syntax error: {exc}")
        return report.as_dict()

    module_classes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and any(_is_module_base(b) for b in node.bases)
    ]
    if not module_classes:
        report.reasons.append("no nn.Module subclass found")
        return report.as_dict()

    for cls in module_classes:
        forward = next(
            (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"),
            None,
        )
        if forward is None:
            continue

        # var name -> encoder attr that produced it (encoder outputs only)
        encoder_outputs: dict[str, str] = {}
        diff_vars: dict[str, tuple[str, str]] = {}  # diff var -> (operand_a, operand_b)
        cat_vars: dict[str, list[ast.expr]] = {}     # cat result var -> element nodes

        for stmt in ast.walk(forward):
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                continue
            tname = target.id
            rhs = stmt.value

            enc = _self_attr_call(rhs)
            if enc is not None:
                encoder_outputs[tname] = enc[0]
                continue

            ops = _abs_sub_operands(rhs)
            if ops is not None:
                a_names = _names_in(ops[0])
                b_names = _names_in(ops[1])
                a = next(iter(a_names), None)
                b = next(iter(b_names), None)
                diff_vars[tname] = (a, b)
                continue

            elts = _cat_elements(rhs)
            if elts is not None:
                cat_vars[tname] = elts

        # Also catch a difference / concat that is returned inline (not assigned).
        for sub in ast.walk(forward):
            ops = _abs_sub_operands(sub)
            if ops is not None:
                a = next(iter(_names_in(ops[0])), None)
                b = next(iter(_names_in(ops[1])), None)
                diff_vars.setdefault(f"_inline_{id(sub)}", (a, b))
            elts = _cat_elements(sub)
            if elts is not None:
                cat_vars.setdefault(f"_inline_{id(sub)}", elts)

        # --- shared encoder: one attr applied to >= 2 distinct inputs ---
        attr_to_vars: dict[str, list[str]] = {}
        for var, attr in encoder_outputs.items():
            attr_to_vars.setdefault(attr, []).append(var)
        shared_attr = next((a for a, vs in attr_to_vars.items() if len(vs) >= 2), None)
        distinct_encoder_attrs = list(attr_to_vars.keys())

        # --- abs diff whose operands are two encoder outputs ---
        feature_diff = None
        for dvar, (a, b) in diff_vars.items():
            if a in encoder_outputs and b in encoder_outputs and a != b:
                feature_diff = (dvar, a, b)
                break

        # --- concat containing both features + the diff ---
        diff_in_concat = False
        concat_var = None
        if feature_diff is not None:
            dvar, fa, fb = feature_diff
            for cvar, elts in cat_vars.items():
                elt_names: set[str] = set()
                for e in elts:
                    elt_names |= _names_in(e)
                if dvar in elt_names and fa in elt_names and fb in elt_names:
                    diff_in_concat = True
                    concat_var = cvar
                    break

        # --- concat result reaches a head (returned or passed into self.<head>) ---
        concat_to_head = False
        if concat_var is not None:
            for sub in ast.walk(forward):
                if isinstance(sub, ast.Call) and any(
                    isinstance(n, ast.Name) and n.id == concat_var for n in ast.walk(sub)
                ):
                    head = _self_attr_call(sub)
                    if head is not None and head[0] != shared_attr:
                        concat_to_head = True
                        break
            for sub in ast.walk(forward):
                if isinstance(sub, ast.Return) and sub.value is not None:
                    if concat_var in _names_in(sub.value):
                        concat_to_head = True

        # Decide for this class.
        this_ok = bool(shared_attr) and feature_diff is not None and diff_in_concat
        if this_ok or shared_attr or feature_diff or diff_in_concat:
            report.model_class = cls.name
            report.encoder_attr = shared_attr
            report.shared_encoder = bool(shared_attr)
            report.independent_backbones = (
                shared_attr is None and len(distinct_encoder_attrs) >= 2
            )
            report.abs_diff_of_features = feature_diff is not None
            report.diff_and_both_features_in_concat = diff_in_concat
            report.concat_passed_to_head = concat_to_head
            if feature_diff is not None:
                report.diff_var = feature_diff[0]
                report.feature_vars = [feature_diff[1], feature_diff[2]]
            if this_ok:
                report.ok = True
                return report.as_dict()

    # Build reasons from the best-matching class we recorded (if any).
    if not report.shared_encoder:
        report.reasons.append(
            "no shared encoder: did not find one `self.<attr>(x)` applied to >= 2 distinct inputs"
        )
    if report.independent_backbones:
        report.reasons.append(
            "independent backbones detected: two different encoder attributes used "
            "for left and right (weights are NOT shared)"
        )
    if not report.abs_diff_of_features:
        report.reasons.append(
            "no feature-level abs difference: did not find abs(f_L - f_R) on two encoder outputs "
            "(note: pixel-level abs(img_l - img_r) does NOT count)"
        )
    if not report.diff_and_both_features_in_concat:
        report.reasons.append(
            "concat does not contain all of [f_L, f_R, |f_L - f_R|]"
        )
    return report.as_dict()


# ---------------------------------------------------------------------------
# 2. Live forward-pass structural check
# ---------------------------------------------------------------------------


def forward_pass_structural_check(
    model: Any,
    dummy_left: Any,
    dummy_right: Any,
    encoder_module: Any = None,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> dict:
    """Run one forward pass and structurally confirm the feature-level difference.

    Args:
        model: an instantiated ``nn.Module`` whose ``forward(left, right)`` returns logits.
        dummy_left / dummy_right: dummy input tensors of the model's expected shape.
        encoder_module: optional explicit reference to the shared encoder submodule
            (e.g. ``model.resnet``). When given, the check verifies that exactly this
            instance is called twice and that one concat input equals ``|f_L - f_R|``.
        rtol / atol: tolerances for the abs-difference content check.

    Returns a dict with:
        ok, shared_encoder, encoder_call_count, feat_dim, concat_width,
        expected_width, width_matches, abs_op_observed, abs_diff_in_concat,
        head_in_features, head_matches, reasons.
    """
    import torch
    import torch.nn as nn

    result: dict[str, Any] = {
        "ok": False,
        "shared_encoder": False,
        "encoder_call_count": 0,
        "feat_dim": None,
        "concat_width": None,
        "expected_width": None,
        "width_matches": False,
        "abs_op_observed": False,
        "abs_diff_in_concat": False,
        "head_in_features": None,
        "head_matches": False,
        "reasons": [],
    }

    # Per-instance call counter over all parametric submodules. If a single instance
    # is called >= 2 times, the encoder weights are shared (the same module processes
    # L and R). Two independent backbones would each be called once.
    call_counts: dict[int, int] = {}
    out_shapes: dict[int, tuple] = {}
    id_to_module: dict[int, Any] = {}
    handles = []

    def make_hook(mod):
        def hook(m, inp, out):
            call_counts[id(m)] = call_counts.get(id(m), 0) + 1
            id_to_module[id(m)] = m
            try:
                out_shapes[id(m)] = tuple(out.shape)
            except Exception:
                pass
        return hook

    for _name, mod in model.named_modules():
        if mod is model:
            continue
        if any(True for _ in mod.parameters(recurse=False)) or mod is encoder_module:
            handles.append(mod.register_forward_hook(make_hook(mod)))

    # Trace torch.abs / Tensor.abs and torch.cat.
    real_abs = torch.abs
    real_tensor_abs = torch.Tensor.abs
    real_cat = torch.cat
    abs_results: list = []
    cat_records: list[dict] = []

    def traced_abs(x, *a, **k):
        out = real_abs(x, *a, **k)
        abs_results.append(out)
        result["abs_op_observed"] = True
        return out

    def traced_tensor_abs(self, *a, **k):
        out = real_tensor_abs(self, *a, **k)
        abs_results.append(out)
        result["abs_op_observed"] = True
        return out

    def traced_cat(tensors, dim=0, out=None):
        res = real_cat(tensors, dim=dim, out=out) if out is not None else real_cat(tensors, dim=dim)
        try:
            inputs = list(tensors)
            cat_records.append({"width": int(res.shape[-1]), "n_inputs": len(inputs), "inputs": inputs})
        except Exception:
            cat_records.append({"width": None, "n_inputs": None, "inputs": []})
        return res

    encoder_outputs: list = []
    enc_handle = None
    if encoder_module is not None:
        def enc_hook(m, inp, out):
            encoder_outputs.append(out)
        enc_handle = encoder_module.register_forward_hook(enc_hook)

    try:
        torch.abs = traced_abs
        torch.Tensor.abs = traced_tensor_abs
        torch.cat = traced_cat
        was_training = model.training
        model.eval()
        with torch.no_grad():
            model(dummy_left, dummy_right)
        if was_training:
            model.train()
    except Exception as exc:  # pragma: no cover - surfaced as a reason
        result["reasons"].append(f"forward pass raised: {exc!r}")
        return _finalize_forward(result, handles, enc_handle,
                                 real_abs, real_tensor_abs, real_cat, torch)
    finally:
        torch.abs = real_abs
        torch.Tensor.abs = real_tensor_abs
        torch.cat = real_cat

    # Shared encoder: explicit module if given, else max call count over submodules.
    if encoder_module is not None:
        ec = call_counts.get(id(encoder_module), 0)
        result["encoder_call_count"] = ec
        result["shared_encoder"] = ec >= 2
        if id(encoder_module) in out_shapes:
            result["feat_dim"] = out_shapes[id(encoder_module)][-1]
    else:
        if call_counts:
            top_id = max(call_counts, key=lambda k: call_counts[k])
            result["encoder_call_count"] = call_counts[top_id]
            result["shared_encoder"] = call_counts[top_id] >= 2
            # feat_dim: a module called >= 2 times producing a 2D (N, C) output.
            twice = [i for i, c in call_counts.items() if c >= 2 and len(out_shapes.get(i, ())) == 2]
            if twice:
                result["feat_dim"] = max(out_shapes[i][-1] for i in twice)

    if not result["shared_encoder"]:
        result["reasons"].append(
            "no submodule instance was called >= 2 times — encoder weights are not shared "
            "(left and right appear to use independent backbones)"
        )

    feat_dim = result["feat_dim"]
    if feat_dim:
        result["expected_width"] = 3 * feat_dim

    # Concat width == 3 * feat_dim and one input equals |f_L - f_R|.
    if cat_records:
        widths = [c["width"] for c in cat_records if c["width"] is not None]
        if result["expected_width"] in widths:
            result["concat_width"] = result["expected_width"]
            result["width_matches"] = True
        elif widths:
            result["concat_width"] = max(widths)

        # Semantic content check: does any concat input equal abs(f_L - f_R)?
        if len(encoder_outputs) >= 2:
            f_l, f_r = encoder_outputs[0], encoder_outputs[1]
            target = (f_l - f_r).abs()
            for rec in cat_records:
                for t in rec["inputs"]:
                    try:
                        if t.shape == target.shape and torch.allclose(t, target, rtol=rtol, atol=atol):
                            result["abs_diff_in_concat"] = True
                            break
                    except Exception:
                        continue
                if result["abs_diff_in_concat"]:
                    break
        else:
            # No encoder reference: fall back to "an abs result of the right width is concatenated".
            for rec in cat_records:
                for t in rec["inputs"]:
                    if any(t is a for a in abs_results):
                        result["abs_diff_in_concat"] = True
                        break

    if not result["width_matches"]:
        result["reasons"].append(
            f"no concat of width 3*feat_dim ({result['expected_width']}) observed; "
            f"saw widths {[c['width'] for c in cat_records]}"
        )
    if not result["abs_op_observed"]:
        result["reasons"].append("no torch.abs / .abs() operation observed during forward")
    if not result["abs_diff_in_concat"]:
        result["reasons"].append("no concat input matched |f_L - f_R|")

    # Classifier head: first nn.Linear whose in_features == 3 * feat_dim.
    if feat_dim:
        for mod in model.modules():
            if isinstance(mod, nn.Linear) and mod.in_features == 3 * feat_dim:
                result["head_in_features"] = mod.in_features
                result["head_matches"] = True
                break
        if not result["head_matches"]:
            linears = [m.in_features for m in model.modules() if isinstance(m, nn.Linear)]
            result["reasons"].append(
                f"no nn.Linear with in_features == 3*feat_dim ({3 * feat_dim}); "
                f"Linear in_features seen: {linears}"
            )

    result["ok"] = bool(
        result["shared_encoder"]
        and result["abs_op_observed"]
        and result["abs_diff_in_concat"]
        and result["width_matches"]
        and result["head_matches"]
    )
    for h in handles:
        h.remove()
    if enc_handle is not None:
        enc_handle.remove()
    return result


def _finalize_forward(result, handles, enc_handle, real_abs, real_tensor_abs, real_cat, torch):
    torch.abs = real_abs
    torch.Tensor.abs = real_tensor_abs
    torch.cat = real_cat
    for h in handles:
        h.remove()
    if enc_handle is not None:
        enc_handle.remove()
    return result
