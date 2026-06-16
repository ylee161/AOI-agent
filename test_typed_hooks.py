"""Tests for the typed, dataset-agnostic hook system in shared/script_template.py.

The historical failure mode was the LLM free-writing the "architecture block"
(model + optimizer + loss). It is now composed from *typed switches* that each
resolve to a pre-written code path, with the AOI-specific switches
(VIEW_FUSION_MODE, GROUP_ROBUST_MODE) capability-gated so the repo stays a clean,
dataset-agnostic pipeline.

Coverage:
  (a) Capability gating — stereo / group options are offered only when the
      dataset descriptor supports them, and are dormant otherwise.
  (b) AOI-ON config (HAS_STEREO_VIEWS=True, GROUP_COLUMN set) builds a valid
      torch model + loss from the resolved block — no placeholder, no free-write.
  (c) GENERIC-OFF config (HAS_STEREO_VIEWS=False, GROUP_COLUMN=None) falls back to
      a single-image model + plain validation, and still builds a valid model +
      loss with no stereo / group code present.
  (d) The Siamese FEATURE_DIFF_CANDIDATE path still builds and forwards a pair.
  (e) Full rendered scripts (free-write and every typed config) compile, and the
      typed loss switch is threaded through the two former hard-wired BCE sites.

The resolved architecture block is exec'd in a namespace mirroring the names the
template provides (device, pos_weight, nn, optim, ...) and we then call the
generated build_model()/build_criterion()/build_optimizer() — i.e. we prove the
switches yield a working model + loss with zero LLM involvement.
"""

import os
import unittest

import torch
import torch.nn as nn
import torch.optim as optim

from mle_star_agent.shared.script_template import (
    DatasetCapabilities,
    HookConfig,
    LOSS_MENU,
    OPTIMIZER_MENU,
    STEREO_VIEW_FUSION_MODES,
    GROUP_ROBUST_MODES,
    build_architecture_block,
    get_script_template,
    offered_view_fusion_modes,
    offered_group_robust_modes,
    resolve_view_fusion,
    resolve_group_robust,
)

# Build models with random weights so the test never hits the network.
os.environ.setdefault("AOI_PRETRAINED", "0")

# AOI dataset: stereo pairs + a board grouping column.
AOI_CAPS = DatasetCapabilities(HAS_STEREO_VIEWS=True, GROUP_COLUMN="board")
# Generic dataset cloned by a new user: single image, no group column.
GENERIC_CAPS = DatasetCapabilities(HAS_STEREO_VIEWS=False, GROUP_COLUMN=None)


def _exec_block(cfg: HookConfig, caps: DatasetCapabilities) -> dict:
    """Exec the resolved architecture block in a namespace mirroring the template
    prerequisites and return the populated namespace."""
    block = build_architecture_block(cfg, caps)
    # The block must be self-contained Python (this is what replaces free-writing).
    compile(block, "<architecture_block>", "exec")
    ns = {
        "os": os,
        "torch": torch,
        "nn": nn,
        "optim": optim,
        "device": torch.device("cpu"),
        # pos_weight = n_g / n_ng, exactly as the template computes it.
        "pos_weight": torch.tensor([3.0]),
        "DRY_RUN": True,
        "DRY_RUN_EPOCHS": 1,
    }
    exec(block, ns)
    return ns


def _forward_logits(ns: dict, model) -> torch.Tensor:
    """Drive a forward pass that matches the resolved view-fusion mode."""
    in_channels = ns["IN_CHANNELS"]
    if ns["FEATURE_DIFF_CANDIDATE"]:
        img_l = torch.randn(2, 3, 64, 64)
        img_r = torch.randn(2, 3, 64, 64)
        return model(img_l, img_r).view(-1)
    image = torch.randn(2, in_channels, 64, 64)
    return model(image).view(-1)


# ---------------------------------------------------------------------------
# (a) Capability gating
# ---------------------------------------------------------------------------
class CapabilityGatingTests(unittest.TestCase):
    def test_stereo_views_offered_only_when_capable(self):
        self.assertEqual(offered_view_fusion_modes(AOI_CAPS), list(STEREO_VIEW_FUSION_MODES))
        # No stereo views -> only the single-image fallback is offered.
        self.assertEqual(offered_view_fusion_modes(GENERIC_CAPS), ["single"])

    def test_group_modes_offered_only_when_group_column_set(self):
        self.assertEqual(offered_group_robust_modes(AOI_CAPS), list(GROUP_ROBUST_MODES))
        self.assertEqual(offered_group_robust_modes(GENERIC_CAPS), ["off"])

    def test_view_fusion_falls_back_to_single_without_stereo(self):
        # Even when the caller *requests* a stereo mode, no stereo views => single.
        view = resolve_view_fusion(HookConfig(VIEW_FUSION_MODE="concat_diff"), GENERIC_CAPS)
        self.assertEqual(view.mode, "single")
        self.assertEqual(view.in_channels, 3)
        self.assertFalse(view.feature_diff)

    def test_group_robust_disabled_without_group_column(self):
        # Even when the caller requests group_dro, no GROUP_COLUMN => disabled.
        code = resolve_group_robust(HookConfig(GROUP_ROBUST_MODE="group_dro"), GENERIC_CAPS)
        self.assertIn("GROUP_ROBUST_ACTIVE = False", code)
        self.assertIn("GROUP_COLUMN = None", code)

    def test_group_robust_active_when_group_column_set(self):
        code = resolve_group_robust(HookConfig(GROUP_ROBUST_MODE="group_dro"), AOI_CAPS)
        self.assertIn("GROUP_ROBUST_ACTIVE = True", code)
        self.assertIn("GROUP_COLUMN = 'board'", code)

    def test_universal_switches_always_available(self):
        # LOSS_MODE / OPTIMIZER_MODE are dataset-agnostic and never gated.
        self.assertEqual(set(LOSS_MENU), {"bce", "focal", "logit_adjust"})
        self.assertEqual(set(OPTIMIZER_MENU), {"adamw", "sgd", "adam"})


# ---------------------------------------------------------------------------
# (b) AOI-ON: builds a valid model + loss with no free-writing
# ---------------------------------------------------------------------------
class AoiOnBuildTests(unittest.TestCase):
    def test_aoi_on_builds_model_and_loss(self):
        cfg = HookConfig(
            LOSS_MODE="focal",
            OPTIMIZER_MODE="adamw",
            VIEW_FUSION_MODE="concat_diff",
            GROUP_ROBUST_MODE="group_dro",
        )
        ns = _exec_block(cfg, AOI_CAPS)

        # Stereo concat path: 9-channel single-tensor model.
        self.assertEqual(ns["IN_CHANNELS"], 9)
        self.assertFalse(ns["FEATURE_DIFF_CANDIDATE"])
        self.assertTrue(ns["GROUP_ROBUST_ACTIVE"])

        model = ns["build_model"]()
        self.assertIsInstance(model, nn.Module)
        logits = _forward_logits(ns, model)
        self.assertEqual(logits.shape, (2,))

        criterion = ns["build_criterion"]()
        labels = torch.tensor([0.0, 1.0])
        loss = criterion(logits, labels)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

        # Optimizer / scheduler hooks are wired and usable.
        optimizer = ns["build_optimizer"](model)
        scheduler = ns["build_scheduler"](optimizer)
        loss.backward()
        optimizer.step()
        scheduler.step()

    def test_every_loss_and_optimizer_mode_builds(self):
        for loss_mode in LOSS_MENU:
            for opt_mode in OPTIMIZER_MENU:
                cfg = HookConfig(LOSS_MODE=loss_mode, OPTIMIZER_MODE=opt_mode)
                ns = _exec_block(cfg, AOI_CAPS)
                model = ns["build_model"]()
                logits = _forward_logits(ns, model)
                loss = ns["build_criterion"]()(logits, torch.tensor([0.0, 1.0]))
                self.assertTrue(
                    torch.isfinite(loss),
                    f"{loss_mode}/{opt_mode} produced non-finite loss",
                )
                # Optimizer actually constructs over the model parameters.
                self.assertGreater(len(ns["build_optimizer"](model).param_groups), 0)


# ---------------------------------------------------------------------------
# (c) GENERIC-OFF: single-image fallback, plain validation
# ---------------------------------------------------------------------------
class GenericOffBuildTests(unittest.TestCase):
    def test_generic_off_builds_single_image_model_and_loss(self):
        # Request the AOI options explicitly; they must stay dormant.
        cfg = HookConfig(
            LOSS_MODE="bce",
            OPTIMIZER_MODE="sgd",
            VIEW_FUSION_MODE="siamese_feature_diff",
            GROUP_ROBUST_MODE="group_dro",
        )
        block = build_architecture_block(cfg, GENERIC_CAPS)
        # AOI-specific code is absent — dormant when the capability is off.
        self.assertNotIn("SiameseFeatureDiff", block)
        self.assertNotIn("GROUP_ROBUST_ACTIVE = True", block)
        self.assertIn("IN_CHANNELS = 3", block)

        ns = _exec_block(cfg, GENERIC_CAPS)
        self.assertEqual(ns["IN_CHANNELS"], 3)
        self.assertFalse(ns["FEATURE_DIFF_CANDIDATE"])
        self.assertFalse(ns["GROUP_ROBUST_ACTIVE"])
        self.assertIsNone(ns["GROUP_COLUMN"])

        model = ns["build_model"]()
        self.assertIsInstance(model, nn.Module)
        logits = _forward_logits(ns, model)
        self.assertEqual(logits.shape, (2,))
        loss = ns["build_criterion"]()(logits, torch.tensor([1.0, 0.0]))
        self.assertTrue(torch.isfinite(loss))

    def test_group_objective_falls_back_to_plain_validation(self):
        ns = _exec_block(HookConfig(), GENERIC_CAPS)
        # Plain validation: the objective is just the global val loss.
        self.assertEqual(ns["group_robust_objective"](0.42), 0.42)


# ---------------------------------------------------------------------------
# (d) Siamese FEATURE_DIFF_CANDIDATE path still works
# ---------------------------------------------------------------------------
class SiamesePathTests(unittest.TestCase):
    def test_siamese_builds_and_forwards_pair(self):
        cfg = HookConfig(VIEW_FUSION_MODE="siamese_feature_diff")
        ns = _exec_block(cfg, AOI_CAPS)
        self.assertTrue(ns["FEATURE_DIFF_CANDIDATE"])
        model = ns["build_model"]()
        logits = _forward_logits(ns, model)  # forwards (img_l, img_r)
        self.assertEqual(logits.shape, (2,))
        loss = ns["build_criterion"]()(logits, torch.tensor([0.0, 1.0]))
        self.assertTrue(torch.isfinite(loss))


# ---------------------------------------------------------------------------
# (e) Full rendered scripts compile; loss switch threaded
# ---------------------------------------------------------------------------
class RenderedScriptTests(unittest.TestCase):
    def test_freewrite_script_still_renders_and_compiles(self):
        script = get_script_template("/tmp/split.json")
        compile(script, "<freewrite>", "exec")
        # The free-write placeholder is preserved (backward compatible).
        self.assertIn("ARCHITECTURE BLOCK START", script)
        # The criterion is resolved through the threading shim.
        self.assertIn("_resolve_criterion()", script)

    def test_typed_scripts_render_compile_and_thread_loss(self):
        for view_mode in STEREO_VIEW_FUSION_MODES:
            for loss_mode in LOSS_MENU:
                cfg = HookConfig(LOSS_MODE=loss_mode, VIEW_FUSION_MODE=view_mode)
                script = get_script_template(
                    "/tmp/split.json", hook_config=cfg, capabilities=AOI_CAPS
                )
                compile(script, f"<{view_mode}-{loss_mode}>", "exec")
                # No free-writing placeholder remains.
                self.assertNotIn("ARCHITECTURE BLOCK START", script)
                self.assertIn("TYPED HOOK BLOCK", script)
                # The chosen loss is wired and reached via the threaded shim.
                self.assertIn("def build_criterion", script)
                self.assertIn("criterion = _resolve_criterion()", script)

    def test_typed_render_rejects_non_stereo_dataset(self):
        # The stereo template cannot render a single-image dataset; the caller is
        # told to compose against a single-image template instead.
        with self.assertRaises(ValueError):
            get_script_template(
                "/tmp/split.json",
                hook_config=HookConfig(),
                capabilities=GENERIC_CAPS,
            )


if __name__ == "__main__":
    unittest.main()
