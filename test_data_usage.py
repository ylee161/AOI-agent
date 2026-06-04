"""Tests for the CHECK 3 Data Usage audit (mle_star_agent.shared.data_usage_validator).

The audit statically confirms a training script USES the declared data modalities:
both stereo images, the Excel labels, and the provided split. It must produce zero
false positives on the legitimate cases (mono, the no_stereo_fusion ablation) and
must never crash on unparseable code.
"""

import os
import unittest

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

from mle_star_agent.shared.data_usage_validator import validate_data_usage_source


# A complete-enough stereo script: loads both _L and _R, reads the Excel labels,
# and reads the provided split.
_STEREO_OK = '''
import pandas as pd
labels = pd.read_excel("results.xlsx")
split = load_json("checkpoints/data_split_grouped.json")
train_paths = split["train"]
img_l = load_image(path.replace("_R_", "_L_"))
img_r = load_image(path.replace("_L_", "_R_"))
'''

# Stereo script that loads ONLY the left image.
_STEREO_LEFT_ONLY = '''
import pandas as pd
labels = pd.read_excel("results.xlsx")
split = load_json("data_split.json")
img_l = load_image(path_for("_L_"))
'''

# Same left-only pattern, but declared as the no_stereo_fusion ablation.
_STEREO_LEFT_ONLY_ABLATION = '''
ABLATION_VARIANT_NAME = "no_stereo_fusion"
import pandas as pd
labels = pd.read_excel("results.xlsx")
split = load_json("data_split.json")
img_l = load_image(path_for("_L_"))
'''

# Mono script: a single image per cell, no _R anywhere.
_MONO_OK = '''
import pandas as pd
labels = pd.read_excel("results.xlsx")
split = load_json("data_split.json")
img = load_image(sample_path)
'''

# Stereo script that loads both images but never reads the Excel labels.
_NO_EXCEL = '''
split = load_json("data_split.json")
img_l = load_image(path.replace("_R_", "_L_"))
img_r = load_image(path.replace("_L_", "_R_"))
'''

_UNPARSEABLE = '''
def f(:
    this is not valid python @#$%
'''


class DataUsageAuditTests(unittest.TestCase):
    def test_a_stereo_left_only_is_flagged(self):
        report = validate_data_usage_source(_STEREO_LEFT_ONLY, input_modality="stereo")
        self.assertFalse(report["ok"])
        self.assertIn("right_image_unused", report["reasons"])
        self.assertIn("stereo input but right image unused", report["messages"])

    def test_b_no_stereo_fusion_ablation_is_not_flagged(self):
        report = validate_data_usage_source(
            _STEREO_LEFT_ONLY_ABLATION, input_modality="stereo"
        )
        self.assertEqual(report["ablation_variant_name"], "no_stereo_fusion")
        self.assertNotIn("right_image_unused", report["reasons"])
        self.assertNotIn("stereo input but right image unused", report["messages"])
        # Everything else is loaded, so the ablation should pass cleanly.
        self.assertTrue(report["ok"])

    def test_c_mono_single_image_is_not_flagged(self):
        report = validate_data_usage_source(_MONO_OK, input_modality="mono")
        self.assertNotIn("right_image_unused", report["reasons"])
        self.assertTrue(report["ok"])

    def test_d_missing_read_excel_is_flagged_labels_not_loaded(self):
        report = validate_data_usage_source(_NO_EXCEL, input_modality="stereo")
        self.assertFalse(report["ok"])
        self.assertIn("labels_not_loaded", report["reasons"])
        self.assertIn("labels not loaded", report["messages"])

    def test_e_unparseable_code_is_inconclusive_not_a_crash(self):
        report = validate_data_usage_source(_UNPARSEABLE, input_modality="stereo")
        self.assertTrue(report["inconclusive"])
        self.assertEqual(report["reasons"], [])
        self.assertIn("usage check inconclusive", report["messages"])
        # Inconclusive must never be a violation.
        self.assertTrue(report["ok"])

    def test_complete_stereo_script_passes(self):
        report = validate_data_usage_source(_STEREO_OK, input_modality="stereo")
        self.assertTrue(report["ok"], report["messages"])
        self.assertEqual(report["reasons"], [])

    def test_ignores_provided_split_is_flagged(self):
        script = '''
import pandas as pd
labels = pd.read_excel("results.xlsx")
img_l = load_image(path.replace("_R_", "_L_"))
img_r = load_image(path.replace("_L_", "_R_"))
'''
        report = validate_data_usage_source(script, input_modality="stereo")
        self.assertFalse(report["ok"])
        self.assertIn("ignores_provided_split", report["reasons"])
        self.assertIn("ignores provided split", report["messages"])


if __name__ == "__main__":
    unittest.main()
