import math
import unittest

from wandb_tail import wandb_payload


class WandbPayloadTests(unittest.TestCase):
    def test_layer_norms_become_scalar_series(self):
        original = {"step": 7, "loss": 1.2, "layer_grad_norms": [0.4, 0.8]}
        payload = wandb_payload(original)
        self.assertNotIn("layer_grad_norms", payload)
        self.assertEqual(payload["layer_grad_norm/layer_1"], 0.4)
        self.assertEqual(payload["layer_grad_norm/layer_2"], 0.8)
        self.assertIn("layer_grad_norms", original)

    def test_non_finite_layer_norm_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "not finite"):
            wandb_payload({"step": 1, "layer_grad_norms": [math.nan]})


if __name__ == "__main__":
    unittest.main()
