import math
import unittest

from wandb_tail import remote_sequence_floor, wandb_payload


class WandbPayloadTests(unittest.TestCase):
    @staticmethod
    def record(values):
        return {
            "schema_version": 2,
            "sequence": 11,
            "global_step": 7,
            "phase": {"index": 1, "name": "sft", "kind": "sft"},
            "event": {"type": "optimization", "values": values},
        }

    def test_layer_norms_become_scalar_series(self):
        original = self.record({"loss": 1.2, "layer_gradient_norms": [0.4, 0.8]})
        payload = wandb_payload(original)
        self.assertNotIn("layer_gradient_norms", payload)
        self.assertEqual(payload["layer_grad_norm/layer_1"], 0.4)
        self.assertEqual(payload["layer_grad_norm/layer_2"], 0.8)
        self.assertIn("layer_gradient_norms", original["event"]["values"])
        self.assertEqual(payload["global_step"], 7)
        self.assertEqual(payload["metric_sequence"], 11)
        self.assertEqual(payload["phase/name"], "sft")

    def test_non_finite_layer_norm_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "not finite"):
            wandb_payload(self.record({"layer_gradient_norms": [math.nan]}))

    def test_old_untyped_rows_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "schema_version"):
            wandb_payload({"step": 1, "loss": 2.0})

    def test_invalid_coordinates_are_rejected(self):
        record = self.record({"loss": 1.0})
        record["sequence"] = True
        with self.assertRaisesRegex(ValueError, "sequence"):
            wandb_payload(record)

        record = self.record({"loss": 1.0})
        record["phase"]["name"] = ""
        with self.assertRaisesRegex(ValueError, "phase"):
            wandb_payload(record)

    def test_event_values_cannot_override_coordinates(self):
        payload = wandb_payload(
            self.record({"global_step": 999, "metric_sequence": 999})
        )
        self.assertEqual(payload["global_step"], 7)
        self.assertEqual(payload["metric_sequence"], 11)

    def test_post_training_update_is_flattened_without_schema_guessing(self):
        record = self.record(
            {
                "transaction_id": "sha256:" + "a" * 64,
                "algorithm": "grpo",
                "optimizer_step": 9,
                "loss": -0.25,
                "mean_reward": 0.75,
            }
        )
        record["event"]["type"] = "post_training_update"
        payload = wandb_payload(record)
        self.assertEqual(payload["event_type"], "post_training_update")
        self.assertEqual(payload["algorithm"], "grpo")
        self.assertEqual(payload["optimizer_step"], 9)
        self.assertEqual(payload["mean_reward"], 0.75)

    def test_new_remote_run_keeps_sequence_zero(self):
        self.assertEqual(remote_sequence_floor(None), -1)
        self.assertEqual(remote_sequence_floor(-1), -1)
        self.assertEqual(remote_sequence_floor(0), 0)
        with self.assertRaisesRegex(ValueError, "lastHistoryStep"):
            remote_sequence_floor(True)


if __name__ == "__main__":
    unittest.main()
