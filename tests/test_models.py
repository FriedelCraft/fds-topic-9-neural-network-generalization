import unittest

import numpy as np

from topic9.models import gradient_check, initialize, loss_and_grad, metrics
from topic9.training import train


class ModelTests(unittest.TestCase):
    def test_finite_differences_cover_linear_and_nonlinear_models(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(8, 3))
        y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        for width, parameter_count in ((0, 4), (4, 21)):
            for penalty in (0.0, 0.027):
                with self.subTest(width=width, penalty=penalty):
                    params = initialize(3, width, 41)
                    result = gradient_check(params, x, y, weight_decay=penalty)
                    self.assertEqual(result["parameters_checked"], parameter_count)
                    self.assertLess(result["max_abs_error"], 1e-8)
                    self.assertLess(result["max_relative_error"], 1e-5)

    def test_confident_wrong_predictions_have_finite_loss_and_gradient(self):
        params = {"w": np.array([1000.0]), "b": np.array(0.0)}
        x = np.array([[-1.0], [1.0]])
        y = np.array([1, 0])
        loss, grad = loss_and_grad(params, x, y)
        self.assertEqual(loss, 1000.0)
        self.assertEqual(metrics(params, x, y)["accuracy"], 0)
        np.testing.assert_allclose(grad["w"], [1.0])
        self.assertEqual(grad["b"], 0)

    def test_weight_decay_penalizes_weights_only(self):
        params = {"w": np.array([2.0]), "b": np.array(3.0)}
        x = np.array([[-1.0], [1.0]])
        y = np.array([0, 1])
        loss, grad = loss_and_grad(params, x, y)
        penalized_loss, penalized_grad = loss_and_grad(params, x, y, weight_decay=0.5)
        self.assertAlmostEqual(penalized_loss - loss, 1.0)
        np.testing.assert_allclose(penalized_grad["w"] - grad["w"], [1.0])
        self.assertEqual(penalized_grad["b"], grad["b"])


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[-1.0], [1.0]])
        self.y = np.array([0, 1])
        self.params = {"w": np.array([0.0]), "b": np.array(0.0)}

    def test_fixed_budget_counts_updates_and_logs_terminal_iterate(self):
        result = train(
            self.params, self.x, self.y, self.x, self.y,
            learning_rate=0.5, max_steps=5, log_every=2,
        )
        self.assertEqual((result.steps, result.selected_step, result.stop_reason), (5, 5, "budget"))
        self.assertEqual(result.validation_evaluations, 6)
        self.assertEqual([row["step"] for row in result.history], [0, 2, 4, 5])
        self.assertLess(result.history[-1]["train_loss"], result.history[0]["train_loss"])
        self.assertLess(gradient_check(result.model, self.x, self.y)["max_abs_error"], 1e-8)
        np.testing.assert_array_equal(self.params["w"], [0.0])

    def test_patience_restores_validation_checkpoint(self):
        result = train(
            self.params, self.x, self.y, self.x, 1 - self.y,
            learning_rate=0.5, max_steps=20, patience=3,
        )
        self.assertEqual((result.steps, result.selected_step, result.stop_reason), (3, 0, "early_stopping"))
        self.assertGreater(result.history[-1]["val_loss"], result.history[0]["val_loss"])
        self.assertLess(result.final_gradient_norm, result.history[0]["gradient_norm"])
        np.testing.assert_array_equal(result.model["w"], self.params["w"])

    def test_gradient_rule_can_stop_before_first_update(self):
        result = train(
            self.params, np.zeros_like(self.x), self.y, self.x, self.y,
            learning_rate=0.5, max_steps=10, gradient_tol=1e-8,
        )
        self.assertEqual((result.steps, result.stop_reason), (0, "converged"))
        self.assertEqual(result.final_gradient_norm, 0)


if __name__ == "__main__":
    unittest.main()
