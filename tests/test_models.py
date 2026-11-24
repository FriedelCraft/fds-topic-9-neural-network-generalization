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

    def test_gradients_at_project_widths_with_nonzero_biases(self):
        rng = np.random.default_rng(123)
        y = np.array([0, 1, 1, 0, 1, 0, 1])
        for features in (2, 30):
            x = rng.normal(size=(len(y), features))
            for width in (0, 2, 8, 32):
                params = initialize(features, width, 67)
                for key in params:
                    if key.startswith("b"):
                        params[key][...] = rng.normal(scale=0.2, size=params[key].shape)
                original = {key: value.copy() for key, value in params.items()}
                count = features + 1 if width == 0 else width * (features + 2) + 1
                for decay in (0.0, 0.005):
                    with self.subTest(features=features, width=width, decay=decay):
                        result = gradient_check(params, x, y, weight_decay=decay)
                        self.assertEqual(result["parameters_checked"], count)
                        self.assertLess(result["max_abs_error"], 1e-8)
                        for key in params:
                            np.testing.assert_array_equal(params[key], original[key])

    def test_one_hidden_unit_backprop_by_hand(self):
        x = np.array([[1.0, 2.0]])
        y = np.array([1])
        params = {
            "w1": np.array([[0.25], [-0.125]]),
            "b1": np.array([np.arctanh(0.5)]),
            "w2": np.array([2.0]),
            "b2": np.array(-1.0),
        }
        # Hidden activation = 0.5, logit = 0, probability = 0.5.
        # Output delta = -0.5 and hidden delta = -0.5 * 2 * (1 - 0.5**2).
        loss, grad = loss_and_grad(params, x, y, weight_decay=0.2)
        self.assertAlmostEqual(loss, np.log(2) + 0.1 * (0.25**2 + 0.125**2 + 2**2))
        self.assertAlmostEqual(metrics(params, x, y)["loss"], np.log(2))
        self.assertEqual(metrics(params, x, y)["accuracy"], 1.0)
        np.testing.assert_allclose(grad["w1"], [[-0.70], [-1.525]])
        np.testing.assert_allclose(grad["b1"], [-0.75])
        np.testing.assert_allclose(grad["w2"], [0.15])
        self.assertAlmostEqual(float(grad["b2"]), -0.5)

    def test_gradient_check_rejects_nonfinite_values(self):
        x = np.array([[1.0]])
        y = np.array([0])
        params = initialize(1, 0, 67)
        for epsilon in (0, -1e-5, np.nan, np.inf):
            with self.subTest(epsilon=epsilon):
                with self.assertRaises(ValueError):
                    gradient_check(params, x, y, epsilon=epsilon)
        params["w"][0] = np.nan
        with np.errstate(invalid="ignore"):
            with self.assertRaises(FloatingPointError):
                gradient_check(params, x, y)

    def test_confident_wrong_predictions_have_finite_loss_and_gradient(self):
        params = {"w": np.array([1000.0]), "b": np.array(0.0)}
        x = np.array([[-1.0], [1.0]])
        y = np.array([1, 0])
        loss, grad = loss_and_grad(params, x, y)
        self.assertEqual(loss, 1000.0)
        self.assertEqual(metrics(params, x, y)["accuracy"], 0)
        np.testing.assert_allclose(grad["w"], [1.0])
        self.assertEqual(grad["b"], 0)

    def test_confident_correct_predictions_keep_small_gradients(self):
        x = np.array([[1.0]])
        small = np.exp(-40.0)
        for label, logit in ((1, 40.0), (0, -40.0)):
            with self.subTest(label=label):
                params = {"w": np.array([logit]), "b": np.array(0.0)}
                y = np.array([label])
                loss, grad = loss_and_grad(params, x, y)
                expected = (1 - 2 * label) * small / (1 + small)
                # Zero absolute tolerance also catches cancellation to zero.
                np.testing.assert_allclose(loss, np.log1p(small), rtol=1e-12, atol=0)
                np.testing.assert_allclose(grad["w"], [expected], rtol=1e-12, atol=0)
                np.testing.assert_allclose(grad["b"], expected, rtol=1e-12, atol=0)
                self.assertEqual(metrics(params, x, y)["accuracy"], 1.0)

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
