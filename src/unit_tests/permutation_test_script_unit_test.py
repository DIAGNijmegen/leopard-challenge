import unittest
import numpy as np
import pandas as pd
import yaml
from io import StringIO



class TestSurvivalAnalysisUtils(unittest.TestCase):

    def test_zscore_normalize_dict(self):
        data = {'a': 1, 'b': 2, 'c': 3}
        result = zscore_normalize_dict(data)
        self.assertAlmostEqual(result['a'], -1.0)
        self.assertAlmostEqual(result['b'], 0.0)
        self.assertAlmostEqual(result['c'], 1.0)

    def test_zscore_normalize_dict_zero_std(self):
        data = {'a': 5, 'b': 5, 'c': 5}
        result = zscore_normalize_dict(data)
        self.assertEqual(result['a'], 0)
        self.assertEqual(result['b'], 0)
        self.assertEqual(result['c'], 0)

    def test_invert_pred_dict(self):
        data = {'x': 2.5, 'y': -1.0}
        result = invert_pred_dict(data)
        self.assertEqual(result['x'], -2.5)
        self.assertEqual(result['y'], 1.0)

    def test_bootstrap_c_index(self):
        events = np.array([1, 0, 1, 1])
        times = np.array([10, 12, 9, 15])
        preds = np.array([0.2, 0.3, 0.1, 0.4])
        c_index, ci_l, ci_u = bootstrap_c_index(events, times, preds, n_bootstraps=3)
        self.assertTrue(0 <= c_index <= 1)
        self.assertTrue(ci_l <= c_index <= ci_u)

    def test_calculate_p_value_permutation(self):
        events = np.array([1, 0, 1, 1])
        times = np.array([10, 12, 9, 15])
        preds1 = np.array([0.1, 0.4, 0.3, 0.2])
        preds2 = np.array([0.2, 0.3, 0.2, 0.1])
        diff, p, null_distr = calculate_p_value_permutation(events, times, preds1, preds2, n_permutations=3)
        self.assertTrue(-1 <= diff <= 1)
        self.assertTrue(0 <= p <= 1)
        self.assertEqual(len(null_distr), 3)

    def test_load_config(self):
        fake_yaml = """
        input_dir: /tmp/input
        ensemble_teams: [team1, team2]
        datasets:
          - dataset1: 10
        output_dir: /tmp/output
        clinical_variables: /tmp/clinical
        """
        f = StringIO(fake_yaml)
        config = yaml.safe_load(f)
        self.assertIn("input_dir", config)
        self.assertIn("ensemble_teams", config)
        self.assertEqual(config["datasets"][0]['dataset1'], 10)


# Run the tests
unittest.main(argv=[''], verbosity=2, exit=False)
