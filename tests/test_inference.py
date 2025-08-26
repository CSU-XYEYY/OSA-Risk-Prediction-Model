import unittest
import pandas as pd
from src.RF_inference_Regression import load_and_preprocess_data, predict_and_output

class TestInference(unittest.TestCase):

    def setUp(self):
        self.test_file_path = 'path/to/test_data.csv'  # Update with actual test data path
        self.model_filename = 'models/model_fold_5.pkl'

    def test_load_and_preprocess_data(self):
        data = load_and_preprocess_data(self.test_file_path)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.isnull().values.any(), "Data contains null values after preprocessing")

    def test_predict_and_output(self):
        # Here we will test if the predict_and_output function runs without errors
        try:
            predict_and_output(self.test_file_path, self.model_filename)
        except Exception as e:
            self.fail(f"predict_and_output raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()