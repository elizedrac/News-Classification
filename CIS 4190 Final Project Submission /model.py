"""model.py

Linear SVM model for news source classification (FoxNews vs NBC).
Uses TF-IDF vectorization and scikit-learn LinearSVC.
"""

import os
from typing import Any, Iterable, List
import joblib


class Model:
    """
    Linear SVM model for news source classification.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the model and load pretrained weights (from weights folder)
        (Implementation below corresponds to a 'weights' folder located in the root)
        """
        # TODO: Add here the path to our weights (submitted .zip folder through gradescope)
        model_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct paths to weight files
        vectorizer_path = os.path.join(model_dir, "weights", "tfidf_vectorizer.joblib")
        model_path = os.path.join(model_dir, "weights", "best_svm_model.pkl")
        
        # Load the pretrained vectorizer and classifier
        self.vectorizer = joblib.load(vectorizer_path)
        self.clf = joblib.load(model_path)

    def eval(self) -> None:
        """
        Set model to evaluation mode.
        For sklearn models, this is a no-op but required by the interface.
        """
        pass

    def predict(self, batch: Iterable[Any]) -> List[Any]:
        """
        Predict news source for a batch of preprocessed headlines.
        
        Inputs:
            batch: Iterable of preprocessed input strings (produced by preprocess.py)
        Returns:
            predictions list (0=FoxNews, 1=NBC) with the same length as `batch`.
        """
        # Convert batch to list if it's not already
        batch_list = list(batch)
        
        # Transform text to TF-IDF features
        X = self.vectorizer.transform(batch_list)
        
        # Predict and return as list
        return self.clf.predict(X).tolist()


def get_model() -> Model:
    """
    Factory function required by the evaluator.
    Returns a model instance with pretrained weights loaded.
    """
    return Model()