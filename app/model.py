"""
Model wrapper — isolates the HuggingFace pipeline so it can be swapped easily.
Changing the model only requires updating MODEL_NAME here.
"""

from transformers import pipeline

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


class SentimentModel:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._pipe = pipeline(
            "text-classification",
            model=model_name,
            truncation=True,
            max_length=512,
        )

    def predict(self, text: str) -> dict:
        """
        Run inference on a single text string.
        Returns {'label': 'POSITIVE'|'NEGATIVE', 'score': float}.
        """
        results = self._pipe(text)
        return results[0]   # pipeline always returns a list; we send one item