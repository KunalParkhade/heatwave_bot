import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import os

class IntentClassifier:
    def __init__(self, training_data_path):
        self.training_data_path = training_data_path
        self.vectorizer = CountVectorizer()
        self.label_encoder = LabelEncoder()
        self.model = MultinomialNB()
        self.intents = []
        self._load_training_data()

    def _load_training_data(self):
        """Load and preprocess training data."""
        with open(self.training_data_path, "r") as f:
            training_data = json.load(f)

        sentences = []
        labels = []

        for intent, examples in training_data["intents"].items():
            # self.intents.append(intent)
            sentences.extend(examples)  # Flatten the list of sentences
            labels.extend([intent] * len(examples))

        # Fit the vectorizer and model
        print("Sentences to be vectorized:", sentences)
        print("Labels: ", labels)
        X = self.vectorizer.fit_transform(sentences)
        y = self.label_encoder.fit_transform(labels)
        self.model.fit(X, y)

    def predict_intent(self, user_query):
        """Predict the intent of a user's query."""
        X = self.vectorizer.transform([user_query])
        intent = self.model.predict(X)[0]
        return intent
    

# Testing the IntentClassifier
if __name__ == "__main__":
    training_data_path = os.path.join("data", "training_data.json")
    classifier = IntentClassifier(training_data_path)

    # Test queries
    test_queries = [
        "What are the symptoms of heatstroke?",
        "Where can I find cooling shelters?",
        "Give me some safety tips for heatwaves."
    ]

    for query in test_queries:
        print(f"Query: {query}")
        print(f"Predicted Intent: {classifier.predict_intent(query)}")
        print("-" * 50)