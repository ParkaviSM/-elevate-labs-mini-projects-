import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
real["label"] = 1

# Combine
data = pd.concat([fake, real])

# Shuffle
data = data.sample(frac=1, random_state=42)

# Split input/output
X = data["text"]
y = data["label"]

# Vectorize
vectorizer = TfidfVectorizer()
X_vector = vectorizer.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vector, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Save model
pickle.dump(model, open("model/model.pkl", "wb"))

# Save vectorizer
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("Model saved successfully")