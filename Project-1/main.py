import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
real["label"] = 1


data = pd.concat([fake, real])

# Input and output
X = data["text"]
y = data["label"]
vectorizer = TfidfVectorizer()
X_vector = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vector, y)


news = ["India launched Chandrayaan successfully"]

news_vector = vectorizer.transform(news)

prediction = model.predict(news_vector)

if prediction[0] == 1:
    print("Real News")
else:
    print("Fake News")