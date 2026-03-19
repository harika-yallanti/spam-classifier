import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Step 1: Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Step 2: Keep only required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Remove extra spaces
df['label'] = df['label'].str.strip()

# Step 3: Convert labels (ham=0, spam=1)
df.loc[:,'label'] = df['label'].map({'ham': 0, 'spam': 1}).astype(int)

df = df.dropna(subset=['label'])

# Convert to integer
df['label'] = df['label'].astype(int)

# Step 4: Split data
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Convert text → numbers using TF-IDF
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    ngram_range=(1,2)
    )

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(df['label'].dtype)

print("Unique labels:", df['label'].unique())
print("Data type:", df['label'].dtype)

# Step 6: Train model
model = MultinomialNB(class_prior=[0.4, 0.6])
model.fit(X_train_tfidf, y_train)

# Step 7: Test model
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred, zero_division=0))


# Step 8: Predict new messages
def predict_spam(message):
    message_tfidf = vectorizer.transform([message])
    result = model.predict(message_tfidf)[0]
    return "Spam" if result == 1 else "Ham"


# Try your own messages
print("\nCustom Predictions:")
print(predict_spam("Congratulations! You won a free ticket!"))
print(predict_spam("Hey, are we meeting today?"))
print(predict_spam("WINNER!! Claim your prize now"))
print(predict_spam("Free entry in a contest"))
print(predict_spam("Let's go for lunch"))

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model saved successfully!")