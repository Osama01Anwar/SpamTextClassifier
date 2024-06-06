import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib
import tkinter as tk
from tkinter import messagebox

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load the dataset
df = pd.read_csv('spam_large.csv')

# Preprocess the text data
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

df['message'] = df['message'].apply(preprocess)
df['message'] = df['message'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Convert text data to numerical data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# Predict the labels for the test set
y_pred = clf.predict(X_test_vec)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
accuracy_percentage = accuracy * 100

# Save the model and vectorizer
joblib.dump(clf, 'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Load the model and vectorizer
clf = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to predict if a message is spam or ham
def predict_message(message):
    message = preprocess(message)
    message = ' '.join([word for word in word_tokenize(message) if word not in stop_words])
    message_vec = vectorizer.transform([message])
    prediction = clf.predict(message_vec)
    return prediction[0]

# Create the UI using tkinter
def classify_message():
    message = entry.get()
    if message:
        prediction = predict_message(message)
        messagebox.showinfo("Prediction", f'The message is classified as: {prediction}\nModel Accuracy: {accuracy_percentage:.2f}%')
    else:
        messagebox.showwarning("Input Error", "Please enter a message to classify.")

# Set up the main application window
root = tk.Tk()
root.title("Spam Classifier")
root.geometry("400x200")

# Add a label and text entry widget
label = tk.Label(root, text="Enter a message to classify as spam or ham:")
label.pack(pady=10)
entry = tk.Entry(root, width=50)
entry.pack(pady=10)

# Add a button to classify the message
button = tk.Button(root, text="Classify", command=classify_message)
button.pack(pady=10)

# Start the tkinter main loop
root.mainloop()
