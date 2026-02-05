import nltk
import random
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required data
nltk.download('punkt')

# Knowledge base (you can add more lines)
corpus = [
    "Hello, how can I help you?",
    "I am an AI chatbot created using NLP.",
    "Artificial Intelligence is the future.",
    "Machine learning is a subset of AI.",
    "NLP helps computers understand human language.",
    "Python is widely used for AI and ML.",
    "You can ask me about AI, NLP, or Python."
]

# Preprocess text
def preprocess(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    return text

corpus = [preprocess(sentence) for sentence in corpus]

# Chat function
def chatbot():
    print("AI Chatbot (type 'bye' to exit)")
    
    while True:
        user_input = input("You: ")
        user_input = preprocess(user_input)

        if user_input == "bye":
            print("Bot: Goodbye 👋")
            break

        corpus.append(user_input)

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(corpus)

        similarity = cosine_similarity(vectors[-1], vectors[:-1])
        idx = similarity.argsort()[0][-1]

        response = corpus[idx]
        print("Bot:", response)

        corpus.pop()

chatbot()