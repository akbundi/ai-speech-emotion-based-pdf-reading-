import pyttsx3
import pdfminer.high_level
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Set the voice and rate
voice = engine.getProperty('voices')[1]
engine.setProperty('voice', voice.id)
engine.setProperty('rate', 150)

def train_emotion_classifier(sentences, labels):
    # Create the CountVectorizer
    vectorizer = CountVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)

    # Create the Naive Bayes classifier
    classifier = MultinomialNB()

    
    X_train, X_test, y_train, y_test = train_test_split(sentence_vectors, labels, test_size=0.2, random_state=42)
    classifier.fit(X_train, y_train)
    return classifier, vectorizer

def detect_emotion(sentence, classifier, vectorizer):
    # Transform the input sentence into a feature vector
    sentence_vector = vectorizer.transform([sentence])

    # Predict the emotion using the trained classifier
    predicted_emotion = classifier.predict(sentence_vector)[0]
    return predicted_emotion

def read_pdf_with_emotions(pdf_path, chosen_emotion):
    with open(pdf_path, 'rb') as pdf_file:
        pdf_text = pdfminer.high_level.extract_text(pdf_file)

    sentences = pdf_text.split('. ')
    classifier, vectorizer = train_emotion_classifier(sentences, ['neutral'] * len(sentences))

    for sentence in sentences:
        # Perform emotion detection on each sentence
        detected_emotion = detect_emotion(sentence, classifier, vectorizer)

        # Use the chosen emotion if it was explicitly selected
        if chosen_emotion:
            emotion = chosen_emotion
        else:
            emotion = detected_emotion

        # Modulate voice based on chosen or detected emotion
        if emotion == 'happy':
            engine.setProperty('volume', 1.2)
            engine.setProperty('rate', 160)
        elif emotion == 'sad':
            engine.setProperty('volume', 0.8)
            engine.setProperty('rate', 120)
        elif emotion == 'angry':
            engine.setProperty('volume', 1.0)
            engine.setProperty('rate', 180)
        elif emotion == 'boredom':
            engine.setProperty('volume', 0.7)
            engine.setProperty('rate', 100)
        elif emotion == 'calmness':
            engine.setProperty('volume', 0.9)
            engine.setProperty('rate', 110)
        elif emotion == 'nostalgia':
            engine.setProperty('volume', 0.8)
            engine.setProperty('rate', 130)
        elif emotion == 'romance':
            engine.setProperty('volume', 1.1)
            engine.setProperty('rate', 140)
        elif emotion == 'satisfaction':
            engine.setProperty('volume', 1.0)
            engine.setProperty('rate', 150)
        elif emotion == 'confusion':
            engine.setProperty('volume', 0.8)
            engine.setProperty('rate', 130)
        elif emotion == 'amusement':
            engine.setProperty('volume', 1.1)
            engine.setProperty('rate', 170)
        elif emotion == 'sexual desire':
            engine.setProperty('volume', 1.0)
            engine.setProperty('rate', 160)
            # Add additional voice modulation for sexual desire
        elif emotion == 'sympathy':
            engine.setProperty('volume', 0.9)
            engine.setProperty('rate', 120)
            # Add additional voice modulation for sympathy
        elif emotion == 'triumph':
            engine.setProperty('volume', 1.1)
            engine.setProperty('rate', 150)
            # Add additional voice modulation for triumph
        else:
            engine.setProperty('volume', 1.0)
            engine.setProperty('rate', 150)

        engine.say(sentence + '.')
        engine.runAndWait()

def open_pdf():
    global pdf_path
    pdf_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    read_pdf_button.config(state=tk.NORMAL)

def read_pdf():
    available_emotions = ['happy', 'sad', 'angry', 'boredom', 'calmness', 'nostalgia', 'romance', 'satisfaction', 'confusion', 'amusement', 'sexual desire', 'sympathy', 'triumph', 'neutral']
    chosen_emotion = None
    while chosen_emotion not in available_emotions:
        emotion_index = simpledialog.askinteger("Choose Emotion", "Select an emotion by entering its number:", minvalue=1, maxvalue=len(available_emotions))
        if emotion_index is not None:
            chosen_emotion = available_emotions[emotion_index - 1]
        else:
            messagebox.showwarning("Invalid Choice", "Please choose a valid emotion.")
    read_pdf_with_emotions(pdf_path, chosen_emotion)

# Create the main window
root = tk.Tk()
root.title("PDF Reader with Emotions")
# Create and pack widgets
open_pdf_button = tk.Button(root, text="Open PDF", command=open_pdf)
open_pdf_button.pack(pady=20)

read_pdf_button = tk.Button(root, text="Read PDF with Emotions", command=read_pdf, state=tk.DISABLED)
read_pdf_button.pack(pady=10)

root.mainloop()