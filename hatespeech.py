import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
import nltk
import re
import keyboard
import speech_recognition as sr
import string

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stopword = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

data = pd.read_csv("C:\\Users\\Raghavendra\\OneDrive\\Desktop\\hate speech detection\\labeled_data.csv")

data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Speech", 2: "No Hate and Offensive Speech"})
data = data[["tweet", "labels"]]

def clean(text):
    text = str(text).lower()
    text = re.sub('[.?]', '', text)
    text = re.sub('https?://\S+|www.\S+', '', text)
    text = re.sub('<.?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w\d\w', '', text)
    words = [word for word in text.split(' ') if word not in stopword]
    stemmed_words = [stemmer.stem(word) for word in words]
    text = " ".join(stemmed_words)
    return text

data["tweet"] = data["tweet"].apply(clean)

x = np.array(data["tweet"])
y = np.array(data["labels"])

cv = CountVectorizer()
X = cv.fit_transform(x)

model = DecisionTreeClassifier()
model.fit(X, y)

def predict_hate_speech(text):
    text = clean(text)
    text_vectorized = cv.transform([text])
    prediction = model.predict(text_vectorized)
    return prediction[0]

def speech_to_text():
    recognizer = sr.Recognizer()
    while True:
        with sr.Microphone() as source:
            print("Listening... Speak something:")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            print("You said: " + text)
            prediction = predict_hate_speech(text)
            print("Prediction: " + prediction)
        except sr.UnknownValueError:
            print("Sorry, I could not understand your speech.")
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
        
        if keyboard.is_pressed('esc'):
            break

if __name__ == "__main__":
    speech_to_text()

