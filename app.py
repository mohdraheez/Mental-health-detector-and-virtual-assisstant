import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from flask import Flask, render_template, request,jsonify

# Download NLTK data
nltk.download('popular')

# Load pre-trained model
model = load_model('model.h5')

# Load intents data
intents = json.loads(open('intents.json').read())

with open('intents.json', 'r') as file:
    intents2 = json.load(file)['intents']

words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Load lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to clean up a sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to create a bag of words
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

# Function to predict the class of an input sentence
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Function to get a response based on the detected intent
def getResponse(ints, intents_json):
    if ints:
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result
    else:
        return "Sorry, I didn't understand that."

# Function to generate suggestions based on a keyword
# Function to generate suggestions based on a keyword
# Function to generate suggestions based on a keyword
# Function to generate suggestions based on a keyword
def generate_suggestions(keyword, intents_json):
    suggestions = []
    list_of_intents = intents_json
    for intent in list_of_intents:
        # Check if the keyword matches any pattern associated with the intent
        if any(keyword.lower() in pattern.lower() for pattern in intent['patterns']):
            # Extract patterns from the intent
            patterns = intent['patterns']
            # Add patterns to the list
            suggestions.extend(patterns)
    return suggestions






def chatbot_response(msg):
    res = getResponse(predict_class(msg, model), intents)
    return res

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/assistant")
def home():
    return render_template("index.html")

import json

@app.route("/assistant/<string:ids>")
def home2(ids):
    id_array = ids.split(',')
    return render_template("index.html", id_array=json.dumps(id_array))


@app.route("/")
def detector():
    return render_template("index2.html")

@app.route("/get_suggestions", methods=["GET"])
def get_suggestions():
    # Get the keyword from the request
    keyword = request.args.get('keyword')
    # Generate suggestions based on the keyword
    suggestions = generate_suggestions(keyword, intents2)
    # Return suggestions as JSON response
    return jsonify({"suggestions": suggestions})

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    print("get_bot_response:- " + userText)

    chatbot_response_text = chatbot_response(userText)

    return chatbot_response_text

keyword = "depression"  # Example keyword
suggestions = generate_suggestions(keyword, intents2)
print("Suggestions for keyword '{}':".format(keyword))
print(suggestions)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
