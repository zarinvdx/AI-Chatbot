from glob import glob
from deep_translator import GoogleTranslator
from flask import Flask, render_template, request, jsonify, session
import aiml
import csv
import requests
import speech_recognition as sr
import pyttsx3
import tempfile
import nltk
from nltk.sem import Expression
from nltk.inference import ResolutionProver
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
import difflib

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'secret123'

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

engine = pyttsx3.init()
recognizer = sr.Recognizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    words = text.lower().split()
    processed = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    replacements = {"steps": "how to", "explain": "how to"}
    return " ".join([replacements.get(w, w) for w in processed])

def translate_text(text, target_lang):
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        print(f"Translation Error: {e}")
        return "Sorry, I couldn't translate that."

def load_csv_data(filename):
    try:
        with open(filename, 'r', encoding='utf-16') as f:
            reader = csv.reader(f)
            next(reader, None)
            return {row[0].strip().lower(): row[1] for row in reader if len(row) > 1}
    except Exception as e:
        print(f"Error Loading CSV: {e}")
        return {}

def find_closest_match(query, data):
    query = preprocess_text(query)
    questions = list(data.keys())
    processed = [preprocess_text(q) for q in questions]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(processed + [query])
    similarity = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    idx = similarity.argmax()
    return questions[idx] if similarity[idx] > 0.2 else None

def get_exercise_details(exercise_name):
    api_key = "/wfiqRfVAE7yVKGO08e7OA==kEi235Qn3ukAR2G8"
    url = f"https://api.api-ninjas.com/v1/exercises?name={exercise_name}"
    headers = {"X-Api-Key": api_key}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data[:3] if data else []
        return []
    except Exception as e:
        print(f"API Error: {e}")
        return []

def format_exercise_response(ex):
    return f"""
    <p><b>Exercise:</b> {ex.get('name', 'Unknown')}</p>
    <p><b>Type:</b> {ex.get('type', 'N/A')}</p>
    <p><b>Muscle:</b> {ex.get('muscle', 'N/A')}</p>
    <p><b>Difficulty:</b> {ex.get('difficulty', 'N/A')}</p>
    <p><b>Instructions:</b> {ex.get('instructions', 'No instructions available.')}</p>
    """

#kb loader
def load_kb(filename):
    knowledge_base = []
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if not row or row[0].strip().startswith('#') or row[0].strip() == '':
                continue
            statement = row[0].strip().rstrip('.')
            try:
                expr = Expression.fromstring(statement)
                knowledge_base.append(expr)
            except Exception as e:
                print(f"Skipping invalid entry: {row[0]} - Error: {e}")
    return knowledge_base

#parse fol
def parse_to_fol(statement):
    statement = statement.strip().lower().rstrip('.')

    if "does not need equipment" in statement or "doesn't need equipment" in statement:
        subject = statement.replace("does not need equipment", "").replace("doesn't need equipment", "").replace("i know that", "").strip()
        return f"Not(NeedsEquipment({subject.capitalize()}))"

    if "needs equipment" in statement:
        subject = statement.replace("needs equipment", "").replace("i know that", "").strip()
        return f"NeedsEquipment({subject.capitalize()})"

    if "is not a cardio exercise" in statement or "is not cardio" in statement:
        subject = statement.replace("is not a cardio exercise", "").replace("is not cardio", "").replace("i know that", "").strip()
        return f"Not(IsCardio({subject.capitalize()}))"

    if "is a cardio exercise" in statement or "is cardio" in statement:
        subject = statement.replace("is a cardio exercise", "").replace("is cardio", "").replace("i know that", "").strip()
        return f"IsCardio({subject.capitalize()})"

    if "targets" in statement:
        parts = statement.replace("i know that", "").split("targets")
        if len(parts) == 2:
            subject = parts[0].strip().capitalize()
            muscle = parts[1].replace("muscles", "").strip().capitalize()
            return f"TargetsMuscle({subject}, {muscle})"

    if "is an alternative to" in statement:
        parts = statement.replace("i know that", "").split("is an alternative to")
        if len(parts) == 2:
            return f"AlternativeTo({parts[0].strip().capitalize()}, {parts[1].strip().capitalize()})"

    if "is a" in statement and "level exercise" in statement:
        parts = statement.replace("i know that", "").split("is a")
        difficulty = parts[1].replace("level exercise", "").strip()
        return f"Difficulty({parts[0].strip().capitalize()}, {difficulty.capitalize()})"

    if "is a" in statement:
        parts = statement.replace("i know that", "").split("is a")
        if len(parts) == 2:
            return f"Type({parts[0].strip().capitalize()}, {parts[1].strip().capitalize()})"

    return None


def humanize_fact(expr_str):
    if expr_str.startswith("IsCardio("):
        activity = expr_str[9:-1]
        return f"{activity} is a cardio exercise"

    if expr_str.startswith("Not(IsCardio("):
        activity = expr_str[13:-2]
        return f"{activity} is not a cardio exercise"

    if expr_str.startswith("NeedsEquipment("):
        item = expr_str[15:-1]
        return f"{item} needs equipment"

    if expr_str.startswith("Not(NeedsEquipment("):
        item = expr_str[19:-2]
        return f"{item} does not need equipment"

    if expr_str.startswith("Type("):
        parts = expr_str[5:-1].split(",")
        return f"{parts[0].strip()} is a type of {parts[1].strip()}"

    if expr_str.startswith("TargetsMuscle("):
        parts = expr_str[14:-1].split(",")
        return f"{parts[0].strip()} targets {parts[1].strip().lower()} muscles"

    if expr_str.startswith("AlternativeTo("):
        parts = expr_str[14:-1].split(",")
        return f"An alternative to {parts[0].strip()} is {parts[1].strip()}"

    if expr_str.startswith("Difficulty("):
        parts = expr_str[10:-1].split(",")
        return f"{parts[0].strip()} is a {parts[1].strip().lower()} level exercise"

    if expr_str.startswith("Not("):
        return f"It’s not true that {expr_str[4:-1]}"

    return expr_str

#first order logic
def check_fact(statement):
    fol = parse_to_fol(statement)
    if not fol:
        return "I couldn't understand what you're asking."
    expr = Expression.fromstring(fol)

    if ResolutionProver().prove(expr, kb):
        return "Correct"
    elif ResolutionProver().prove(Expression.fromstring(f"Not({str(expr)})"), kb):
        return "Incorrect"
    else:
        return "I don’t know"


def add_to_kb(statement):
    fol = parse_to_fol(statement)
    if not fol:
        return "I couldn't understand that statement in logical form."
    expr = Expression.fromstring(fol)
    for k in kb:
        if str(k) == f"Not({fol})":
            return f"That contradicts with what I already know: {k}"
    kb.append(expr)
    return f"Okay, I'll remember that {humanize_fact(fol)}."

#extra functionality for logic game
def play_logic_game():
    if not kb:
        return "Knowledge base is empty. Teach me something first."
    fact = random.choice(kb)
    question = str(fact)
    is_negated = question.startswith("Not(")
    if is_negated:
        inner = question[4:-1]
        question_text = f"Is it false that {humanize_fact(question)}?"
    else:
        question_text = f"Is it true that {humanize_fact(question)}?"
    return {"question": question_text, "answer": "no" if is_negated else "yes"}


#image classification
def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    return labels[class_index]

kernel = aiml.Kernel()
kernel.setTextEncoding(None)
kernel.bootstrap(learnFiles="chatbot.xml")

csv_data = load_csv_data('chatbot.csv')
kb = load_kb('kb.csv')

model = load_model('fitness_model.h5')
labels = sorted([folder for folder in os.listdir('static/images') if os.path.isdir(os.path.join('static/images', folder))])

image_lookup = {}
for label in labels:
    path_pattern = os.path.join('static/images', label, '*')
    image_files = glob(path_pattern)
    # Make paths relative to static
    image_files = [f.replace("static/", "") for f in image_files]
    if image_files:
        image_lookup[label.lower()] = image_files


@app.route('/')
def home():
    return render_template('Chatbot.html')

@app.route('/message', methods=['POST'])
def message():
    data = request.get_json()
    user_input = data.get('message', '').strip()

    if not user_input:
        return jsonify({'response': "Hmm, I didn’t catch that. Mind trying again?"})

    #Logic game handler
    if user_input == "play logic game":
        session['game'] = play_logic_game()
        return jsonify({'response': session['game']['question'] + " (yes/no)"})

    if user_input in ["yes", "no"] and 'game' in session:
        correct = session['game']['answer']
        if user_input == correct:
            result = "Nice job! That’s right!"
        else:
            result = f"Not quite — the correct answer was '{correct}'."
        session.pop('game')
        return jsonify({'response': result})

    #fol input handling
    if user_input.startswith("i know that"):
        fact = user_input.replace("i know that", "").strip()
        return jsonify({'response': add_to_kb(fact)})

    if user_input.startswith("check that"):
        fact = user_input.replace("check that", "").strip()
        return jsonify({'response': check_fact(fact)})

    if user_input == "show me everything you know":
        # Deduplicate by string value
        unique_facts = list({str(f): f for f in kb}.values())
        facts = [humanize_fact(str(expr)) for expr in unique_facts]
        return jsonify({'response': "<br>".join(
            f"- {fact}" for fact in facts) if facts else "I don’t know anything yet. Help me learn something new!"})

    #translation
    if user_input.startswith("translate this to "):
        parts = user_input.split(":", 1)
        if len(parts) == 2:
            lang = parts[0].replace("translate this to ", "").strip()
            text = parts[1].strip()
            return jsonify({'response': f"Translated to {lang.capitalize()}: {translate_text(text, lang)}"})
        return jsonify({'response': "Oops! Use this format: 'translate this to [language]: [text]'"})

    #image classification
    if "what is in this image" in user_input:
        image_path = session.get('uploaded_image_path')
        if image_path:
            predicted = classify_image(image_path)
            img_html = f'<img src="/{image_path}" style="max-width: 200px; border-radius: 10px;">'
            return jsonify({'response': f"Looks like this is: <b>{predicted}</b><br>{img_html}"})
        else:
            return jsonify({'response': "Please upload an image first using the 'Choose File' button before asking me to classify it."})

    if "what does" in user_input and "look like" in user_input:
        keyword = user_input.replace("what does", "").replace("look like", "").strip().lower()
        keyword = keyword.replace("-", " ").replace("_", " ")

        # Try to find closest match in image_lookup keys
        closest_match = difflib.get_close_matches(keyword, image_lookup.keys(), n=1, cutoff=0.6)
        if closest_match:
            label = closest_match[0]
            img_file = random.choice(image_lookup[label])
            img_tag = f'<img src="/static/{img_file}" style="max-width: 200px; border-radius: 10px;">'
            return jsonify({'response': f"Here’s what <b>{label}</b> looks like:<br>{img_tag}"})
        else:
            return jsonify({'response': f"Sorry, I don’t have a picture of '{keyword}' yet."})

    #AIML fallback
    response = kernel.respond(user_input.upper())
    if response.strip() and not response.lower().startswith("sorry"):
        return jsonify({'response': response})

    #chatbot.csv closest match
    closest = find_closest_match(user_input, csv_data)
    if closest:
        return jsonify({'response': csv_data[closest]})

    #exercise API
    if user_input.startswith("tell me about"):
        name = user_input.replace("tell me about", "").strip()
        results = get_exercise_details(name)
        if results:
            return jsonify({'response': "<hr>".join([format_exercise_response(ex) for ex in results])})

    #final fallback
    return jsonify({'response': "I’m still learning — try rephrasing your question?"})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'response': "No image uploaded."})
    file = request.files['image']
    if file.filename == '':
        return jsonify({'response': "No file selected."})

    filename = secure_filename(file.filename)
    path = os.path.join('static', filename)
    file.save(path)
    session['uploaded_image_path'] = path
    return jsonify({'response': "Image uploaded!"})

@app.route('/voice_input', methods=['POST'])
def voice_input():
    data = request.get_json()
    voice_text = data.get('voice', '').strip()

    if not voice_text:
        return jsonify({'response': "Didn't catch that."})


    with app.test_request_context('/message', method='POST', json={"message": voice_text}):
        response = message()  # Call the message() function directly
        response_data = response.get_json()
        reply_text = response_data.get('response', "Sorry, I didn’t get that.")

    #convert response to speech
    tts_engine = pyttsx3.init()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_path = temp_file.name
    temp_file.close()

    tts_engine.save_to_file(reply_text, temp_path)
    tts_engine.runAndWait()

    os.replace(temp_path, "static/response.mp3")

    return jsonify({'response': reply_text})

if __name__ == "__main__":
    app.run(debug=True)
