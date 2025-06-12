import joblib
from pydub import AudioSegment
from flask import *
from flask_cors import CORS, cross_origin
import sqlite3 as sq
import speech_recognition as sr
from gtts import gTTS

app = Flask(__name__)

cors = CORS(app)
pipe_lr = joblib.load(open("emotion_classifier_pipe_lr.pkl", "rb"))

# Function


def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
                       "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"}


def voicetotext(file):
    r = sr.Recognizer()
    with sr.AudioFile(file) as source:
        audio_text = r.listen(source)
        text = r.recognize_google(audio_text)
    return text

import os
from flask import request
from pydub import AudioSegment

@app.route('/audio', methods=['POST'])
def process_audio():
    try:
        mode = request.args.get('mode', 'text')
        data = request.get_data()
        filename = request.headers.get("filename", "upload.wav")
        ext = os.path.splitext(filename)[1].lower()

        if request.content_length > 10 * 1024 * 1024:
            return 'File too large!', 400

        temp_file = f"temp{ext}"
        with open(temp_file, "wb") as f:
            f.write(data)

        # Ensure ffmpeg is configured
        AudioSegment.converter = "ffmpeg.exe"
        AudioSegment.ffmpeg = "ffmpeg.exe"
        AudioSegment.ffprobe = "ffprobe.exe"

        # Convert to wav for processing
        track = AudioSegment.from_file(temp_file)
        wav_path = "static/file.wav"
        track.export(wav_path, format='wav')

        # Process audio
        transcript = voicetotext(wav_path)

        if mode == 'text':
            prediction = predict_emotions(transcript)
            proba = get_prediction_proba(transcript)
        else:  # tone mode
            prediction = predict_emotiontone(wav_path)
            proba = ""

        print("Transcript:", transcript)
        print("Prediction:", prediction)
        print("Probability:", proba)

        emoji_icon = emotions_emoji_dict.get(prediction, "🙂")
        return f"{transcript}-{prediction}-{emoji_icon}"

    except Exception as e:
        print("Error processing audio:", str(e))
        return f"Some error occurred - Please try again - {emotions_emoji_dict.get('neutral', '🙂')}"

        
import librosa
import numpy as np
from keras.models import load_model
label_mapping=['angry','disgust','fear','happy','neutral','ps','sad']
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc
# Load the trained model
model = load_model("speech_emotion_rnn_model.h5")

# Function to process and predict emotion
def predict_emotiontone(audio_path):
    # Extract MFCC features
    mfcc_features = extract_mfcc(audio_path)
    mfcc_features = np.expand_dims(mfcc_features, axis=0)  # Reshape for model input
    mfcc_features = np.expand_dims(mfcc_features, -1)  # Add channel dimension
    
    # Predict
    prediction = model.predict(mfcc_features)
    predicted_label_index = np.argmax(prediction)
    print(predicted_label_index)
    # Map index back to label
    predicted_label = label_mapping[predicted_label_index]
    print(f"Predicted Emotion: {predicted_label}")
    return predicted_label

@app.route('/register', methods=['POST'])
def register():
    conn = sq.connect("register.db")
    conn.execute("create table if not exists info(id integer primary key AUTOINCREMENT,name varchar(100),email varchar(100),phone varchar(100),password varchar(100))")

    r = request.json
    conn.execute("insert into info(name,password,email,phone) values(?,?,?,?)",
                 (r["name"], r["password"], r["email"], r["phone"]))
    conn.commit()
    conn.close()
    return 's'


@app.route('/login', methods=["POST"], strict_slashes=False)
def login():
    r = request.json
    print(r)
    conn = sq.connect("register.db")
    n = conn.execute("select * from info where email=? and password=?",
                     (r["email"], r["password"])).fetchone()
    return json.dumps(n)


def text_to_audio(text, language='en', filename='output.mp3'):
    # Create a gTTS object
    tts = gTTS(text=text, lang=language, slow=False)
    # Save the audio file
    tts.save("static/"+filename)


@app.route('/check', methods=["POST"], strict_slashes=False)
def chcek():
    r = request.json
    print(r)
    indian_english_code = 'en-in'
    conn = sq.connect("register.db")
    # conn.execute("drop table data")
    conn.execute(
        "create table if not exists data(id integer PRIMARY KEY AUTOINCREMENT,pron varchar(1000),sid int)")
    conn.execute("insert into data (pron,sid)values(?,?)", (r["data"], r["i"]))
    conn.commit()
    text_to_audio(r["data"], language=indian_english_code)
    return "created successfull"


@app.route('/getall', methods=["POST"], strict_slashes=False)
def getall():
    r = request.json
    conn = sq.connect("register.db")
    x = conn.execute("select * from data where sid='%s'" % (r["i"])).fetchall()
    return json.dumps(x)


@app.route('/delete', methods=["POST"], strict_slashes=False)
def delete():
    r = request.json
    conn = sq.connect("register.db")
    conn.execute("delete from data where id='%s'" % (r["i"]))
    conn.commit()
    return 's'


if __name__ == '__main__':
    app.run('0.0.0.0')
