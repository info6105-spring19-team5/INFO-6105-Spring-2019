# pythonspot.com
from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from sklearn.externals import joblib
import requests
import json
import pickle
import numpy as np
import pandas as pd

 
# musixmatch api base url
base_url = "https://api.musixmatch.com/ws/1.1/"

# api key
api_key = "&apikey=5e6203280f5c220a14303055a070091d"

track_charts = "chart.tracks.get"
lyrics_url = "track.lyrics.get?track_id="

format_url = '?chart_name=top&format=json&callback=callback&country='

#Countries
getCountry = {'au' : 'Australia', 'ca'  : 'Canada', 'uk': 'United Kingdom'}

# Getting classifiers ready
vect = pickle.load(open('countv.p','rb'))
clf = pickle.load(open('clf_countv.p','rb'))
le = pickle.load(open('label_encoder.p','rb'))


# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)

class ReusableForm(Form):
    name = TextAreaField('Name:', validators=[validators.required()])
    email = TextField('Email:', validators=[validators.required(), validators.Length(min=6, max=35)])
    password = TextField('Password:', validators=[validators.required(), validators.Length(min=3, max=35)])

# Utility function to build html image tag
def path_to_image_html(path):
    return '<img src="'+ path + '" width="60" >'

# Utility function to calculate filename of the emoji based on predcited probability
def getImageName(mood,prob):
    fileName = 'static\\'
    if prob in range(0,20):
        fileName += mood + "\\1.png"       
    elif prob in range(20,40):
        fileName += mood + "\\2.png"
    elif prob in range(40,60):
        fileName += mood + "\\3.png"
    elif prob in range(60,80):
        fileName += mood + "\\4.png"
    elif prob in range(80,101):
        fileName += mood + "\\5.png"
    return fileName
 
# Predicts probability and labels the mood of the text passed.
def classify(text):
    x_vect = vect.transform([text])
    proba = np.max(clf.predict_proba(x_vect))
    pred = clf.predict(x_vect)[0]
    label = le.inverse_transform([pred])
    return label,round(proba*100,2)

@app.route('/predict', methods=['GET','POST'])
def make_prediction():
    countryCode = request.form.get('country')
    form = ReusableForm(request.form)
    api_call = base_url + track_charts + format_url + countryCode + api_key
    #requesting trackids of the country passed
    req = requests.get(api_call)
    result = req.json()
    data = json.dumps(result, sort_keys=True, indent=2)
    d = json.loads(data)
    track_ids = [x['track']['track_id'] for x in d['message']['body']['track_list']]
    track_name = [x['track']['track_name'] for x in d['message']['body']['track_list']]
    artist_name = [x['track']['artist_name']for x in d['message']['body']['track_list']]
    mood = []
    emoji = []
    probability = []
    for id in track_ids:
        api_cal = base_url + lyrics_url + str(id) + api_key
        #requesting lyrics of the trackid passed.
        r = requests.get(api_cal)
        dat = r.json()
        text = dat['message']['body']['lyrics']['lyrics_body'].split('...')[0]
        prediction = classify(text)
        currentMood = prediction[0]
        mood.extend(currentMood)
        proba = str(round(prediction[1]))
        probability.append(proba)
        filename = getImageName(currentMood,round(prediction[1]))
        emoji.append(filename[0])
    #dataframe for results table
    df = pd.DataFrame({
        'TrackId':pd.Series(track_ids),
        'Track Name':pd.Series(track_name),
        'Artist Name':pd.Series(artist_name),
        'Mood':pd.Series(mood),
        'Confidence':pd.Series(probability),
        'Emoji':pd.Series(emoji),
    })
    return render_template('results.html',tables=[df.to_html(classes='df',index = False,escape = False,formatters=dict(Emoji=path_to_image_html))],titles = ['Musixmatch TrackId','Track Name','Artist Name','Mood','Confidence','Emoji'],country=getCountry[countryCode])

@app.route("/predictLyrics", methods=['GET', 'POST'])
def predictLyrics():
    form = ReusableForm(request.form)
    lyrics = request.form.get('message')
    prediction = classify(lyrics)
    mood = prediction[0][0]
    prob = prediction[1]
    # building message for results
    message = "The song is " + str(prob) + "% " + mood
    return render_template('lyrics.html', form=form, mood = message)

@app.route("/", methods=['GET', 'POST'])
def home():
    form = ReusableForm(request.form)
    return render_template('index.html', form=form)

@app.route("/chart", methods=['GET', 'POST'])
def country():
    form = ReusableForm(request.form)
    return render_template('hello.html', form=form)

@app.route("/lyrics", methods=['GET', 'POST'])
def lyrics():
    form = ReusableForm(request.form)
    return render_template('lyrics.html', form=form, mood ='')
	
@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      return render_template("test.html",result = result)
	
 
if __name__ == "__main__":
    app.run()
