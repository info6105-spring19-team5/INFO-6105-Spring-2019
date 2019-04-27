# pythonspot.com
from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from sklearn.externals import joblib
import requests
import json
import pickle
import numpy as np
import tweepy
import pandas as pd
import matplotlib.pyplot as plt
import io
import re, string
import base64
import matplotlib.ticker as mtick


# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)

vectorizer = pickle.load(open('tf_idf_vectorizer.p','rb'))

toxic_model = pickle.load(open('toxic_model.p','rb'))
toxic_r = pickle.load(open('toxic_r.p','rb'))

obscene_model = pickle.load(open('obscene_model.p','rb'))
obscene_r = pickle.load(open('obscene_r.p','rb'))

threat_model = pickle.load(open('threat_model.p','rb'))
threat_r = pickle.load(open('threat_r.p','rb'))

insult_model = pickle.load(open('insult_model.p','rb'))
insult_r = pickle.load(open('insult_r.p','rb'))

identity_attack_model = pickle.load(open('identity_attack_model.p','rb'))
identity_attack_r = pickle.load(open('identity_attack_r.p','rb'))


label_cols = ['toxic', 'severe_toxicity', 'obscene', 'threat', 'insult','identity_attack']
models = {"toxic" : [toxic_model,toxic_r],
            "obscene":[obscene_model,obscene_r],"threat":[threat_model,threat_r],"insult":[insult_model,insult_r],
            "identity_attack":[identity_attack_model,identity_attack_r]}

# Fill the X's with the credentials obtained by  
# following the above mentioned procedure. 
consumer_key='uEDIF8SwN1mB8Ervc74wPXlF5'
consumer_secret='A2Z0R8ieiE1kAzEmVpUHvqwJiSdJ6kuGehLJChHgOV6dmvJQec'
access_key='1089095897470914560-JNAsPgAkFKtwzcFKUcGmFbsptzv0vq'
access_secret='wiqO2gAoHRVZPP7S5UqY5SawC5j9vDMOosPdsYUk8WcPC'

class ReusableForm(Form):
    name = TextAreaField('Name:', validators=[validators.required()])
    email = TextField('Email:', validators=[validators.required(), validators.Length(min=6, max=35)])
    password = TextField('Password:', validators=[validators.required(), validators.Length(min=3, max=35)])

def get_model(name,pred):
    if(pred == 'tox'):
        return models[name]
    else:
        return id_models[name]

def classify(text):
    v = vectorizer.transform(text)
    p = np.zeros((len(text), len(label_cols)))
    for i, j in enumerate(label_cols):
        model = get_model(j,"tox")
        p[:,i] = model[0].predict_proba(v.multiply(model[1]))[:,1]
    result = pd.concat([pd.DataFrame(p*100, columns = label_cols)], axis=1)
    return result


def getTwitterComments(userHandle):
     # Authorization to consumer key and consumer secret 
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret) 
    # Access to user's access key and access secret 
    auth.set_access_token(access_key, access_secret) 
    # Calling api 
    api = tweepy.API(auth) 
    # 200 tweets to be extracted 
    number_of_tweets=200
    tweets = api.user_timeline(screen_name=userHandle,count = number_of_tweets) 
    # Empty Array 
    tweetText=[]  
    # create array of tweet information: username,  
    # tweet id, date/time, text 
    tweets_for_csv = [tweet.text for tweet in tweets] # CSV file created  
    for j in tweets_for_csv: 
        tweetText.append(j) 

    return tweetText 

def build_graph(dataframe):
    img = io.BytesIO()
    
    
    ax=dataframe.plot(kind = 'barh', width=0.5)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
   
    
    plt.savefig(img, format='png')
    img.seek(0) 
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)

@app.route("/", methods=['GET', 'POST'])
def home():
    form = ReusableForm(request.form)
    return render_template('index.html', form=form)

@app.route("/demo", methods=['GET', 'POST'])
def demo():
    return render_template('demo.html', overall = 0,comment = '')

@app.route("/analyze", methods=['GET', 'POST'])
def analyze():
    form = ReusableForm(request.form)
    return render_template('analyze.html', form=form)
	
@app.route("/search", methods=['GET', 'POST'])
def search():
    form = ReusableForm(request.form)
    return render_template('search.html', form=form)

@app.route("/contact", methods=['GET', 'POST'])
def contact():
    form = ReusableForm(request.form)
    return render_template('contact.html', form=form)

@app.route("/share", methods=['GET', 'POST'])
def share():
    form = ReusableForm(request.form)
    return render_template('share.html', form=form)

@app.route("/submitComment", methods=['GET', 'POST'])
def submitComment():
    text = [request.form.get('comment')]
    result = classify(text)
    overall = round(result.iloc[0]['toxic'],2)
    return render_template('demo.html',overall = overall,comment = text[0])

@app.route("/analyzeText", methods=['GET', 'POST'])
def analyseText():
    text = [request.form.get('comment')]
    result = classify(text)
    overall = ""
    if(result.iloc[0]['toxic']>50):
        overall = "Toxic!"
    else:
        overall = "Non-toxic"
    graph1_url = build_graph(result)
    return render_template('results.html',tables=[result.to_html(classes='result',index = False,escape = False)],titles = label_cols, graph1=graph1_url,overall=overall)

@app.route("/analyzeTwitter", methods=['GET', 'POST'])
def analyseTwitter():
    form = ReusableForm(request.form)
    userHandle = request.form.get('handle')
    comments = getTwitterComments(userHandle)
    #predicting and cumulating the results of each comment
    df = pd.DataFrame()
    for comment in comments:
        x = classify([comment])
        df = df.append(x)

    new_df = pd.DataFrame(df.mean().to_dict(),index=[df.index.values[-1]])
    overall = ""
    if(new_df.iloc[0]['toxic']>50):
        overall = "Toxic!"
    else:
        overall = "Non-toxic"
    graph1_url = build_graph(new_df)
    return render_template('results.html',titles = label_cols, graph1=graph1_url,overall=overall)
	
 
if __name__ == "__main__":
    # Getting the classifier ready
    app.run()
