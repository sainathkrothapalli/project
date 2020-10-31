from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from flask import Flask, render_template,request, url_for
from keras.datasets import imdb
#import tensorflow as tf
import nltk
import numpy as np
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
from keras.models import load_model
app=Flask(__name__)
model=load_model('Myproject.h5')
@app.route('/')


def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method=='POST':
       
        texts=request.form['text']
        review = re.sub('[^a-zA-Z]', ' ',texts)
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review=' '.join(review)
        voc_size=5000
        one_hotr=[one_hot(review,voc_size)]
        sent_length=600
        embedded_docs=pad_sequences(one_hotr,padding='pre',maxlen=sent_length)
        prediction=model.predict(embedded_docs)
        prediction=(prediction>0.5)
        sentiment=''
        if prediction==True:
            sentiment='positive'
        else:
            sentiment='negative'
       

       # output=texts

    return render_template('index.html',output='The given review is  {}'.format(sentiment))


if __name__=='__main__':
    app.run(debug=True)