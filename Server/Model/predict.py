import pandas as pd
import numpy as np
import re
import joblib
import spacy
import gensim
import nltk  

from nltk.corpus import stopwords
from imblearn.under_sampling import NearMiss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")
stop=set(stopwords.words('english'))

class predict:
  def __init__(self):
    self.__model = None
    pass
  
  # to preprocess text
  def __textpreprocessing(self,data,feature):
    # cleaning noisy and unnecessary stuff
    data[feature] = data[feature].apply(lambda value:value.lower())
    data[feature] = data[feature].apply(lambda value:re.sub('[,\.!?]', '', value))
    # tokenisation and remove punctuations
    data[feature] = data[feature].apply(lambda value : gensim.utils.simple_preprocess(value, deacc=True))
    # remove stopwords
    data[feature] = data[feature].apply(lambda value: [word for word in value if word not in stop]) 
    # bigram and trigram
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data[feature], min_count=5, threshold=50) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data[feature]], threshold=50)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    data[f'{feature}_bigram'] = data[feature].apply(lambda value:bigram_mod[value])
    data[f'{feature}_trigram'] = data[f'{feature}_bigram'].apply(lambda value:trigram_mod[value])

    data['highlights_trigram'] = data['highlights_trigram'].apply(lambda value: ' '.join(value))
    # vectorisation using spacy
    data['vectors'] = data['highlights_trigram'].apply(lambda value: nlp(value).vector)
    data = data.drop(['highlights','highlights_bigram','highlights_trigram'],axis=1)
    
    return np.stack(data['vectors'])

  # To train new data 
  def trainModel(self,data):
      highlights = data
      train = highlights.loc[highlights['highlights_sentiment']!='0.0'].iloc[:,1:]

      y = train['highlights_sentiment'].map({'POSITIVE':1,'NEGATIVE':0})

      X = self.textpreprocessing(train,'highlights')

      nm1 = NearMiss(
          sampling_strategy='auto',  # undersamples only the majority class
          version=1,
          n_neighbors=3,
          n_jobs=4)  # I have 4 cores in my laptop

      X_resampled, y_resampled = nm1.fit_resample(X, y)

      clf =  LogisticRegression(solver='lbfgs',penalty='l2',random_state=100).fit(X_resampled, y_resampled)
      self.__model = clf
      joblib.dump(clf,'Model/data/sentiment_predict.pkl')

  # get pickle file
  def getModel(self):
    return joblib.load('Model/sentiment_predict.pkl')

  # to predict new text
  def predict_text(self,text):
    data = {'highlights':[text]}
    data = pd.DataFrame(data)
    clf = self.getModel()  
    if (clf.predict(self.__textpreprocessing(data,'highlights'))[0])==1:
      return 'POSITIVE'
    else:
      return 'NEGATIVE' 

