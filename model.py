import re,sys
import numpy as np
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer,TfidfVectorizer
from sklearn import svm,linear_model,ensemble
from sklearn.multioutput import MultiOutputClassifier

#assigning predictor and target variables

stopwords = ['0', 'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'];

#start process_tweet
logfile = open('dump.txt','wb')

NEGATE = ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
              "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
              "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
              "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
              "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
              "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
              "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
              "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]

def add_features(description):
    # process the descriptions
    #Replace WORD with word word
    description=' '.join( (word.lower()+' '+word.lower() if len(word)>=3 and word.isupper() else word) for word in tweet.split() )
    #Replace negative words with NOT
    description=' '.join( ('not' if word in NEGATE else word) for word in description.split() )
    #trim punctuations
    description = re.sub(r'[\'",.;?]',' ', description)
    #Convert www.* or https?://* to URL
    description = re.sub(r'((https?://[^\s]+)|(www\.[^\s]+))','LINK',description)
    print >> logfile,  description
    return description
#end

#########################################################################################################
# LOGISTIC REGRESSION

def learn(descriptions,labels,test_descriptions,test_labels):
  vectorizer = TfidfVectorizer(ngram_range=(1,2),use_idf=True) #,stop_words=stopwords)

  train_vectors = vectorizer.fit_transform(descriptions)
  test_vectors = vectorizer.transform(test_descriptions)

  logreg = linear_model.LogisticRegression()
  multi_target_clf = MultiOutputClassifier(logreg, n_jobs=-1)
  multi_target_clf.fit(train_vectors, labels)

  # Z = multi_target_clf.predict(test_vectors)

  if len(test_labels)!=0:
    print('AccuracyLogRegression',multi_target_clf.score(test_vectors,test_labels))
  else:
    test_labels = multi_target_clf.predict(test_vectors)
  return test_labels