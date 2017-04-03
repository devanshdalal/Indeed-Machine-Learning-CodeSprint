import re,sys
import numpy as np
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer,TfidfVectorizer
from sklearn import svm,linear_model,ensemble
from sklearn.multioutput import MultiOutputClassifier

#assigning predictor and target variables

stopwords = ['0', 'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'];

#start process_tweet
logfile = open('dump.txt','a')

NEGATE = ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
              "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
              "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
              "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
              "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
              "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
              "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
              "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]

all_tags = ['part-time-job','full-time-job','hourly-wage','salary','associate-needed','bs-degree-needed','ms-or-phd-needed',
  'licence-needed','1-year-experience-needed','2-4-years-experience-needed','5-plus-years-experience-needed','supervising-job']
n_tags = 12

def add_features(desc):
    # process the desc
    #Replace WORD with word word
    desc=' '.join( (word.lower()+' '+word.lower() if len(word)>=3 and word.isupper() else word) for word in tweet.split() )
    #Replace negative words with NOT
    desc=' '.join( ('not' if word in NEGATE else word) for word in desc.split() )
    #trim punctuations
    desc = re.sub(r'[\'",.;?]',' ', desc)
    #Convert www.* or https?://* to URL
    desc = re.sub(r'((https?://[^\s]+)|(www\.[^\s]+))','LINK',desc)
    print >> logfile,  desc
    return desc
#end

#########################################################################################################
# LOGISTIC REGRESSION

def learn(desc,labels,test_desc,test_labels):
  vectorizer = TfidfVectorizer(ngram_range=(1,2),use_idf=True) #,stop_words=stopwords)

  train_vectors = vectorizer.fit_transform(desc)
  test_vectors = vectorizer.transform(test_desc)

  print(labels[:10])
  logreg = linear_model.LogisticRegression()
  multi_target_clf = MultiOutputClassifier(logreg, n_jobs=-1)
  multi_target_clf.fit(train_vectors, labels)

  # Z = multi_target_clf.predict(test_vectors)

  if len(test_labels)!=0:
    print('AccuracyLogRegression',multi_target_clf.score(test_vectors,test_labels))
  else:
    test_labels = multi_target_clf.predict(test_vectors)
  return test_labels

#########################################################################################################

def score(labels,test_labels):
  assert(len(labels)==len(test_labels))
  stp,stn,sfp,sfn=0,0,0,0
  for i,x1 in enumerate(labels):
    x2 = test_labels[i]
    for j,y in enumerate(all_tags):
      if( y in x1 and y in x2 ):
        stp+=1
      elif(y in x1 and y not in x2):
        sfn+=1
      elif(y not in x1 and y in x2):
        sfp+=1
      else:
        stn+=1
    p,r=stp*1.0/(stp+sfp),stp*1.0/(stp+sfn)
  return 2*p*r/(p+r)

def normalize_text(desc):
  #  Remove ascii characters
  desc = ''.join([i.lower() if ord(i) < 128 else ' ' for i in desc.decode('utf-8')])
  desc = re.sub(r'((https?://[^\s]+)|(www\.[^\s]+))','LINK',desc)
  # print(desc)
  # 
  desc = re.findall(r"[\w']+", desc)
  desc = ' '.join(str(desc) )
  # print(desc)
  
  print >> logfile,  desc
  return desc

def find_experience(desc):
  ld = len(desc)
  key_words = ['experience','year']
  values = []
  for trigger in key_words:
    for y in re.findall(trigger, desc):
      print('trigger',desc[:100])
      values+= re.findall('\d+', desc[min(ind-100,0):max(ind+100,ld-1)])
  if len(values)>0:
    print('values',values)

def text_extract(desc,labels):

  test_labels = []
  for i,x in enumerate(desc):
    desc[i]=normalize_text(x)
    find_experience(desc[i])



  
  return test_labels