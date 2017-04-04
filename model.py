import re,sys
import numpy as np
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer,TfidfVectorizer
from sklearn import svm,linear_model,ensemble
from sklearn.multioutput import MultiOutputClassifier
from collections import Counter
from nltk.stem.wordnet import WordNetLemmatizer
WNL = WordNetLemmatizer()

#assigning predictor and target variables

stopwords = ['0', 'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'];

#start process_tweet
logfile = open('dump.txt','a')

numbers = ['one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen']

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
all_tags_map = {'part-time-job':0,'full-time-job':0,'hourly-wage':1,'salary':1,'associate-needed':2,'bs-degree-needed':2,'ms-or-phd-needed':2,
  'licence-needed':2,'1-year-experience-needed':3,'2-4-years-experience-needed':3,'5-plus-years-experience-needed':3,'supervising-job':4}
n_tags = 12

def normalize_text(desc):
  #  Remove ascii characters
  desc = ''.join([i.lower() if ord(i) < 128 else ' ' for i in desc.decode('utf-8')])
  desc = re.sub(r'((https?://[^\s]+)|(www\.[^\s]+))','LINK',desc)
  # 
  desc = re.findall(r"[\w']+", desc)
  desc = ' '.join( map( lambda x: str( WNL.lemmatize(x) ) ,desc) )
  # print(desc)
  # exit(0)
  
  print >> logfile,  desc
  return desc

def score(labels,test_labels,use_list=[1]*n_tags):
  print('using list',use_list,len(labels))
  assert(len(labels)==len(test_labels))
  stp,stn,sfp,sfn=0,0,0,0
  for i,x1 in enumerate(labels):
    x2 = test_labels[i]
    for j,y in enumerate(all_tags):
      if(use_list[j]==0):
        continue
      if( y in x1 and y in x2 ):
        stp+=1
      elif(y in x1 and y not in x2):
        sfn+=1
      elif(y not in x1 and y in x2):
        sfp+=1
      else:
        stn+=1
  print(stp,sfp,sfn,stn)
  p,r=stp*1.0/(stp+sfp),stp*1.0/(stp+sfn)
  print('precision',p,'recall',r)
  return 2*p*r/(p+r)

#########################################################################################################
# LOGISTIC REGRESSION

def learn(desc,labels,test_desc,test_labels):
  desc = list( map( lambda x: normalize_text(x), desc) )
  test_desc= list(map( lambda x: normalize_text(x), test_desc))
  vectorizer = TfidfVectorizer(ngram_range=(1,2),use_idf=True) #,stop_words=stopwords)

  train_vectors = vectorizer.fit_transform(desc)
  test_vectors = vectorizer.transform(test_desc)

  print(labels[:10])
  logreg = linear_model.LogisticRegression()
  multi_target_clf = MultiOutputClassifier(logreg, n_jobs=-1)
  multi_target_clf.fit(train_vectors, labels)


  if len(test_labels)!=0:
    print('AccuracyLogRegression',multi_target_clf.score(test_vectors,test_labels))
    Z = multi_target_clf.predict(test_vectors)
    print('f1_score',f1_score( map(lambda x:x[-1],test_labels) , map(lambda x:x[-1],Z), average='macro' ) )
    use_list = [0]*n_tags
    use_list[11]=1
    print('f1_score2',score( map(lambda x: all_tags[11] if x[-1]==1 else '',test_labels) , 
      map(lambda x: all_tags[11] if x[-1]==1 else '',Z),use_list ) )
    print( list(map(lambda x:x[-1],test_labels)) , list(map(lambda x:x[-1],Z)) )

  else:
    test_labels = multi_target_clf.predict(test_vectors)
  return test_labels

#########################################################################################################

def find_experience(desc,sr):
  ld = len(desc)
  # print(desc)
  key_words = ['experience'] #['experience','year']
  semi = ['year','minimum','work']
  values = []
  confidence = 0
  desc = re.findall(r"[\w']+", desc)
  for trigger in key_words:
    for i,u in enumerate(desc):
      if trigger in u:
        consider = desc[max(i-sr,0):min(i+sr,ld-1)]
        for j,z in enumerate(consider):
          for semi_words in semi:
            if semi_words in z:
              confidence+=1
          matchObj = re.match( r'\d+(\+)?', z, re.M|re.I)
          if matchObj:
            values.append(matchObj.group())
          if z in numbers:
            values.append(numbers.index(z)+1)
  values = list(set( filter(lambda x: x>0 and x<20, map(int,values) ) ))
  if confidence>0 and len(values)>0:
    exp = Counter(values).most_common(1)[0][0]
    if exp<2:
      return all_tags[8]
    elif exp<5:
      return all_tags[9]
    else:
      return all_tags[10]
  return ''

def find_supervision(desc,sr):
  ld = len(desc)
  # print(desc)
  key_words = ['supervis','manag'] #['experience','year']
  semi = ['inspect','moniter','manag','responsibl']
  values = []
  confidence = 0
  desc = re.findall(r"[\w']+", desc)
  for trigger in key_words:
    for i,u in enumerate(desc):
      if trigger in u:
        consider = desc[max(i-sr,0):min(i+sr,ld-1)]
        for j,z in enumerate(consider):
          for semi_words in semi:
            if semi_words in z:
              confidence+=1
          matchObj = re.match( r'\d+(\+)?', z, re.M|re.I)
          if matchObj:
            values.append(matchObj.group())
          if z in numbers:
            values.append(numbers.index(z)+1)
  values = list(set( filter(lambda x: x>0 and x<20, map(int,values) ) ))
  if confidence>0 and len(values)>0:
    exp = Counter(values).most_common(1)[0][0]
    if exp<2:
      return all_tags[8]
    elif exp<5:
      return all_tags[9]
    else:
      return all_tags[10]
  return ''

def text_extract(desc,labels):
  test_labels = []
  for i,x in enumerate(desc):
    desc[i]=normalize_text(x)

  for i,x in enumerate(desc):
    exp = find_experience(x,6)
    test_labels.append(exp)

  if labels!=[]:
    use_list = [0]*12
    use_list[8]=use_list[9]=use_list[10]=1
    print('score',score(labels,test_labels,use_list))
  return test_labels