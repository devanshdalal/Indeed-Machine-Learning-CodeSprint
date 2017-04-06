# -*- coding: utf-8 -*-
import re,sys,os,time
import numpy as np
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer,TfidfVectorizer
from sklearn import svm,linear_model,ensemble
from sklearn.multioutput import MultiOutputClassifier
from collections import Counter
from nltk.stem.porter import *
stemmer = PorterStemmer()

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
  desc = desc.replace('/', ' per ')
  desc = desc.replace('$', ' DOLLAR ')
  desc = ''.join([i.lower() if ord(i) < 128 else ' ' for i in desc.decode('utf-8')])
  desc = re.sub(r'((https?://[^\s]+)|(www\.[^\s]+))','LINK',desc)
  # 
  desc = re.findall(r"[\w']+", desc)
  desc = ' '.join( map( stemmer.stem ,desc) )
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
    # print(x1,x2,stp,sfp,sfn,stn)
    for j,y in enumerate(all_tags):
      if(use_list[j]==0):
        continue
      # print(y)
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

def find_job_timings(desc,sr,key_words,semi,conf):
  ld = len(desc)
  # print(desc)
  # key_words = ['per','hour'] #['experience','year']
  # semi1 = ['hr','wage','earn','dollar','week']
  # semi1 = ['year','salary','earn','dollar','annum']
  confidence = 0
  desc = re.findall(r"[\w']+", desc)
  for trigger in key_words:
    for i,u in enumerate(desc):
      if trigger in u:
        consider = desc[max(i-sr,0):min(i+sr,ld-1)]
        for j,z in enumerate(consider):
          for semi_word in semi:
            if semi_word!=trigger and semi_word in z:
              confidence+=1
  if confidence>conf:
    return True
  return False

def text_extract(desc,labels):
  test_labels = [[] for _ in desc]
  for i,x in enumerate(desc):
    desc[i]=normalize_text(x)

  for i,x in enumerate(desc):
    # exp = find_experience(x,6)
    # test_labels[i].append(exp)
    if find_job_timings(x,4,['per','hour'],['dollar','wage','earn','week','day','hr'],2):
      test_labels[i].append(all_tags[2])
    elif find_job_timings(x,6,['per','year','annum'],['salary','earn','dollar'],1):
      test_labels[i].append(all_tags[3])

  # print(test_labels)


  if labels!=[]:
    use_list = [0]*12
    # use_list[2]=1
    use_list[3]=1
    print('score',score(labels,test_labels,use_list))
  return test_labels


##############################################################################################################################

def compress_labels(y_labels):
  y_converted = [0]*5
  if y_labels[0]==1 or y_labels[1]==1:
    y_converted[0]=1+y_labels[:2].index(1)
  if y_labels[2]==1 or y_labels[3]==1:
    y_converted[1]=1+y_labels[2:4].index(1)
  if y_labels[4]==1 or y_labels[5]==1 or y_labels[6]==1 or y_labels[7]==1:
    y_converted[2]=1+y_labels[4:8].index(1)
  if y_labels[8]==1 or y_labels[9]==1 or y_labels[10]==1 :
    y_converted[3]=1+y_labels[8:11].index(1)
  if y_labels[11]==1:
    y_converted[4]=1
  return y_converted

def expand_labels(x):
  res_labels = [0]*n_tags
  if x[0]>0:
    res_labels[x[0]-1]=1
  if x[1]>0:
    res_labels[x[1]+1]=1
  if x[2]>0:
    res_labels[x[2]+3]=1
  if x[3]>0:
    res_labels[x[3]+7]=1
  if x[4]>0:
    res_labels[11]=1
  return res_labels

def get_text_labels(x):
  res_labels = []
  if len(x)>5:
    x = compress_labels(x)
  if x[0]>0:
    res_labels.append(all_tags[x[0]-1])
  if x[1]>0:
    res_labels.append(all_tags[x[1]+1])
  if x[2]>0:
    res_labels.append(all_tags[x[2]+3])
  if x[3]>0:
    res_labels.append(all_tags[x[3]+7])
  if x[4]>0:
    res_labels.append(all_tags[11])
  return res_labels

def isTimeFormat(input):
    try:
        time.strptime(input, '%H:%M')
        return True
    except ValueError:
        return False

def feature_experience(desc,sr):
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
  values = sorted(list(set( filter(lambda x: x>0 and x<20, map(int,values) ) )))
  if confidence>0 and len(values)>0:
    exp = Counter(values).most_common(1)[0][0]
    return [exp,values[-1],values[0]]
    #  mode, largest, smallest
  return [0,0,0]

def extract_features(text):

  feature_vec = []
  text = text.lower()
  desc = list(map(lambda xx: xx.strip(' .,?!*'),text.split()))
  ld = len(desc)

  # currency terms
  currency = re.findall( r'[\$]?[ ]*[0-9,.]+k?[\$]?' ,text)
  for j,xx in enumerate(currency):
    xx = re.sub('[,$]', '', xx)
    xx = re.sub('[k]', '000', xx)
    xx = re.sub('[ ]', '', xx)
    currency[j] = int(xx)
  sorted(currency)
  


  # Datetime 
  def interval():
    import parsedatetime
    p = parsedatetime.Calendar()
    max_hr,min_hr=0,24
    for i,xx in enumerate(desc):
      if(isTimeFormat(xx) or p.parse(xx)[1]>=1):
        s=''.join(desc[max(0,i-1):min(i+2,ld)])
        tm, status = p.parse(s)
        if status>=2:
          if tm.tm_hour<min_hr:
            min_hr = tm.tm_hour
          elif tm.tm_hour>max_hr:
            max_hr = tm.tm_hour
    if min_hr>max_hr:
      return 8
    else:
      return max_hr - min_hr
  feature_vec.append(interval())

  if find_job_timings(text,4,['per','hour'],['dollar','wage','earn','week','day','hr'],2):
    feature_vec.append(1)
  else:
    feature_vec.append(0)

  if find_job_timings(text,6,['per','year','annum'],['salary','earn','dollar'],1):
    feature_vec.append(1)
  else:
    feature_vec.append(0)


  # extract normalized features
  text = normalize_text(text)
  jd_dict = map( stemmer.stem, ['salary','wage', 'full', 'part', 'master','bachelor','phd','associate','licence','education','experience','hour','plus',
                  'supervise','manage','lead','mentor','moniter','pay'])
  for xx in jd_dict:
    feature_vec.append(text.count(xx))
  feature_vec+=feature_experience(text,5)
  return feature_vec


def learn_simple(desc,labels,test_desc,test_labels):
  desc = list( map( lambda x: normalize_text(x), desc) )
  test_desc= list(map( lambda x: normalize_text(x), test_desc))

  train_vectors, test_vectors = [], []
  for x in desc:
    train_vectors.append( extract_features(x) )
  for x in test_desc:
    test_vectors.append( extract_features(x) )
  logreg = linear_model.LogisticRegression()
  multi_target_clf = MultiOutputClassifier(logreg, n_jobs=-1)
  multi_target_clf.fit(train_vectors, labels)

  if len(test_labels)!=0:
    print('AccuracyLogRegression',multi_target_clf.score(test_vectors,test_labels))
    Z = multi_target_clf.predict(test_vectors)
    print('f1_score',f1_score( map(lambda x:x[-1],test_labels) , map(lambda x:x[-1],Z), average='macro' ) )
    use_list = [0]*n_tags
    use_list[8]=use_list[9]=use_list[10]=1
    print('f1_score2',score( map(get_text_labels,test_labels) , map(get_text_labels,Z) ) )
    print( list(map(lambda x:x[-1],test_labels)) , list(map(lambda x:x[-1],Z)) )

  else:
    test_labels = multi_target_clf.predict(test_vectors)
  return test_labels


##############################################################################################################################


def prepare_manual(test_lines,train_lines):
  needed = [0, 1, 2, 4, 6, 7, 8, 10, 11, 16, 17, 21, 22, 23, 24, 27, 30, 31, 32, 33, 36, 39, 40, 41, 42, 44, 45, 46, 47, 48, 50, 51, 52, 53, 56, 57, 58, 60, 61, 62, 67, 68, 70, 71, 72, 73, 74, 75, 76, 79, 81, 83, 84, 85, 86, 87, 88, 89, 91, 92, 93, 94, 98, 101, 102, 103, 106, 108, 111, 116, 118, 119, 120, 123, 125, 126, 129, 130, 131, 132, 133, 134, 135, 138, 142, 144, 145, 148, 153, 156, 158, 159, 161, 164, 165, 166, 167, 168, 169, 170, 177, 178, 179, 184, 185, 187, 188, 190, 191, 193, 197, 198, 199, 202, 203, 206, 207, 211, 212, 213, 214, 215, 217, 218, 220, 222, 223, 224, 225, 227, 228, 229, 230, 233, 234, 236, 241, 242, 244, 245, 249, 250, 252, 256, 257, 258, 260, 262, 263, 267, 268, 270, 271, 273, 274, 276, 277, 279, 281, 282, 284, 285, 286, 287, 288, 289, 291, 292, 293, 294, 295, 299, 300, 301, 302, 303, 304, 305, 306, 308, 309, 311, 312, 315, 316, 317, 324, 325, 326, 328, 331, 332, 333, 335, 339, 342, 344, 345, 346, 349, 350, 351, 352, 360, 361, 362, 363, 369, 370, 372, 376, 377, 381, 382, 389, 391, 394, 396, 890, 891, 892, 893, 894, 895, 896, 899, 900, 901, 905, 912, 914, 919, 920, 924, 928, 930, 933, 934, 935, 936, 937, 940, 942, 946, 947, 949, 952, 953, 954, 955, 956, 957, 959, 965, 966, 967, 968, 969, 972, 975, 977, 978, 979, 981, 983, 984, 986, 987, 988, 992, 993, 996, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1014, 1015, 1016, 1020, 1021, 1024, 1026, 1027, 1028, 1029, 1031, 1032, 1033, 1035, 1036, 1039, 1043, 1045, 1046, 1047, 1050, 1052, 1053, 1054, 1055, 1057, 1059, 1061, 1064, 1067, 1070, 1073, 1076, 1077, 1080, 1082, 1085, 1086, 1087, 1095, 1096, 1097, 1098, 1101, 1102, 1103, 1108, 1110, 1115, 1121, 1127, 1128, 1129, 1134, 1135, 1137, 1139, 1142, 1143, 1144, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1171, 1172, 1174, 1176, 1177, 1178, 1180, 1183, 1184, 1186, 1189, 1190, 1193, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1206, 1207, 1208, 1210, 1211, 1214, 1215, 1216, 1217, 1219, 1220, 1223, 1224, 1226, 1228, 1229, 1237, 1241, 1244, 1245, 1246, 1247, 1250, 1254, 1259, 1265, 1268, 1269, 1270, 1273, 1275, 1276, 1277, 1279, 1282, 1285, 1287, 1288, 1295, 1296, 1299, 1300, 1301, 1309, 1311, 1312, 1314, 1319, 1320, 1322, 1325, 1326, 1328, 1329, 1332, 1334, 1338, 1341, 1342, 1343, 1345, 1346, 1348, 1349, 1353, 1354, 1355, 1359, 1360, 1361, 1362, 1368, 1369, 1371, 1375, 1377, 1380, 1381, 1385, 1387, 1389, 1390, 1394, 1403, 1405, 1406, 1407, 1409, 1410, 1411, 1412, 1413, 1414, 1415, 1419, 1421, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1437, 1440, 1441, 1442, 1444, 1446, 1450, 1452, 1453, 1457, 1460, 1461, 1462, 1465, 1468, 1471, 1476, 1477, 1480, 1481, 1482, 1484, 1486, 1487, 1489, 1493, 1497, 1499, 1501, 1502, 1504, 1505, 1510, 1511, 1512, 1513, 1514, 1518, 1520, 1521, 1524, 1526, 1528, 1529, 1530, 1533, 1535, 1537, 1540, 1541, 1542, 1543, 1544, 1545, 1547, 1548, 1549, 1550, 1551, 1552, 1554, 1556, 1557, 1563, 1564, 1565, 1566, 1567, 1568, 1576, 1579, 1584, 1585, 1586, 1588, 1590, 1592, 1595, 1598, 1602, 1603, 1604, 1605, 1606, 1609, 1610, 1611, 1616, 1619, 1624, 1625, 1626, 1630, 1631, 1636, 1638, 1640, 1641, 1642, 1644, 1646, 1650, 1651, 1652, 1654, 1655, 1658, 1659, 1660, 1661, 1664, 1667, 1669, 1672, 1673, 1676, 1677, 1678, 1679, 1680, 1681, 1682, 1684, 1685, 1686, 1687, 1690, 1691, 1693, 1695, 1698, 1702, 1704, 1706, 1711, 1713, 1720, 1721, 1722, 1726, 1729, 1730, 1732, 1735, 1736, 1738, 1747, 1749, 1751, 1753, 1754, 1755, 1756, 1758, 1760, 1761, 1762, 1764, 1766, 1768, 1770, 1775, 1776, 1778, 1781, 1783, 1785, 1786, 1787, 1788, 1791, 1792, 1793, 1795, 1796, 1798, 1799, 1803, 1804, 1806, 1808, 1810, 1813, 1814, 1815, 1818, 1819, 1821, 1823, 1824, 1826, 1827, 1829, 1830, 1832, 1833, 1835, 1840, 1841, 1842, 1843, 1844, 1845, 1846, 1848, 1851, 1852, 1853, 1855, 1856, 1857, 1858, 1859, 1861, 1862, 1863, 1867, 1868, 1869, 1872, 1873, 1876, 1877, 1878, 1881, 1884, 1887, 1895, 1898, 1901, 1902, 1903, 1904, 1906, 1907, 1908, 1910, 1912, 1918, 1920, 1921, 1922, 1923, 1924, 1926, 1928, 1930, 1932, 1933, 1934, 1937, 1938, 1940, 1942, 1943, 1944, 1946, 1947, 1949, 1950, 1952, 1955, 1956, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1971, 1972, 1973, 1978, 1980, 1981, 1982, 1983, 1990, 1991, 1992, 1997, 1999, 2000, 2001, 2002, 2003, 2007, 2008, 2011, 2012, 2014, 2015, 2016, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2027, 2029, 2030, 2031, 2033, 2034, 2035, 2036, 2039, 2041, 2042, 2045, 2046, 2048, 2053, 2055, 2060, 2063, 2066, 2068, 2072, 2076, 2077, 2078, 2079, 2080, 2084, 2085, 2086, 2088, 2090, 2091, 2092, 2095, 2097, 2098, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2116, 2117, 2118, 2119, 2121, 2122, 2123, 2124, 2125, 2126, 2128, 2129, 2132, 2133, 2134, 2137, 2138, 2142, 2143, 2145, 2146, 2147, 2149, 2150, 2151, 2153, 2155, 2156, 2157, 2158, 2162, 2163, 2165, 2170, 2171, 2172, 2173, 2175, 2176, 2177, 2181, 2182, 2183, 2185, 2186, 2187, 2189, 2191, 2196, 2197, 2198, 2200, 2202, 2203, 2205, 2206, 2210, 2211, 2212, 2213, 2214, 2216, 2218, 2221, 2222, 2223, 2226, 2229, 2230, 2231, 2234, 2235, 2236, 2238, 2239, 2243, 2244, 2245, 2246, 2248, 2249, 2255, 2257, 2259, 2260, 2261, 2262, 2264, 2265, 2266, 2267, 2268, 2269, 2270, 2273, 2274, 2278, 2281, 2282, 2283, 2285, 2289, 2293, 2294, 2295, 2296, 2298, 2300, 2302, 2303, 2306, 2307, 2313, 2316, 2320, 2325, 2329, 2330, 2331, 2332, 2333, 2334, 2335, 2339, 2340, 2344, 2345, 2346, 2347, 2349, 2354, 2355, 2356, 2357, 2358, 2359, 2360, 2361, 2362, 2366, 2368, 2370, 2371, 2374, 2375, 2376, 2377, 2378, 2380, 2381, 2382, 2383, 2384, 2385, 2392, 2393, 2396, 2397, 2401, 2402, 2403, 2404, 2408, 2411, 2412, 2413, 2416, 2423, 2426, 2428, 2430, 2431, 2433, 2434, 2435, 2437, 2438, 2439, 2443, 2444, 2446, 2447, 2448, 2450, 2452, 2454, 2456, 2458, 2461, 2463, 2464, 2467, 2469, 2470, 2473, 2474, 2477, 2478, 2479, 2480, 2481, 2482, 2484, 2485, 2488, 2490, 2491, 2494, 2496, 2497, 2498, 2501, 2504, 2505, 2506, 2507, 2508, 2509, 2510, 2512, 2514, 2518, 2524, 2526, 2529, 2531, 2532, 2533, 2534, 2535, 2537, 2538, 2539, 2540, 2541, 2542, 2543, 2550, 2553, 2554, 2555, 2556, 2558, 2560, 2562, 2566, 2567, 2568, 2569, 2572, 2575, 2576, 2578, 2579, 2580, 2582, 2583, 2584, 2585, 2587, 2588, 2589, 2591, 2592, 2597, 2600, 2601, 2604, 2609, 2611, 2612, 2614, 2616, 2618, 2625, 2626, 2628, 2630, 2634, 2635, 2636, 2637, 2646, 2647, 2649, 2655, 2656, 2660, 2663, 2664, 2665, 2667, 2668, 2669, 2671, 2673, 2674, 2675, 2677, 2678, 2679, 2680, 2681, 2682, 2684, 2685, 2686, 2688, 2689, 2690, 2692, 2693, 2694, 2695, 2699, 2700, 2704, 2705, 2707, 2709, 2711, 2712, 2713, 2714, 2721, 2722, 2729, 2730, 2732, 2737, 2741, 2750, 2752, 2754, 2756, 2757, 2758, 2762, 2764, 2765, 2766, 2767, 2772, 2775, 2776, 2778, 2780, 2783, 2786, 2788, 2789, 2792, 2793, 2795, 2796, 2798, 2801, 2802, 2803, 2810, 2816, 2818, 2821, 2823, 2826, 2827, 2829, 2831, 2832, 2833, 2834, 2837, 2838, 2839, 2842, 2843, 2844, 2849, 2851, 2854, 2855, 2856, 2860, 2863, 2864, 2866, 2869, 2870, 2874, 2875, 2876, 2878, 2891, 2894, 2895, 2897, 2899, 2901, 2902, 2903, 2905, 2907, 2912, 2913, 2915, 2917, 2918]

  os.system('mkdir -p tmp/tests/')
  for i,x in enumerate(needed):
    test_lines[x][0].replace('•','\n')
    test_lines[x][0].replace('*','\n')
    open('tmp/tests/'+ str(x+2)+'.txt','w+').write(test_lines[x][0]+'\n')

  os.system('mkdir -p tmp/train/')
  for i,x in enumerate(train_lines):
    train_lines[i][1].replace('•','\n')
    train_lines[i][1].replace('*','\n')
    open('tmp/train/'+ str(i+2)+'.txt','w+').write(train_lines[i][1]+'\n\n\n\n\n\n\n'+train_lines[i][0])