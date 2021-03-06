#!/bin/python3
import csv,time
import sys,math,random
import numpy as np 
# import pyautogui
# from Tkinter import Tk
import re

def csv_writer(data, path):
    with open(path, "wb") as file:
        file.write(data)

def csv_reader(csv_path):
    return list(csv.reader(open(csv_path, "rt"),delimiter='\t'))[1:]


train_data = csv_reader('train.tsv')
test_data = csv_reader('test.tsv')
train2_data = csv_reader('train2.tsv')
orig_tags = csv_reader('tmp/orig_tags.tsv')

n = len(train_data)
m = len(test_data)
all_tags = ['part-time-job','full-time-job','hourly-wage','salary','associate-needed','bs-degree-needed','ms-or-phd-needed',
  'licence-needed','1-year-experience-needed','2-4-years-experience-needed','5-plus-years-experience-needed','supervising-job']
all_tags_map = {'part-time-job':0,'full-time-job':0,'hourly-wage':1,'salary':1,'associate-needed':2,'bs-degree-needed':2,'ms-or-phd-needed':2,
  'licence-needed':2,'1-year-experience-needed':3,'2-4-years-experience-needed':3,'5-plus-years-experience-needed':3,'supervising-job':4}
n_tags = 12
importance = [True, True, True, False, True, False, True, True, True, False, True, True, False, False, False, False, True, True, False, False, False, True, True, True, True, False, False, True, False, False, True, True, True, True, False, False, True, False, False, True, True, True, True, False, True, True, True, True, True, False, True, True, True, True, False, False, True, True, True, False, True, True, True, False, False, False, False, True, True, False, True, True, True, True, True, True, True, False, False, True, False, True, False, True, True, True, True, True, True, True, False, True, True, True, True, False, False, False, True, False, False, True, True, True, False, False, True, False, True, False, False, True, False, False, False, False, True, False, True, True, True, False, False, True, False, True, True, False, False, True, True, True, True, True, True, True, False, False, True, False, False, False, True, False, True, True, False, False, True, False, False, False, False, True, False, False, True, False, True, True, False, True, False, False, True, True, True, True, True, True, True, False, False, False, False, False, False, True, True, True, False, False, False, False, True, True, False, True, True, False, True, True, False, True, False, False, False, True, True, True, False, False, True, True, False, False, True, True, False, False, False, True, True, True, True, True, False, True, True, False, True, False, True, True, True, True, False, True, True, True, True, False, False, True, True, False, True, False, False, False, False, True, True, False, True, True, False, False, False, True, True, False, True, False, False, False, True, True, True, False, True, False, True, True, False, False, False, True, True, False, True, True, False, True, True, False, True, True, False, True, False, True, True, False, True, True, True, True, True, True, False, True, True, True, True, True, False, False, False, True, True, True, True, True, True, True, True, False, True, True, False, True, True, False, False, True, True, True, False, False, False, False, False, False, True, True, True, False, True, False, False, True, True, True, False, True, False, False, False, True, False, False, True, False, True, True, True, False, False, True, True, True, True, False, False, False, False, False, False, False, True, True, True, True, False, False, False, False, False, True, True, False, True, False, False, False, True, True, False, False, False, True, True, False, False, False, False, False, False, True, False, True, False, False, True, False, True, False, False, True, True, True, False, False, False, False, False, True, False, False, True, True, True, False, True, False, True, False, True, True, False, False, False, False, True, False, True, False, False, True, True, True, False, True, True, True, True, True, True, True, True, False, False, False, False, True, False, True, False, False, False, True, False, True, True, True, True, False, False, False, True, True, True, False, True, False, True, True, False, False, False, False, False, True, False, False, True, True, False, True, False, False, True, True, False, False, False, True, True, False, False, True, False, True, False, False, True, False, False, False, False, False, False, False, False, True, False, False, False, True, True, True, True, False, False, False, False, False, False, True, True, True, True, True, True, False, False, False, True, False, False, True, False, False, False, False, True, True, True, True, True, False, False, True, False, False, True, True, True, False, False, True, False, False, True, True, True, False, False, False, True, False, False, True, False, True, True, False, False, True, False, False, True, True, False, False, False, True, False, False, True, True, False, True, False, True, False, True, False, False, True, True, True, False, False, True, False, False, False, True, False, True, True, False, False, True, True, False, True, True, False, True, True, False, False, False, True, False, True, True, False, False, False, True, True, False, True, True, False, True, True, False, True, False, False, True, False, False, True, True, False, True, False, False, True, False, False, False, False, False, False, True, True, True, True, True, False, False, True, False, True, True, True, True, False, False, True, True, True, False, False, False, True, False, False, True, False, True, False, True, True, False, True, False, True, False, True, False, False, True, True, False, True, False, True, False, True, True, True, True, False, False, False, False, True, True, True, False, False, True, True, True, True, False, True, True, True, False, True, True, True, False, True, True, False, False, True, True, False, True, True, False, False, True, True, False, True, False, False, False, False, False, False, True, False, False, True, True, True, True, False, True, False, True, True, False, True, True, True, True, False, False, False, False, True, False, False, False, False, True, False, True, False, False, True, True, True, False, False, False, True, False, True, True, False, True, True, False, True, False, True, True, False, True, True, False, True, True, False, False, True, True, True, True, True, True, False, False, False, True, True, True, False, False, False, False, False, True, True, False, False, True, True, False, False, True, False, False, True, False, False, True, False, True, False, False, False, True, False, True, False, False, False, False, False, False, False, True, False, True, True, False, True, True, True, True, False, False, True, True, False, False, False, True, False, False, True, False, False, False, False, True, False, False, True, False, False, True, True, True, False, True, True, False, True, True, False, True, False, False, True, True, True, True, True, True, True, False, False, True, True, True, False, False, False, True, False, False, False, False, False, False, True, False, True, False, False, False, False, True, True, False, False, False, True, False, False, False, True, False, True, False, False, True, True, True, True, True, False, False, True, False, True, False, False, False, True, True, False, True, False, False, True, True, True, True, True, True, False, True, False, False, False, False, False, True, True, True, True, True, False, False, True, False, False, True, False, True, True, True, False, True, False, True, True, False, True, True, True, False, False, False, True, True, False, False, True, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, False, False, True, True, True, False, False, False, True, True, False, False, True, False, True, True, True, True, False, True, True, True, False, True, True, False, False, True, False, False, False, True, False, True, True, True, False, False, True, False, True, True, True, True, False, True, False, True, False, True, False, False, True, False, False, True, False, False, True, False, False, True, False, False, True, True, False, False, True, False, True, False, False, True, True, True, False, False, False, False, False, False, False, True, True, True, True, False, False, True, True, True, False, False, False, False, True, False, True, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, True, True, False, False, False, False, True, True, False, True, False, True, False, False, True, True, True, False, False, True, True, True, True, True, True, True, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, False, False, True, True, False, True, False, True, True, True, False, True, False, False, True, True, False, True, False, False, True, True, False, False, True, False, False, False, True, True, True, True, True, True, True, False, False, True, True, True, False, True, True, False, False, True, True, True, True, False, True, True, False, False, True, True, False, True, False, True, True, False, False, False, False, False, False, False, True, False, False, False, True, False, False, True, True, True, True, False, False, True, False, False, False, True, False, False, False, False, True, False, False, False, False, False, True, False, False, True, True, True, False, False, True, False, True, True, True, False, True, False, False, True, False, False, True, False, True, True, False, False, False, False, False, False, True, True, False, False, True, True, True, False, False, False, False, False, False, False, True, False, True, True, False, True, False, False, False, False, True, True, False, True, False, False, True, True, False, True, True, False, False, True, False, True, False, False, False, True, False, False, True, True, True, False, True, True, False, True, True, False, False, False, True, True, True, False, False, False, True, True, True, True, False, False, False, False, False, True, True, False, True, False, False, False, True, False, True, False, False, True, True, False, False, False, True, False, True, False, True, True, False, False, False, True, False, False, False, False, False, False, False, False, True, False, True, True, True, False, True, True, True, True, True, True, True, False, False, False, True, False, True, False, False, True, True, True, True, True, True, True, True, True, True, True, True, False, True, False, False, True, True, True, False, True, False, True, False, False, False, True, False, True, True, False, False, False, True, False, False, True, True, True, False, False, True, False, False, True, False, False, True, False, False, False, False, True, True, False, False, True, True, True, False, True, False, True, True, False, True, False, False, False, True, False, False, False, True, False, True, False, True, True, False, True, True, False, False, False, False, True, True, True, True, True, False, False, False, True, False, True, True, False, False, True, False, True, False, True, True, True, False, False, True, False, True, False, True, False, False, True, True, True, True, True, True, False, True, True, True, True, True, True, False, True, False, True, True, False, False, False, False, False, True, True, True, True, True, True, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, True, True, True, False, True, False, True, False, True, False, False, True, False, False, True, False, False, False, True, True, True, True, True, False, False, True, True, True, False, False, False, False, True, False, False, True, False, False, False, False, True, True, True, False, False, False, True, True, False, False, False, False, True, False, True, False, True, True, True, False, True, False, True, False, False, False, True, True, True, False, True, True, False, False, True, True, True, True, False, False, True, False, False, True, False, True, False, False, True, True, False, False, True, True, True, True, True, True, True, False, True, True, True, True, False, False, True, True, False, True, False, True, False, False, True, False, False, False, True, False, True, False, True, False, False, False, False, True, False, True, False, False, False, False, False, False, True, True, True, False, False, False, True, False, False, True, True, False, True, False, False, True, True, False, True, False, False, False, False, False, False, False, False, True, False, True, False, True, False, True, True, True, True, False, True, False, True, True, True, False, True, False, True, False, True, False, True, False, False, False, False, True, True, False, True, False, False, True, False, True, False, True, True, True, True, False, False, True, True, True, False, True, True, False, True, True, False, False, False, True, True, False, True, False, True, False, True, False, False, True, True, True, False, False, True, True, False, True, False, True, True, False, True, True, False, True, True, False, True, True, False, True, False, False, False, False, True, True, True, True, True, True, True, False, True, False, False, True, True, True, False, True, True, True, True, True, False, True, True, True, False, False, False, True, True, True, False, False, True, True, False, False, True, True, True, False, False, True, False, False, True, False, False, True, False, False, False, False, False, False, False, True, False, False, True, False, False, True, True, True, True, False, True, True, True, False, True, False, True, False, False, False, False, False, True, False, True, True, True, True, True, False, True, False, True, False, True, False, True, True, True, False, False, True, True, False, True, False, True, True, True, False, True, True, False, True, True, False, True, False, False, True, True, False, False, False, False, False, True, True, True, True, True, True, True, True, False, True, True, True, False, False, False, False, True, False, True, True, True, True, False, False, False, False, False, False, True, True, True, False, False, False, False, True, False, True, True, True, True, True, False, False, False, True, True, False, False, True, True, False, True, True, True, False, True, True, True, True, True, True, True, False, False, True, False, True, True, True, False, True, True, True, True, False, False, True, False, True, True, False, False, True, True, False, True, False, False, False, False, True, False, True, False, False, False, False, True, False, False, True, False, False, True, False, True, False, False, False, True, False, False, False, True, True, True, True, True, False, False, False, True, True, True, False, True, False, True, True, True, False, False, True, False, True, True, False, False, False, True, True, True, True, True, True, True, True, True, True, True, False, False, False, True, True, True, True, False, True, True, True, True, True, True, False, True, True, False, False, True, True, True, False, False, True, True, False, False, False, True, True, False, True, True, True, False, True, True, True, False, True, False, True, True, True, True, False, False, False, True, True, False, True, False, False, False, False, True, True, True, True, False, True, True, True, False, False, False, True, True, True, False, True, True, True, False, True, False, True, False, False, False, False, True, True, True, False, True, False, True, True, False, True, True, False, False, False, True, True, True, True, True, False, True, False, True, False, False, True, True, True, False, False, True, False, False, True, True, True, False, False, True, True, True, False, True, True, False, False, False, True, True, True, True, False, True, True, False, False, False, False, False, True, False, True, False, True, True, True, True, False, True, True, True, True, True, True, True, False, False, True, True, False, False, False, True, False, False, True, True, True, False, True, False, False, False, True, False, False, False, True, True, True, True, False, True, False, True, False, True, True, False, False, True, True, False, False, False, False, False, True, False, False, True, False, False, False, True, False, False, False, False, True, False, False, False, True, True, True, True, True, True, True, False, False, False, True, True, False, False, False, True, True, True, True, False, True, False, False, False, False, True, True, True, True, True, True, True, True, True, False, False, False, True, False, True, False, True, True, False, False, True, True, True, True, True, False, True, True, True, True, True, True, False, False, False, False, False, False, True, True, False, False, True, True, False, False, False, True, True, True, True, False, False, False, True, False, False, True, True, True, False, False, True, False, False, False, False, False, False, True, False, False, True, False, True, False, True, True, False, True, True, True, False, True, True, True, False, False, False, True, True, False, True, True, True, False, True, False, True, False, True, False, True, False, True, False, False, True, False, True, True, False, False, True, False, True, True, False, False, True, True, False, False, True, True, True, True, True, True, False, True, True, False, False, True, False, True, True, False, False, True, False, True, True, True, False, False, True, False, False, True, True, True, True, True, True, True, False, True, False, True, False, False, False, True, False, False, False, False, False, True, False, True, False, False, True, False, True, True, True, True, True, False, True, True, True, True, True, True, True, False, False, False, False, False, False, True, False, False, True, True, True, True, False, True, False, True, False, True, False, False, False, True, True, True, True, False, False, True, False, False, True, True, False, True, True, True, False, True, True, True, True, False, True, True, True, False, True, True, False, False, False, False, True, False, False, True, True, False, False, True, False, False, False, False, True, False, True, True, False, True, False, True, False, True, False, False, False, False, False, False, True, True, False, True, False, True, False, False, False, True, True, True, True, False, False, False, False, False, False, False, False, True, True, False, True, False, False, False, False, False, True, True, False, False, False, True, False, False, True, True, True, False, True, True, True, False, True, False, True, True, True, False, True, True, True, True, True, True, False, True, True, True, False, True, True, True, False, True, True, True, True, False, False, False, True, True, False, False, False, True, True, False, True, False, True, False, True, True, True, True, False, False, False, False, False, False, True, True, False, False, False, False, False, False, True, True, False, True, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, True, False, True, False, True, False, True, True, True, False, False, False, True, False, True, True, True, True, False, False, False, False, True, False, False, True, True, False, True, False, True, False, False, True, False, False, True, False, True, True, False, False, True, True, False, True, True, False, True, False, False, True, True, True, False, False, False, False, False, False, True, False, False, False, False, False, True, False, True, False, False, True, False, True, False, False, True, True, False, True, False, True, True, True, True, False, False, True, True, True, False, False, True, True, True, False, False, False, False, True, False, True, False, False, True, True, True, False, False, False, True, False, False, True, True, False, True, False, False, True, True, False, False, False, True, True, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, True, False, True, False, True, False, True, True, True, False, True, False, True, False, False, False, False, True, True, False, True, False, True, True, False, False]
# index_of_k = [-1]*len(importance)
# ind = 0
# for i,x in enumerate(importance):
# 	if not x:
# 		index_of_k[i]=ind
# 		ind+=1
# print(train_data[0],len(train_data[0]))
# print(test_data[0])

def create_submission(tags):
	result = [ 'tags' ]
	for i,x in enumerate(tags):
		result.append( ' '.join(x) )
	csv_writer( '\n'.join(result)+'\n','tags.tsv')

def group(ind):
	if ind==0 or ind==1:
		return 0
	elif ind==2 or ind==3:
		return 1
	elif ind==4 or ind==5 or ind==6 or ind==7:
		return 2
	elif ind==8 or ind==9 or ind==10:
		return 3
	elif ind==11:
		return 4


def cost_function(stock_units):
	create_submission(stock_units)
	import clipboard

	clipboard.copy('abc')

	cur = pyautogui.position()
	pyautogui.click(x=511, y=599, clicks=1, button='left')
	pyautogui.moveTo(cur[0], cur[1]) # come back
	time.sleep(0.7)
	cur = pyautogui.position()
	pyautogui.click(x=1076, y=730, clicks=1, button='left')
	pyautogui.moveTo(cur[0], cur[1]) # come back
	time.sleep(0.8)
	cur = pyautogui.position()
	pyautogui.doubleClick(x=257, y=362)
	pyautogui.moveTo(cur[0], cur[1]) # come back
	time.sleep(6)
	patience = 0
	while True:
		cur = pyautogui.position()
		pyautogui.doubleClick(907, 754)
		pyautogui.hotkey('ctrl', 'c')
		pyautogui.moveTo(cur[0], cur[1]) # come back
		a = clipboard.paste()
		if re.match("^\d+?\.\d+?$", a) is None:
			if(patience==5):
				print('problem:',a)
				return -1,False 
		else:
			break
		patience+=1
		time.sleep(2.2)
	return float(a),True

def cost_available(ind,i):
	temp = map(lambda y:y.split(),open('tmp/output.txt','rt').readlines())
	temp = filter( lambda y: y[:2]==[str(ind),str(i)],temp)
	return list(map(lambda z:float(z[2]),temp))


def linear_opt(ind,end):
	sol_tags = [[all_tags[1]] for i in test_data]
	starting_score = 0.1383065613939559
	while ind<end:
		print('ind',ind)
		sol_tags[ind] = []
		base_cost,status=cost_function(sol_tags)
		if not status:
			return ind
		if base_cost == starting_score:
			print >> open("tmp/output.txt", "a"), 'actual_test_case',ind
		else:
			group_done = [False]*5
			for i,x in enumerate(all_tags):
				group_id = group(i)
				if group_done[group_id]:
					continue
				if i==1:
					print >> open("tmp/output.txt", "a"), ind,i,starting_score - base_cost,starting_score
					if starting_score - base_cost>0:
						group_done[group_id]=True
				else:	
					fc,cost = cost_available(ind,i),-1
					if len(fc)==0:
						sol_tags[ind] = [x]
						cost,status=cost_function(sol_tags)
						if not status:
							return ind
						print >> open("tmp/output.txt", "a"), ind,i,cost-base_cost,cost
					else:
						cost=fc[0]
					if cost-base_cost>0:
						group_done[group_id]=True

		sol_tags[ind] = [all_tags[1]]
		ind+=1
	return ind
	

def prepare_new_tab():
	pyautogui.click(x=203, y=347, clicks=1, button='left')
	pyautogui.hotkey('esc',)
	pyautogui.hotkey('ctrl', 'w')
	pyautogui.hotkey('ctrl', 't')
	pyautogui.typewrite('https://www.hackerrank.com/contests/indeed-ml-codesprint-2017/challenges/tagging-raw-job-descriptions')
	pyautogui.press('enter')
	pyautogui.moveTo(x=203, y=347)
	time.sleep(10)
	for x in xrange(1,20):
		pyautogui.scroll(-20)
		time.sleep(1)

def master_loop(rng):
	pyautogui.FAILSAFE = False

	ind = rng[0]
	while ind!=rng[1]:
		prepare_new_tab()
		ind = linear_opt(ind,rng[1])

# master_loop((889,1000))

# cost_function([[] for i in test_data])

# submission  = [['full-time-job'] for i in test_data]
# submission[0]=['full-time-job']
# create_submission(submission)


def rule_based_model():
	from model import text_extract
	labels = []
	descriptions = []
	for i,x in enumerate(train_data):
		descriptions.append(x[1])
		labels.append(x[0].split())


	_ = text_extract(descriptions,labels)

	test_descriptions = []
	for i,x in enumerate(test_data):
		test_descriptions.append(x[0])
	
	test_labels = text_extract(test_descriptions,[])
	create_submission(map(lambda x: [x], test_labels))


# rule_based_model()

def get_desc_of_type(tp):
	for i,x in enumerate(train_data):
		if tp in x[0]:
			print(x[1])


# get_desc_of_type(all_tags[-1])


# from model import prepare_manual
# prepare_manual(test_data,train_data)


def mlpredict(s):
	from model import learn
	descriptions,test_descriptions = [],[]
	labels = []
	for i,x in enumerate(train_data):
		descriptions.append(x[1])
		y_labels = [(1 if i in x[0].split(' ') else 0) for i in all_tags]
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
		labels.append(y_converted)


	if s!=1:
		descriptions, test_descriptions = descriptions[:int(n*s)],descriptions[int(n*s):]
		labels, test_labels = labels[:int(n*s)],labels[int(n*s):]
		predictions = learn(np.array(descriptions),np.array(labels),np.array(test_descriptions),np.array(test_labels))
	else:
		for i,x in enumerate(test_data):
			test_descriptions.append(x[0])
		predictions = learn(np.array(descriptions),np.array(labels),np.array(test_descriptions),np.array([]))
		submission = [[] for i in test_data]
		for i,x in enumerate(predictions):
			res_labels = []
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
			submission[i]=res_labels
		create_submission(submission)

#######################################################################################################################

def mlpredict2(s):
	global train_data
	from model import learn_simple,compress_labels,get_text_labels
	descriptions,test_descriptions = [],[]
	labels = []
	train_data+=train2_data
	for i,x in enumerate(train_data):
		descriptions.append(x[1])
		y_labels = [(1 if i in x[0].split(' ') else 0) for i in all_tags]
		labels.append(compress_labels(y_labels))


	if s!=1:
		descriptions, test_descriptions = descriptions[:int(len(train_data)*s)],descriptions[int(len(train_data)*s):]
		labels, test_labels = labels[:int(len(train_data)*s)],labels[int(len(train_data)*s):]
		predictions = learn_simple(np.array(descriptions),np.array(labels),np.array(test_descriptions),np.array(test_labels))
	else:
		for i,x in enumerate(test_data):
			test_descriptions.append(x[0])
		predictions = learn_simple(np.array(descriptions),np.array(labels),np.array(test_descriptions),np.array([]))
		submission = [[] for i in test_data]
		for i,x in enumerate(predictions):
			res_labels = get_text_labels(x)
			submission[i]=res_labels #if importance[i] else orig_tags[i]
		create_submission(submission) 

# print("testing logestic regression")
mlpredict2(1)


def reconstruct():
	m = len(test_data)
	tc = [False]*m
	train2 = [[] for i in range(m)]
	a= map( lambda x: x.split(), open("tmp/out.txt").readlines())
	for i,xx in enumerate(a):
		if xx[0][0]=="a":
			tc[int(xx[1])]=True
		else:
			if float(xx[2])>0:
				train2[int(xx[0])].append(all_tags[ int(xx[1]) ] )

	# for i in range(m):
	# 	if not tc[i]:
	# 		print( ' '.join( set(train2[i]) )+'\t'+test_data[i][0] )
	print tc

# reconstruct()

########################################################################################################


def most_common_near(word):
	
	from model import normalize_text
	from collections import Counter

	global train_data

	train_data = map( lambda xx: normalize_text(xx[1]) ,train_data)
	word = normalize_text(word)


	w_list =[]
	for xx in train_data:
		xx = re.findall(r"[\w']+", xx)
		for ii,yy in enumerate(xx):
			if word==yy:
				w_list+=xx[max(ii-10,0):max(ii+10,len(xx)-1)]
	counts = Counter(w_list)
	print( counts.most_common(50) )

# most_common_near('licence')


########################################################################################################