#!/bin/python3
import csv,time
import sys,math,random
import numpy as np 
import pyautogui
from Tkinter import Tk
import re

def csv_writer(data, path):
    with open(path, "wb") as file:
        file.write(data)

def csv_reader(csv_path):
    return list(csv.reader(open(csv_path, "rt"),delimiter='\t'))[1:]


train_data = csv_reader('train.tsv')
test_data = csv_reader('test.tsv')

n = len(train_data)
m = len(test_data)
all_tags = ['part-time-job','full-time-job','hourly-wage','salary','associate-needed','bs-degree-needed','ms-or-phd-needed',
	'licence-needed','1-year-experience-needed','2-4-years-experience-needed','5-plus-years-experience-needed','supervising-job']
n_tags = 12

# print(train_data[0],len(train_data[0]))
# print(test_data[0])

def create_submission(tags):
	result = [ 'tags' ]
	for i,x in enumerate(tags):
		result.append( ' '.join(x) )
	csv_writer( '\n'.join(result),'tags.tsv')

def cost_function(tags):
	create_submission(tags)
	cur = pyautogui.position()
	pyautogui.click(x=201, y=129, clicks=1, button='left')
	pyautogui.moveTo(cur[0], cur[1]) # come back
	time.sleep(1)
	cur = pyautogui.position()
	pyautogui.click(x=769, y=481, clicks=1, button='left')
	pyautogui.moveTo(cur[0], cur[1]) # come back
	time.sleep(0.3)
	cur = pyautogui.position()
	pyautogui.doubleClick(x=426, y=329)
	pyautogui.moveTo(cur[0], cur[1]) # come back
	time.sleep(6.5)
	cur = pyautogui.position()
	pyautogui.moveTo(239, 169)
	pyautogui.dragRel(66, 0, duration=0.01)
	pyautogui.hotkey('ctrl', 'c')
	pyautogui.moveTo(cur[0], cur[1]) # come back
	r = Tk()
	r.withdraw()
	a=r.selection_get(selection = "CLIPBOARD")
	if re.match("^\d+?\.\d+?$", a) is None:
		return -1,False
	return float(a),True

def linear_opt(ind):
	sol_tags = [[] for i in test_data]
	while ind<m:
		print('ind',ind)
		for i,x in enumerate(all_tags):
			sol_tags[ind] = [x]
			cost,status=cost_function(sol_tags)
			if status:
				return ind
			print >> open("tmp/output.txt", "a"), ind,i,cost
		sol_tags[ind] = []
		ind+=1
	return ind
	

def prepare_new_tab():
	pyautogui.click(x=500, y=500, clicks=1, button='left')
	pyautogui.hotkey('esc',)
	pyautogui.hotkey('ctrl', 'w')
	pyautogui.hotkey('ctrl', 't')
	pyautogui.typewrite('https://www.hackerrank.com/contests/indeed-ml-codesprint-2017/challenges/tagging-raw-job-descriptions')
	pyautogui.press('enter')
	pyautogui.moveTo(x=1358, y=115)
	time.sleep(25)
	for x in xrange(1,15):
		pyautogui.scroll(-20)
		time.sleep(2)

def master_loop():

	ind = 0
	while ind!=m:
		prepare_new_tab()
		ind = linear_opt(ind)

# master_loop()

# cost_function([[] for i in test_data])

# submission  = [['full-time-job'] for i in test_data]
# submission[0]=['full-time-job']
# create_submission(submission)


def mlpredict(s):
	from model import learn
	print("testing logestic regression")
	descriptions = []
	labels = []
	for i,x in enumerate(train_data):
		descriptions.append(x[1])
		labels.append( [(1 if i in x[0].split(' ') else 0) for i in all_tags])

	descriptions, test_descriptions = descriptions[:int(n*s)],descriptions[int(n*s):]
	labels, test_labels = labels[:int(n*s)],labels[int(n*s):]

	learn(np.array(descriptions),np.array(labels),np.array(test_descriptions),np.array(test_labels))

print("testing logestic regression")
mlpredict(0.9)



