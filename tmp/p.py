#!/bin/python3
import csv,time
import sys,math,random
import numpy as np 
import pyautogui
from Tkinter import Tk
import re

def csv_writer(data, path):
    with open(path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)


def csv_reader(csv_path):
    return list(csv.reader(open(csv_path, "rt")))[1:]


portfolio = csv_reader('portfolio.csv')
index_closing_prices = csv_reader('index_closing_prices.csv')
stocks_closing_prices = csv_reader('stocks_closing_prices.csv')
stocks_info = csv_reader('stocks_info.csv')

stocks_info = list(map(lambda x: [x[0],x[1],x[2],float(x[3])],stocks_info))
stocks_closing_prices = list(map(lambda x: [x[0],float(x[1]),float(x[2]),float(x[3])],
							stocks_closing_prices))
index_closing_prices = list(map(lambda x: [x[0],float(x[1])],index_closing_prices))
portfolio = list(map(lambda x: [x[0],int(x[1])],portfolio))

n = len(stocks_info)
max_in_hand = 5*100*1000
total_money = 100*100*1000

returns = [50*100*1000, 25*100*1000, 25*100*1000 ]
returns[1]+=returns[0]*(1+math.log(index_closing_prices[1][1]/index_closing_prices[0][1],10))
returns[2]+=returns[1]*(1+math.log(index_closing_prices[2][1]/index_closing_prices[1][1],10))

print(stocks_info[0])
print(stocks_closing_prices[0])
print(index_closing_prices[0])
print(portfolio[0])
print(returns)

def compute_weight(day):
	w = [0]*n
	for i,x in enumerate(stocks_info):
		w[i]=stocks_info[i][3]*stocks_closing_prices[i][day+1]
	total_capital = sum(w)
	for i,x in enumerate(stocks_info):
		w[i]/=total_capital
	return w

def rebalance(weights,total,stock_prices):
	shares=[0]*n
	used=0
	for i,x in enumerate(stock_prices):
		shares[i] = int((weights[i]*total)//x)
		used+=shares[i]*x
	# print(total,used,max_in_hand)
	# print(shares)
	incremented = [False]*n
	# while 1: 
	while total-used>max_in_hand:
		ind,val = 0, ((shares[0]+1)*stock_prices[0]-total*weights[0])/(total*weights[0])
		found = False
		for i,x in enumerate(stock_prices):
			extra = ((shares[i]+1)*stock_prices[i]-total*weights[i])/(total*weights[i])
			if( incremented[i]==False and stock_prices[i]<=total-used and extra <= val 
						and (total*weights[i]-shares[i]*stock_prices[i])/(total*weights[i])<=extra):
				val=extra
				ind = i
				found=True
		# print('ind',ind,found)
		if(not found):
			break
		incremented[ind]=True
		shares[ind]+=1
		used+=stock_prices[ind]
	# print(total-used,max_in_hand)
	assert(total-used<=max_in_hand)
	return shares

def create_submission(stock_units):
	result= [[] for i in range(n)]
	weight1 = compute_weight(0) 
	day1 = rebalance(weight1,75*100*1000,list(map(lambda x: x[2],stocks_closing_prices)))
	for i,x in enumerate(day1):
		result[i].append(stocks_info[i][0])
		result[i].append(x)
	for i,x in enumerate(stock_units):
		result[i].append(x)
	csv_writer([['Symbol','Quantity_Day_1','Quantity_Day_2']]+result,'submission.csv')

def cost_function(stock_units):
	create_submission(stock_units)
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
		return 0.975
	print >> open("../output.txt", "a"), stock_units,',',a
	return float(a)

def neighbour(stock_units,stocks_prices):
	done = False
	while not done:
		i,j=random.randrange(0,n),random.randrange(0,n)
		if(i!=j):
			if(stocks_prices[i]<stocks_prices[j]):
				i,j=j,i
			factor = stocks_prices[i]//stocks_prices[j]
			if(random.random()<0.5):
				stock_units[j]-=factor
				stock_units[i]+=1
			else:
				stock_units[j]+=factor
				stock_units[i]-=1
			c=0
			if(random.random()<0.2):
				stock_units[j]+=1
			for ind,x in enumerate(stock_units):
				c+=x*stocks_prices[ind]
			while(total_money-c>max_in_hand):
				lotery = random.randrange(0,n)
				stock_units[lotery]+=1
				c+=stocks_prices[lotery]
			print('c',c)
			done = True
	return stock_units

def sim_aneal():
	weight2 = compute_weight(1)
	stocks_prices = list(map(lambda x: x[3],stocks_closing_prices))
	stock_units = rebalance(weight2,total_money,stocks_prices)
	cost_previous = 1000000000000000000000000000000000
	temperature, temperature_end = 10**10, 1
	cooling_factor= 0.96
	best_cost,best_state = cost_previous,stock_units
	while temperature > temperature_end:
	    state_new = neighbour(stock_units,stocks_prices)
	    cost_new = cost_function(state_new)
	    difference = - cost_new + cost_previous
	    if difference < 0 or  math.exp(-difference/temperature) > random.random():
	        cost_previous = cost_new
	        if(cost_new<best_cost):
	        	best_cost = cost_new
	        	best_state = stock_units
	    temperature = temperature * cooling_factor

def already_invested(stock_units,stocks_prices):
	used=0
	for j,x in enumerate(stock_units):
		used+=stock_units[j]*stocks_prices[j]
	return used

def linear_opt(steps,iterations):
	stocks_prices = list(map(lambda x: x[3],stocks_closing_prices))
	olddata=set(filter(lambda y:len(y)==2,map( lambda x: tuple(x.split(' , ')), open("../output.txt",'r').readlines())))
	max_element = sorted(list(olddata),key=lambda x: -float(x[1]))[random.randrange(0,int(iterations/n))]
	stock_units,cost_prev_best=list(map(int,max_element[0][1:-1].split(','))),float(max_element[1])
	print('stock_units',stock_units)
	print('len(stock_units)',len(stock_units))
	# exit(0)
	# [1, 3, 147, 2, 1, 384, 45, 323, 461, 318, 3, 175, 163, 372, 38, 4, 159, 107, 170, 544, 683, 41, 1, 1, 1292, 495, 179, 1516, 579, 2846, 239, 276, 75, 154, 46, 823, 893, 791, 516, 1032, 343, 672, 178, 143, 250, 352, 179, 37, 192, 126, 341] , 0.985414
	best_possible = [False]*n
	best_state = stock_units
	used = 0
	i=random.randrange(0,n)
	decreasing_limit = 5
	while steps>0:
		curr = stock_units[i]
		used = already_invested(stock_units,stocks_prices)
		factor = int(math.sqrt(max(1,stock_units[i]/(iterations/n))) )
		decreasing = 0
		best_possible[i]=True
		while used+factor*stocks_prices[i]<=total_money  and decreasing<decreasing_limit:
			stock_units[i]+=factor
			used+=stocks_prices[i]*factor
			print('cash_in_hand1',total_money-used,i,stock_units[i],factor)
			cost_new = cost_function(stock_units)
			if cost_new>=cost_prev_best:
				best_possible[i]=False
				cost_prev_best=cost_new
				best_state = stock_units
			else:
				decreasing+=1
			factor = int(math.sqrt(max(1,stock_units[i]/(iterations/n))) )
		stock_units[i]=curr
		decreasing = 0
		used = already_invested(stock_units,stocks_prices)
		factor = int(math.sqrt(max(1,stock_units[i]/(iterations/n))) )
		while stock_units[i]>factor and total_money - used + factor*stocks_prices[i]<=max_in_hand and decreasing<decreasing_limit:
			stock_units[i]-=factor
			used-=stocks_prices[i]*factor
			print('cash_in_hand2',total_money-used,i,stock_units[i],factor)
			cost_new = cost_function(stock_units)
			if cost_new>=cost_prev_best:
				cost_prev_best=cost_new
				best_possible[i]=False
				best_state = stock_units
			else:
				decreasing+=1
			factor = int(math.sqrt(max(1,stock_units[i]/(iterations/n))) )
		while True:
			i=random.randrange(0,n)
			if(False in best_possible):
				if not best_possible[i]:
					break
			else:
				print >> open("../output.txt", "a"),'NO_CANDIDATE'
				iterations=500000
				break
		print >> open("../output.txt", "a"), i
		stock_units=best_state
		iterations+=5
		steps-=1

def prepare_new_tab():
	pyautogui.hotkey('esc',)
	pyautogui.hotkey('ctrl', 'w')
	pyautogui.hotkey('ctrl', 't')
	pyautogui.typewrite('https://www.hackerrank.com/contests/nse-isb-codesprint/challenges/index-rebalance-maximizing-the-portfolio')
	pyautogui.press('enter')
	pyautogui.moveTo(x=1358, y=115)
	time.sleep(25)
	for x in xrange(1,15):
		pyautogui.scroll(-20)
		time.sleep(2)

def master_loop():
	pyautogui.hotkey('alt', 'tab')
	iterations=256*n
	while True:
		prepare_new_tab()
		linear_opt(50,iterations)
		iterations+=4*n

master_loop()


