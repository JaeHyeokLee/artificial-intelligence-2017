#-*- coding: utf-8 -*-

import numpy as np
import random
import sys
import os.path as op
import math


train_list = []
test_list = []
salmon_train = []
seabass_train = []
salmon_test = []
seabass_test = []


fd = open('./salmon_train.txt', 'r')
lines = fd.readlines()
for line in lines:
	salmon_train.append(line.split())
for i in salmon_train:
	i.append('salmon')
fd.close()


fd = open('./seabass_train.txt', 'r')
lines = fd.readlines()
for line in lines:
	seabass_train.append(line.split())
for i in seabass_train:
	i.append('seabass')
fd.close()


fd = open('./salmon_test.txt', 'r')
lines = fd.readlines()
for line in lines:
	salmon_test.append(line.split())
for i in salmon_test:
	i.append('salmon')    
fd.close()

fd = open('./seabass_test.txt', 'r')
lines = fd.readlines()
for line in lines:
	seabass_test.append(line.split())
for i in seabass_test:
	i.append('seabass')
fd.close()

#전체 Train Data Set과 전체 Training Data Set을 만든다.
train_list = salmon_train + seabass_train
test_list = salmon_test + seabass_test


#설정한 개수만큼 유전자 리스트를 만들어주는 함수
#Linear Claasifier 의 parameter 인 a, b, c를 유전자로 선택
#유전자 초기값 설정 - 난수 생성
def make_chromosome(n):
	chromosome_list = []
	while len(chromosome_list) < n:
		new_a = random.uniform(0.0, 1.0)
		new_b = random.uniform(0.0, 1.0)
		new_c = random.uniform(10.0, 20.0)
		chromosome_list.append([new_a, new_b, new_c])
	return chromosome_list

#Parameter가 a, b, c 일때의 Error 를 구해주는 함수
def calc_error(input_list, a, b, c):
	true_value = 0
	false_value = 0
    
	for i in input_list:
		if a * float(i[0]) + b * float(i[1]) + c < 0:
			result = 'salmon'
		else:
			result = 'seabass'
        
		if i[2] == result:
			true_value += 1
		else:
			false_value += 1
    #오분류된 항목의 개수 반환    
	return false_value


def get_chromosome_error(train_list, chromosome_list):
	for i in chromosome_list:
		i.append(calc_error(train_list, i[0], i[1], i[2]))
	return chromosome_list



def selection(error_list, numOfElite, mut_prob):
	fitlist = []
	fitlist_reverse = []
	offspring_list = []
	fit_sum = 0
	for i in error_list:
		fitlist.append(i[3])
	for i in fitlist:
		fit_sum += 1.0/i
	for i in fitlist:    
		fitlist_reverse.append((1.0/i)/fit_sum)
	#Random Selection : np.random.choice 함수 이용!
	#오분류된 항목의 개수를 n이라 하면, 해당 개체수가 유전될 확률은 (해당 오류율)/Sum(1/n) 로 계산된다.
	for i in range(len(error_list)):
		temp = []
		# 유전 방법 : np.random.choice 함수 사용 - Random Choice 를 하는데, 해당 원소가 나올 확률을 정할 수 있다.
		# 우수한 개체가 많이 나올수 있도록 오분류된 항목에 역수를 취해서 합을 구한 다음, 해당 확률을 합으로 나눠서 확률 분포 리스트를 직접 만들어준다.
		if random.uniform(0.0, 1.0) >= mut_prob:
		#Mutation 방법 : 난수를 생성해서 설정한 mut_prob보다 작을 경우 mutation 실행
			temp.append(error_list[np.random.choice(len(fitlist), 1, p = fitlist_reverse)[0]][0]+random.uniform(-0.01, 0.01))
			temp.append(error_list[np.random.choice(len(fitlist), 1, p = fitlist_reverse)[0]][1]+random.uniform(-0.01, 0.01))
			temp.append(error_list[np.random.choice(len(fitlist), 1, p = fitlist_reverse)[0]][2]+random.uniform(-0.01, 0.01))
		#Mutatuon이 일어난 경우 : 0.5의 확률로 각 유전자에 변이가 일어난다.
		else:
			if random.uniform(0.0, 1.0) <= 0.5:
				temp.append(error_list[np.random.choice(len(fitlist), 1, p = fitlist_reverse)[0]][0]+random.uniform(-5, 5))
			else:
				temp.append(error_list[np.random.choice(len(fitlist), 1, p = fitlist_reverse)[0]][0]+random.uniform(-0.01, 0.01))
			if random.uniform(0.0, 1.0) <= 0.5:
				temp.append(error_list[np.random.choice(len(fitlist), 1, p = fitlist_reverse)[0]][1]+random.uniform(-5, 5))
			else:
				temp.append(error_list[np.random.choice(len(fitlist), 1, p = fitlist_reverse)[0]][1]+random.uniform(-0.01, 0.01))
			if random.uniform(0.0, 1.0) <= 0.5:
				temp.append(error_list[np.random.choice(len(fitlist), 1, p = fitlist_reverse)[0]][2]+random.uniform(-70, 70))
			else:
				temp.append(error_list[np.random.choice(len(fitlist), 1, p = fitlist_reverse)[0]][2]+random.uniform(-10, 10))
		offspring_list.append(temp)
	error_list.sort(key = lambda x : x[3])
	elite = error_list[:(numOfElite-1)]
	for i in range(numOfElite-1):
		offspring_list[i] = elite[i][:-1]
	return offspring_list



def genetic_algorithm(train_list,numOfChromosome, numOfElite, mut_prob):
	log_list = []
	count = 1
	chromosome_list = make_chromosome(numOfChromosome)
	error_list = get_chromosome_error(train_list, chromosome_list)
	error_list.sort(key = lambda x : x[3])
	best_error = error_list[0][3]
	best_param = [error_list[0][0], error_list[0][1], error_list[0][2]]
	print('Generating train_log_' + str(numOfChromosome) + '_' + str(numOfElite) + '_' + str(mut_prob) + '.txt\n')
	#학습 중단 판정 기준 : 오류율이 0.13 이하일 때
	while error_list[0][3] >=13:
		print(error_list[0])
		print(str(count) + ' - Best Score : ' + str(error_list[0][3]))
		log_list.append([count, error_list[0][3]])
		count += 1
		chromosome_list = selection(error_list, numOfElite, mut_prob)
		error_list = get_chromosome_error(train_list, chromosome_list)
		error_list.sort(key = lambda x : x[3])
		if error_list[0][3] < best_error:
			#best_error 와 best_paramter 저장
			best_error = error_list[0][3]
			best_param = [error_list[0][0], error_list[0][1], error_list[0][2]]
            
		#Train log Text 파일 생성
		filename = 'train_log_' + str(numOfChromosome) + '_' + str(numOfElite) + '_' + str(mut_prob) + '.txt'
		f = open(filename, 'w')
		f.write('=============Train Result==========\n')
		for line in log_list:
			f.write(str(line[0]) + ' : Best Error Rate : ' + str(line[1]/100.0) + '\n')
		f.close()
	print('Generating test_output_' + str(numOfChromosome) + '_' + str(numOfElite) + '_' + str(mut_prob) + '.txt\n')
	

	filename = 'test_output_' + str(numOfChromosome) + '_' + str(numOfElite) + '_' + str(mut_prob) + '.txt'
	print('Generating test_log_' + str(numOfChromosome) + '_' + str(numOfElite) + '_' + str(mut_prob) + '.txt')
	f = open(filename, 'w')
	f.write('=============Test Result==========\n')
	for i in test_list:
		if best_param[0] * float(i[0]) + best_param[1] * float(i[1]) + best_param[2] < 0:
			result = 'salmon'
		else:
			result = 'seabass'
		if i[2] == result:
			f.write('Length of Body : ' + str(i[0]) + ', Length of Tail : '+ str(i[1]) + ', Label: ' + i[2] +  ', Result : ' + result + ' -> True' + '\n')
			print('Length of Body : ' + str(i[0]) + ', Length of Tail : '+ str(i[1]) + ', Label: ' + i[2] +  ', Result : ' + result + ' -> True' + '\n')
		else:
			f.write('Length of Body : ' + str(i[0]) + ', Length of Tail : '+ str(i[1]) + ', Label: ' + i[2] +  ', Result : ' + result + ' -> False' + '\n')
			print('Length of Body : ' + str(i[0]) + ', Length of Tail : '+ str(i[1]) + ', Label: ' + i[2] +  ', Result : ' + result + ' -> False' + '\n')
    	f.write('Score - True : ' + str(100 - calc_error(test_list, best_param[0], best_param[1], best_param[2])) + ', False : ' + str(calc_error(test_list, best_param[0], best_param[1], best_param[2])))
    	print('Score - True : ' + str(100 - calc_error(test_list, best_param[0], best_param[1], best_param[2])) + ', False : ' + str(calc_error(test_list, best_param[0], best_param[1], best_param[2])))
	f.close()

#메인 소스코드에서 argument 로 전체 개체 수, elite 개체 수, mutation 확률을 받는다.
if __name__ == '__main__':
	argnum = len(sys.argv)
    #argumemt 의 개수가 4개이면 Genetic Algorithm 실행
	if argnum == 4:
		numOfChromosome = int(sys.argv[1])
		numOfElite = int(sys.argv[2])
		mut_prob = float(sys.argv[3])
		genetic_algorithm(train_list,numOfChromosome, numOfElite, mut_prob)
	else:
		print('Usage : %s [populationSize] [numOfelite] [mutationProb]') % (op.basename(sys.argv[0])) 