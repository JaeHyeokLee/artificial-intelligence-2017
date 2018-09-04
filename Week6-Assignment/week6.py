#-*- coding: utf-8 -*-

import numpy as np
import random
import sys
import os.path as op

train_list = []
test_list = []
salmon_train = []
seabass_train = []
salmon_test = []
seabass_test = []

#salmon일 경우는 0, seabass의 경우는 1로 Labeling 한다.
fd = open('./salmon_train.txt', 'r')
lines = fd.readlines()
for line in lines:
    salmon_train.append(line.split())
for i in salmon_train:
    #원활한 벡터 연산을 위해서 x0성분에 1을 넣어준다.
    i.append(1)
    i.append(0)
fd.close()


fd = open('./seabass_train.txt', 'r')
lines = fd.readlines()
for line in lines:
    seabass_train.append(line.split())
for i in seabass_train:
    #원활한 벡터 연산을 위해서 x0성분에 1을 넣어준다.
    i.append(1)
    i.append(1)
fd.close()


fd = open('./salmon_test.txt', 'r')
lines = fd.readlines()
for line in lines:
    salmon_test.append(line.split())
for i in salmon_test:
    #원활한 벡터 연산을 위해서 x0성분에 1을 넣어준다.
    i.append(1)    
    i.append(0)
fd.close()

fd = open('./seabass_test.txt', 'r')
lines = fd.readlines()
for line in lines:
    seabass_test.append(line.split())
for i in seabass_test:
    #원활한 벡터 연산을 위해서 x0성분에 1을 넣어준다.
    i.append(1)
    i.append(1)
fd.close()

#전체 Train Data Set과 전체 Training Data Set을 만든다.
train_list = salmon_train + seabass_train
test_list = salmon_test + seabass_test

#선형 분류기의 Error를 계산해주는 함수
def calc_error(input_list, param_list):
    #변수 초기화
    result = 0
    true_value = 0
    false_value = 0
    for i in input_list:
        #선형 분류기를 이용해서 Salmon일 경우는 결과값에 0을, Seabass일 경우는 결과값에 1을 넣어준다.
        if param_list[0] * float(i[0]) + param_list[1] * float(i[1]) + param_list[2] * float(i[2]) < 0.0:
            result = 0
        else:
            result = 1
            
        if i[3] == result:
            true_value += 1
        else:
            false_value += 1
    return false_value
#인자값이 0보다 클 경우는 1, 0보다 작거나 같을 경우는 0을 출력한다.
def G(i):
    if i > 0.0:
        return 1
    else:
        return 0

        #Perceptron 학습 알고리즘
def learn(input_list, param_list, learning_rate):
    #선형 분류기의 학습값
    #[w0, w1, w2]^T * [x0, x1, x2] 의 Dot Product 연산
    result = param_list[0] * float(input_list[0]) + param_list[1] * float(input_list[1]) + param_list[2] * float(input_list[2])
    #new_w_vector = w_Vector + learning_rate * (True_Label - O(Result of Dot Product)) * x_Vector
    new_a = float(param_list[0]) + learning_rate * (input_list[3] - G(result)) * float(input_list[0])
    new_b = float(param_list[1]) + learning_rate * (input_list[3] - G(result)) * float(input_list[1])
    new_c = float(param_list[2]) + learning_rate * (input_list[3] - G(result)) * float(input_list[2])
    param_list = [new_a, new_b, new_c]
    return param_list

#Perceptrom 을 구현한 함수
def perceptron(input_list, learning_rate):
    count = 1
    i = 1
    log_list = []
    #파라미터 초기값 부여 :  랜덤 난수 발생
    a = random.uniform(0.0, 1.0)
    b = random.uniform(0.0, 1.0)
    c = random.uniform(0.0, 1.0)
    param_list = [a, b, c]
    #원활할 학습을 위해 Train Data Set 를 Random하게 섞어준다.
    random.shuffle(input_list)
    error = calc_error(input_list, param_list)
    log_list.append([count, error, param_list])
    print(str(count) + ' - Error : ' + str(error))
    print(param_list)
    print('\n')
    #가장 우수한 선형 분류기의 정보 저장
    best_error = error
    best_param = param_list
    #학습 중단 판정 기준 : 오분류 항목의 개수가 13개 이하일 경우 학습 종료
    while error >= 13:
        count += 1
        i = i%100
        #설정한 학습 중단 기준을 충족시키지 못하면 Parameter를 계속 학습시킨다.
        param_list = learn(input_list[i], param_list, learning_rate)
        error = calc_error(input_list, param_list)
        log_list.append([count, error, param_list])
        print(str(count) + ' - Error : ' + str(error))
        print('Parameter')
        print(param_list)
        print('\n')
        if best_error > error:
            best_error = error
            best_param = param_list
        i+= 1
    
    #Train Output 출력
    filename = 'train_log_' + str(learning_rate) + '.txt'
    f = open(filename, 'w')
    f.write('=============Train Result==========\n')
    for line in log_list:
        f.write(str(line[0]) + ' - Error : ' + str(line[1]) + '\n')
        f.write('Parameter\n')
        f.write(str(line[2]) + '\n\n')
    f.close()
    
    filename = 'test_output_' + str(learning_rate) + '.txt'
    f = open(filename, 'w')
    f.write('=============Test Result==========\n')

    #Test Output 출력
    for i in test_list:
        error = calc_error(test_list, best_param)
        if i[3] == 1:
            true_label = 'Salmon'
        else:
            true_label = 'Seabass'
        
        if best_param[0] * float(i[0]) + best_param[1] * float(i[1]) + best_param[2] * float(i[2]) < 0:
            result = 0
            classified_label = 'Salmon'
        else:
            result = 1
            classified_label = 'Seabass'
        if i[3] == result:
            f.write('Length of Body : ' + str(i[0]) + ', Length of Tail : '+ str(i[1]) + ', Label: ' + true_label +  ', Result : ' + classified_label + ' -> True' + '\n')
            print('Length of Body : ' + str(i[0]) + ', Length of Tail : '+ str(i[1]) + ', Label: ' + true_label +  ', Result : ' + classified_label + ' -> True' + '\n')
        else:
            f.write('Length of Body : ' + str(i[0]) + ', Length of Tail : '+ str(i[1]) + ', Label: ' + true_label +  ', Result : ' + classified_label + ' -> False' + '\n')
            print('Length of Body : ' + str(i[0]) + ', Length of Tail : '+ str(i[1]) + ', Label: ' + true_label +  ', Result : ' + classified_label + ' -> False' + '\n')
    f.write('Score - True : ' + str(100 - error) + ', False : ' + str(error))
    print('Score - True : ' + str(100 - error) + ', False : ' + str(error))
    f.close()

#메인 소스코드에서 argument 로 Learning Rate를 받는다.
if __name__ == '__main__':
    argnum = len(sys.argv)
    #Argumemt의 개수가 2개이면 Perceptron 실행
    if argnum == 2:
        learning_rate = float(sys.argv[1])
        perceptron(train_list, learning_rate)
    else:
        print('Usage : %s [Learning Rate]') % (op.basename(sys.argv[0])) 