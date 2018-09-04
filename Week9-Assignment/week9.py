# -*- coding: utf-8 -*-

import os
import os.path as op
import struct as st
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle


#Train Data Set 생성 - 0번째 원소 1 : Bias Term, 1,2번째 원소 : Input X값
#3번째 원소 : Tran data Set 의 Label
train_list = np.array([[1,0,0,0], [1,0,1,1], [1,1,0,1], [1,1,1,0]])

#Logistic Function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

#분류기의 오류를 계산해주는 함수, 오분류 항목의 개수를 반환해준다.
def calc_error(train_list, output_list):
    true = 0
    false = 0
    
    for i in range(len(train_list)):
        if train_list[i][3] == output_list[i]:
            true += 1
        else:
            false += 1
    return false


#n번째 Layer 의 weight 초기화

_1st_weight_list = []
_2nd_weight_list = []

_1st_weight_list.append(np.random.uniform(-0.1, 0.1, 3))
_1st_weight_list.append(np.random.uniform(-0.1, 0.1, 3))
_2nd_weight_list = np.random.uniform(-0.1, 0.1, 3)
epoch = 0

if __name__ == '__main__':

    
    error_rate = 1.0
    best_error_rate = 1.0
    best_param = []
    label_list = []
    log_list1 = []
    for i in train_list:
        label_list.append(i[3])
    label_list = np.array(label_list)
    #오류율이 0이 될때까지 학습 진행
    while error_rate != 0:
        result_list = []
        _1st_output_vector = []
        _2nd_output_vector = []

        #Network의 구조가 간단해서 별도로 Train 함수를 만들지 않고 반복문을 이용했습니다.
        for i in range(len(train_list)):
            #Multi Layer Perceptron Forward Propagation
            _1st_output1 = sigmoid(np.dot(np.transpose(train_list[i][0:3]), _1st_weight_list[0]))
            _1st_output2 = sigmoid(np.dot(np.transpose(train_list[i][0:3]), _1st_weight_list[1]))
            _1st_output_vector = np.array([1.0, _1st_output1, _1st_output2])
            _2nd_output_vector = sigmoid(np.dot(np.transpose(_1st_output_vector), _2nd_weight_list))

            #Output Layer 에서 최종 Sigmoid 함수의 출력값이 0.5 이상인 경우는 1인 것으로, 그렇지 않은 경웅는 0인 것으로 판단.
            if _2nd_output_vector >= 0.5:
                result_list.append(1.0)
            else:
                result_list.append(0.0)

            output_error = label_list[i] -  _2nd_output_vector
            _2nd_error_term = (output_error * _2nd_output_vector * (1.0 - _2nd_output_vector))

            #BackPropagation - Weight Update
            _2nd_weight_list += _2nd_error_term * _1st_output_vector
            # Output Layer의 출력은 
            _1st_weight_list[0] += _2nd_error_term * (1.0 - _1st_output1) * _1st_output1 * _2nd_weight_list[1] * train_list[i][0:3]
            _1st_weight_list[1] += _2nd_error_term * (1.0 - _1st_output2) * _1st_output2 * _2nd_weight_list[2] * train_list[i][0:3]
        error_rate = calc_error(train_list, result_list)
        log_list1.append([epoch, error_rate/4.0])
        print('Epoch ' + str(epoch) + ' - Error Rate : ' + str(error_rate/4.0) + '\n')
        epoch += 1
        if error_rate < best_error_rate:
                best_weight = [_1st_weight_list, _2nd_weight_list]
                best_error_rate = error_rate


    #Train Output 텍스트 파일로 저장
    filename = 'train_log.txt'
    f = open(filename, 'w')
    f.write('=============Train Result==========\n')
    for line in log_list1:
        f.write('Epoch ' + str(line[0]) + ' - Error Rate : ' + str(line[1]) + '\n')
    f.close()


    test_result_list = []
    test_1st_output_vector = []
    test_2nd_output_vector = []

    #Network의 구조가 간단해서 별도로 Train 함수를 만들지 않고 반복문을 이용했습니다.
    for i in range(len(train_list)):
        #Multi Layer Perceptron Forward Propagation
        test_1st_output1 = sigmoid(np.dot(np.transpose(train_list[i][0:3]), best_weight[0][0]))
        test_1st_output2 = sigmoid(np.dot(np.transpose(train_list[i][0:3]), best_weight[0][1]))
        test_1st_output_vector = np.array([1.0, test_1st_output1, test_1st_output2])
        test_2nd_output_vector = sigmoid(np.dot(np.transpose(test_1st_output_vector), best_weight[1]))

        #Output Layer 에서 최종 Sigmoid 함수의 출력값이 0.5 이상인 경우는 1인 것으로, 그렇지 않은 경우는 0인 것으로 판단.
        if test_2nd_output_vector >= 0.5:
            test_result_list.append(1)
        else:
            test_result_list.append(0)
    #test 결과를 텍스트 파일로 저장           
    filename = 'test_output.txt'
    f = open(filename, 'w')
    f.write('=============Test Result==========\n')
    print('=============Test Result==========\n')        

    for i in range(len(test_result_list)):
        if label_list[i] == test_result_list[i]:
            f.write('Test Sample ' + str(i+1) + ' - Label : ' + str(label_list[i]) + ', Result : ' + str(test_result_list[i]) + ' -> True\n')
            print('Test Sample ' + str(i+1) + ' - Label : ' + str(label_list[i]) + ', Result : ' + str(test_result_list[i]) + ' -> True\n')
        else:
            f.write('Test Sample ' + str(i+1) + ' - Label : ' + str(label_list[i]) + ', Result : ' + str(test_result_list[i]) + ' -> False\n')
            print('Test Sample ' + str(i+1) + ' - Label : ' + str(label_list[i]) + ', Result : ' + str(test_result_list[i]) + ' -> False\n')
                
    error_rate = calc_error(train_list, test_result_list) / len(train_list)
    print('Error Rate : ' + str(error_rate))
    f.write('Error Rate : ' + str(error_rate))
    f.close()