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
train_list = np.array([[0,0,0], [0,1,1], [1,0,1], [1,1,0]])


class NeuralNetwork:
    
    #Class 초기화 함수
    def __init__(self, input_list, hidden_node, numOfLayer, learning_rate):
        #input_list : List, Training Data Set
        #hidden_node : List, 각 hidden_layer의 node 개수
        #numOfLayer : Integer, 신경망의 깊이
        #leaning_rate : Float, 학습률
        #label_list : Test Data Set의 Label(0 또는 1)
        self.weight_list = []
        self.label_list = []
        self.output_list = []
        self.input_list = input_list
        self.hidden_node = np.insert(hidden_node, 0, len(self.input_list[0:2]))
        self.hidden_node = np.append(self.hidden_node, 1)
        self.numOfLayer = numOfLayer
        self.learning_rate = learning_rate 
        
        #True Label List 생성
        for i in self.input_list:
            self.label_list.append(i[2])
        self.label_list = np.array(self.label_list)
        
        #Node 의 개수에 맞게 Weight 초기화
        for i in range(numOfLayer+1):
            a = self.hidden_node[i]
            b = self.hidden_node[i+1]
            temp_list = []
            for j in range(b):
                temp_list.append(np.random.uniform(-1.4, 1.4, a+1))
            self.weight_list.append(temp_list)
    
    #Logistic Function
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    #하나의 perceptron의 내적 연산을 해 주는 함수
    #Input - Input_list : Training Data Set, weight_list : i번째 weight list, output : output node의 개수 
    #output - hidden node의 개수 만큼의 내적 연산 결과값
    def calc_perceptron(self, input_list, weight_list, output):
        temp = []
        temp_list_bias = np.insert(input_list, 0, 1)
        for i in range(len(weight_list)):
            temp.append(self.sigmoid(np.dot(temp_list_bias, weight_list[i])))
        return temp
    
    
    #Feedforward 함수 : 하나의 Training Data 가 신경망을 거쳐 나온 결과값 반환
    def feedforward(self, input_list):
        output = []
        output.append(input_list)
        activation = input_list[0:2]
        for i in range(self.numOfLayer + 1):
            #각 Layer의 output를 activation이라는 리스트 안에 저장한다.
            activation = self.calc_perceptron(activation , self.weight_list[i], self.hidden_node[i])
            output.append(activation)
        return output
    
    
    #BackPropagation - 강의 PPT에 나와 있는 Error Term 을 이용한 Weight Update 
    def backpropagation(self, input_list):
        activation_list = self.feedforward(input_list)
        #해당 Train Data Set 의 마지막 Output
        output = np.array(activation_list[-1])
        output_error = input_list[2] - output
        temp = np.insert(activation_list[-2], 0, 1)
        delta = output_error * output * (1.0 - output) * np.array(temp)
        self.weight_list[-1] += delta
        return self.weight_list
        
            
    #신경망의 오분류 항목의 개수를 출력해준다.
    def calc_error(self, label_list, output_list):
        true = 0
        false = 0
    
        for i in range(len(label_list)):
            if label_list[i] == output_list[i]:
                true += 1
            else:
                false += 1
        return false

if __name__ == '__main__':
    epoch = 0
    error_rate = 1.0
    best_error_rate = 1.0
    best_param = []
    log_list = []
    #클래스 객체 생성
    #레이어 3개, Hidden Node의 개수는 각각 2개, 2개, 3개
    #Learning Rate 는 0.1
    NN = NeuralNetwork(train_list,[2,2,3],3,0.1)
    print('=============Train Result==========\n')
    #오류율이 0이 될때까지 학습 진행
    while error_rate != 0:
        result_list = []
        for i in range(len(train_list)):
            weight_list = NN.backpropagation(train_list[i])
            #Multi Layer Perceptron Forward Propagation
            output_list = NN.feedforward(train_list[i])
            #result_list.append(output_list[-1])
            result_list += output_list[-1]
        #Output Layer 에서 최종 Sigmoid 함수의 출력값이 0.5 이상인 경우는 1인 것으로, 그렇지 않은 경우는 0인 것으로 판단.
        for i in range(len(result_list)):
            if result_list[i] > 0.5:
                result_list[i] = 1
            else:
                result_list[i] = 0
        error_rate = NN.calc_error(NN.label_list, result_list)
        log_list.append([epoch, error_rate/4.0])
        print('Epoch ' + str(epoch) + ' - Error Rate : ' + str(error_rate/4.0) + '\n')
        epoch += 1
        
        if error_rate < best_error_rate:
                best_weight = NN.weight_list
                best_error_rate = error_rate


    #Train Output 텍스트 파일로 저장
    filename = 'train_log.txt'
    f = open(filename, 'w')
    f.write('=============Train Result==========\n')
    for line in log_list:
        f.write('Epoch ' + str(line[0]) + ' - Error Rate : ' + str(line[1]) + '\n')
    f.close()
    test_result_list = []
    for i in range(len(train_list)):
        weight_list = NN.backpropagation(train_list[i])
            #Multi Layer Perceptron Forward Propagation
        output_list = NN.feedforward(train_list[i])
            #result_list.append(output_list[-1])
        test_result_list += output_list[-1]
        #Output Layer 에서 최종 Sigmoid 함수의 출력값이 0.5 이상인 경우는 1인 것으로, 그렇지 않은 경우는 0인 것으로 판단.
    for i in range(len(test_result_list)):
        if test_result_list[i] > 0.5:
            test_result_list[i] = 1
        else:
            test_result_list[i] = 0
        
        #Multi Layer Perceptron Forward Propagation
        #Output Layer 에서 최종 Sigmoid 함수의 출력값이 0.5 이상인 경우는 1인 것으로, 그렇지 않은 경우는 0인 것으로 판단.
    #test 결과를 텍스트 파일로 저장           
    filename = 'test_output.txt'
    f = open(filename, 'w')
    f.write('=============Test Result==========\n')
    print('=============Test Result==========\n')        

    for i in range(len(test_result_list)):
        if NN.label_list[i] == test_result_list[i]:
            f.write('Test Sample ' + str(i+1) + ' - Label : ' + str(NN.label_list[i]) + ', Result : ' + str(test_result_list[i]) + ' -> True\n')
            print('Test Sample ' + str(i+1) + ' - Label : ' + str(NN.label_list[i]) + ', Result : ' + str(test_result_list[i]) + ' -> True\n')
        else:
            f.write('Test Sample ' + str(i+1) + ' - Label : ' + str(NN.label_list[i]) + ', Result : ' + str(test_result_list[i]) + ' -> False\n')
            print('Test Sample ' + str(i+1) + ' - Label : ' + str(NN.label_list[i]) + ', Result : ' + str(test_result_list[i]) + ' -> False\n')
                
    error_rate = NN.calc_error(NN.label_list, test_result_list) / len(test_result_list)
    print('Error Rate : ' + str(error_rate))
    f.write('Error Rate : ' + str(error_rate))
    f.close()