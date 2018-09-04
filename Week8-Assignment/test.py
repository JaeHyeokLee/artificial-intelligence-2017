# -*- coding: utf-8 -*-

import os
import os.path as op
import struct as st
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle


#Logistic Function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


#Logistic Function 의 미분값
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Label의 Index에는 1, 나머지 원소에는 0을 넣어주는 one hot encoding를 해주는 함수
def make_one_hot(input_vector):
    return np.argmax(input_vector)


#분류기의 오류율을 계산해주는 함수. 오분류한 항목의 개수를 반환해준다.
#input : train data set, perceptron을 거쳐 나온 output vector(dot product 에 sigmoid 함수를 씌운 값)
def calc_error(input_list, output_vector):
    true = 0
    false = 0
    one_hot_vector = []
    for i in output_vector:
        one_hot_vector.append(make_one_hot(i))
    for i in range(len(one_hot_vector)):
        if input_list[i][1] == one_hot_vector[i]:
            true += 1
        else:
            false += 1
    return float(false)


#분류된 결과 리스트를 출력하는 함수
#Input : 1 * 785의 x_vector, 1 * 785의 weight_vector 10개
#output : 1 * 10의 output vector
def calc_output_vector(input_list, weight_list):
    output_list = []
    for j in range(len(weight_list)):
        #행렬의 내적 연산 실행
        output_list.append(sigmoid(np.dot(np.transpose(weight_list[j]), input_list[0])))
#         output_list.append(np.dot(np.transpose(weight_list[j]), input_list[0]))
    return np.array(output_list)

# weight의 변화량을 계산해주는 함수.
# (d_vector - o_vector) * o_vector * (1 - o_vector) * x_vector 까지 계산해준다.
#output : (785, 10) weight_vector 10개

def get_weight_delta(train_vector, output_vector, learning_rate):
    
    weight_delta = []

    true_label_vector = np.zeros(10)    
    true_label_vector[train_vector[1]] = 1.0
    temp = learning_rate * (true_label_vector - output_vector) * output_vector * (1.0 - output_vector)
    for i in temp:
        weight_delta.append(i * train_vector[0])
    return np.array(weight_delta)



# MNIST 데이터 경로
_SRC_PATH = './'
_TRAIN_DATA_FILE = _SRC_PATH + u'\\train-images.idx3-ubyte'
_TRAIN_LABEL_FILE = _SRC_PATH + u'\\train-labels.idx1-ubyte'
_TEST_DATA_FILE = _SRC_PATH + u'\\t10k-images.idx3-ubyte'
_TEST_LABEL_FILE = _SRC_PATH + u'\\t10k-labels.idx1-ubyte'

# MNIST 데이터 크기 (28x28)
_N_ROW = 28                 # 세로 28픽셀
_N_COL = 28                 # 가로 28픽셀
_N_PIXEL = _N_ROW * _N_COL

# 출력 이미지 경로
_DST_PATH = u'img_gray'


def drawImage(dataArr, fn):
    fig, ax = plt.subplots()
    ax.imshow(dataArr, cmap='gray')
    #plt.show()
    plt.savefig(fn)
    
    
    
def loadData(fn):
    print('loadData -' + fn)
    
    fd = open(fn, 'rb')
    
    # header: 32bit integer (big-endian)
    magicNumber = st.unpack('>I', fd.read(4))[0]
    nData = st.unpack('>I', fd.read(4))[0]
    nRow = st.unpack('>I', fd.read(4))[0]
    nCol = st.unpack('>I', fd.read(4))[0]
    
    print('magicNumber - ' + str(magicNumber))
    print('nData - ' + str(nData))
    print('nRow - ' + str(nRow))
    print('nCol - ' + str(nCol))
    
    # data: unsigned byte
    dataList = []
    for i in range(nData):
        dataRawList = fd.read(_N_PIXEL)
        dataNumList = st.unpack('B' * _N_PIXEL, dataRawList)
        dataArr = np.array(dataNumList).reshape(_N_ROW, _N_COL)
        dataList.append(dataArr.astype('int32'))
        
    fd.close()
    
    print('done.\n')
    
    return dataList
    


def loadLabel(fn):
    print('loadLabel - ' + str(fn))
    
    fd = open(fn, 'rb')
    
    # header: 32bit integer (big-endian)
    magicNumber = st.unpack('>I', fd.read(4))[0]
    nData = st.unpack('>I', fd.read(4))[0]
    
    print('magicNumber - ' + str(magicNumber))
    print('nData - ' + str(nData))
    
    # data: unsigned byte
    labelList = []
    for i in range(nData):
        dataLabel = st.unpack('B', fd.read(1))[0]
        labelList.append(dataLabel)
        
    fd.close()
    
    print('done.\n')
    
    return labelList



def loadMNIST():
    # 학습 데이터 / 레이블 로드
    trDataList = loadData(_TRAIN_DATA_FILE)
    trLabelList = loadLabel(_TRAIN_LABEL_FILE)
    
    # 테스트 데이터 / 레이블 로드
    tsDataList = loadData(_TEST_DATA_FILE)
    tsLabelList = loadLabel(_TEST_LABEL_FILE)
    
    return trDataList, trLabelList, tsDataList, tsLabelList    




if __name__ == '__main__':
    trDataList, trLabelList, tsDataList, tsLabelList = loadMNIST()
    
    print('len(trDataList) - ' + str(len(trDataList)))
    print('len(trLabelList) - ' + str(len(trLabelList)))
    print('len(tsDataList) - ' + str(len(tsDataList)))
    print('len(tsLabelList) - ' +str( len(tsLabelList)))
    
    if op.exists(_DST_PATH) == False:
        os.mkdir(_DST_PATH)
        
    # 샘플로 5개씩만 출력해보기
    for i in range(5):
        label = trLabelList[i]
        dstFn = _DST_PATH + u'\\tr_%d_label_%d.png' % (i, label)
        #print '%d-th train data: label=%d' % (i, label)
        drawImage(trDataList[i], dstFn)
        
        dstFn = _DST_PATH + u'\\tr_%d_label_%d.txt' % (i, label)
        np.savetxt(dstFn, trDataList[i], fmt='%4d')
        
    for i in range(5):
        label = tsLabelList[i]
        dstFn = _DST_PATH + u'\\ts_%d_label_%d.png' % (i, label)
        #print '%d-th test data: label=%d' % (i, label)
        drawImage(tsDataList[i], dstFn)
        
        dstFn = _DST_PATH + u'\\ts_%d_label_%d.txt' % (i, label)
        np.savetxt(dstFn, tsDataList[i], fmt='%4d')
    #Train Data 생성 - MNIST DATA 를 (784,1) 로 Flatten & Bias Term인 1 삽입 & Label 삽입

    test_list = []
    for i in range(len(tsDataList)):
        temp = (tsDataList[i]/255).reshape(784,)
        temp = np.insert(temp, 0, 1)
        temp_label = tsLabelList[i]
        test_list.append([temp, temp_label])


    #Best Parameter 불러오기
    fd = open('best_param.pkl', 'rb')
    loaded_param = pickle.load(fd)
    fd.close()

    test_result = []
    filename = 'test_output.txt'
    f = open(filename, 'w')
    f.write('=============Test Result==========\n')

    true = 0
    false = 0

    #Test Output 텍스트 파일로 저장
    for i in range(len(test_list)):
        #테스트 데이터셋 분류 시행
        label = test_list[i][1]
        result = make_one_hot(calc_output_vector(test_list[i], loaded_param))
        test_result.append(result)
        if label == result:
            f.write('Test Sample ' + str(i) + ' - Label : ' + str(label) + ', Result : ' + str(result) + ' -> True\n')
            true += 1
        else:
            f.write('Test Sample ' + str(i) + ' - Label : ' + str(label) + ', Result : ' + str(result) + ' -> False\n')
            false += 1
    f.write('Test Result : True - ' + str(true) + ', False - ' + str(false) + ', Error Rate - ' + str(float(false) / (false + true)) + '\n')
    print('Test Result : True - ' + str(true) + ', False - ' + str(false) + ', Error Rate - ' + str(float(false) / (false + true)) + '\n')
    f.close()