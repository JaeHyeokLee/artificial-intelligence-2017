# -*- coding: utf-8 -*-
#필요한 패키지 로드
import numpy as np
import matplotlib.pyplot as plt
import random

#데이터 파일 불러오기
temp = []
f = open("donut.txt", 'r')
lines = f.readlines()
for line in lines:
    temp.append(line)
f.close()

donut = []
for i in temp:
    donut.append(i.split())
#Float Data Type를 가지는 np.array로 변환
donut = np.array(donut).astype(np.float)

#2차원 Gaussian Distribution의 PDF를 구하는 함수
def get_pdf(input_vector, mean_vector, cov):
    inv = np.linalg.inv(cov)
    dif = (input_vector - mean_vector)
    temp = np.matmul(dif.T, inv)
    temp = np.matmul(temp, dif)
    return np.exp(-0.5 * temp) /((2*np.pi) * np.power(np.linalg.det(cov), 0.5))

#공분산행렬, 평균이 주어졌을때, 어떤 군집에 속하게 될 지 분류해주는 함수
def cluster(input_vector, mean_vector_list, cov_list, prob_list):
    output_list = []
    for i in range(len(mean_vector_list)):
        #PDF의 값과 각 분포에 들어갈 확률을 곱해준다.
        output_list.append(prob_list[i] * get_pdf(input_vector, mean_vector_list[i], cov_list[i]))
        
    return np.argmax(output_list)

#분포들의 초기 평균 & 분산 초기화
x_mean = np.mean(donut[:,0])
y_mean = np.mean(donut[:,1])
sd = [np.std(donut[:,0]), np.std(donut[:,1])]

u1 = [x_mean + random.uniform(0, 1), y_mean + random.uniform(0, 1)]
u2 = [x_mean - random.uniform(0, 1), y_mean + random.uniform(0, 1)]
u3 = [x_mean + random.uniform(0, 1), y_mean - random.uniform(0, 1)]
u4 = [x_mean - random.uniform(0, 1), y_mean - random.uniform(0, 1)]
cov = np.diag(sd)

mean_list = [u1, u2, u3, u4]
cov_list = [cov, cov, cov, cov]

def EM(input_list, mean_vector_list, cov_list):
    mean_list = mean_vector_list
    cov_list1 = cov_list
    #각 분포의 가중치 초기화
    prob_list = [0.25, 0.25, 0.25, 0.25]
    aa = 0
    #30번 반복
    while aa < 30:
        aa += 1
        cluster_list = []
        input_subset = []
        clustered_index = []
        #print(cov_list1[0])
        for i in range(len(input_list)):
            clustered_index.append(cluster(input_list[i], mean_list, cov_list1, prob_list))
        prob_list = []
        #Clustering된 항목 수를 전체 수로 나누어 가중치 업데이트
        for i in range(len(mean_list)):
            prob_list.append((np.array(clustered_index) == i).sum())
        #기댓값을 전체 데이터 수로 나누어 확률 업데이트
        prob_list = np.array(prob_list) / float(len(input_list))
        #print(prob_list)
        #항목 분류
        for i in range(len(mean_list)):
            input_subset.append(input_list[np.where(np.array(clustered_index) == i)])        
        cov_list1 = []
        mean_list = []
        for i in input_subset:
            temp = []
            #각 분포에 속하는 데이터셋의 평균과 분산으로 다음 파라미터 업데이트
            temp.append(np.mean(np.array(i)[:,0]))
            temp.append(np.mean(np.array(i)[:,1]))
            mean_list.append(temp)
            cov = [np.std(np.array(i)[:,0]), np.std(np.array(i)[:,1])]
            #print(cov)
            cov_list1.append(np.diag(cov))
    print('centroid list')
    print(mean_list)
    return mean_list
if __name__ == '__main__':
    output = np.array(EM(donut, mean_list, cov_list))
    #그림으로 출력
    input_data = plt.scatter(donut[:,0], donut[:,1], s = 30, c = 'blue')
    output_data = plt.scatter(output[:,0], output[:,1], c = 'red')
    plt.legend((input_data, output_data),
               ('Input Data', 'Output Centroid'),
               scatterpoints=1,
               loc='upper right',
               fontsize=8)
    plt.title('2013920049 Scatter Plot')
    #plt.show()
    plt.savefig('output.png')