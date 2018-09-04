# -*- coding: utf-8 -*-
#필요한 패키지를 로드합니다.
import os
#os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
#Train할 때 저장한 best parameter를 불러옵니다.
model = load_model('best_param.h5')

#저장한 parameter를 모델의 파라미터로 설정합니다.
params = model.get_weights()
model.set_weights(params)
#mnist data를 불러옵니다.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#아래의 Conv2D 함수의 인자인 input_shape 는 (samples, rows, cols, chennels) 의 4D Tensor 형태여야 하므로
#test data 를 모델에 넣기 위해서 (60000, 28, 28, 1) 의 4D Tensor 로 Reshape 해 줍니다.
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
#test data 를 dloat 형태로 바꿔 줍니다.
x_test = x_test.astype('float32')
#255로 나눠 주어 pixel의 grayscale 값이 0에서 1 사이의 값을 가지게 해 줍니다.
x_test /= 255

#test data를 검증해주는 함수입니다.
test_result = model.predict_classes(x_test, verbose=1)

#케라스 Documantation 에 있는 Tutorial대로 4번째 레이어의 출력을 저장하는 함수를 만들었습니다.
#4번째 Layer의 출력은 두 번째 Conv Layer을 지나고 Relu를 지난 Output 입니다.
get_layer_output = backend.function([model.layers[0].input], [model.layers[3].output])
layer_output = get_layer_output([x_test])

#0과 1에 해당되는 Feature Map을 출력할 예정입니다.
#Test Data Set의 55번째, 69번째, 71번째 데이터의 Label 의 0이고
#57, 74, 89번째 Label이 1이므로 이에 해당하는 데이터만 리스트에 저장하겠습니다.

#파일이 저장될 디렉토리를 만듭니다.
if os.path.exists('./Feature_Map') == False:
    os.mkdir('./Feature_Map')
sample_list = [55, 69, 71, 57, 74, 89]
#Feature map 파일을 저장합니다.
for i in sample_list:
    output_shape = np.array(layer_output[0][i]).shape
    for filter_index in range(output_shape[2]):
        fig, ax = plt.subplots()
        ax.imshow(np.array(layer_output[0][i,:,:,filter_index]), cmap = 'gray')
        plt.savefig('./Feature_Map/[' + str(y_test[i]) +  '_' + str(i) + '_' + str(filter_index + 1) + '].png')
        # flush
        plt.cla()
        plt.clf()
        plt.close() 


#test의 결과를 텍스트 파일로 저장하고 콘솔창에 출력합니다.
if __name__ == '__main__':
    filename = 'test_output.txt'
    f = open(filename, 'w')
    f.write('=============Test Result==========\n')

    true = 0
    false = 0

    #Test Output 텍스트 파일로 저장
    for i in range(len(test_result)):
        if y_test[i] == test_result[i]:
            f.write('Test Sample ' + str(i) + ' - Label : ' + str(y_test[i]) + ', Result : ' + str(test_result[i]) + ' -> True\n')
            true += 1
        else:
            f.write('Test Sample ' + str(i) + ' - Label : ' + str(y_test[i]) + ', Result : ' + str(test_result[i]) + ' -> False\n')
            false += 1
    f.write('Test Result : True - ' + str(true) + ', False - ' + str(false) + ', Error Rate - ' + str(float(false) / (false + true)) + '\n')
    print('Test Result : True - ' + str(true) + ', False - ' + str(false) + ', Error Rate - ' + str(float(false) / (false + true)) + '\n')
    f.close()