# -*- coding: utf-8 -*-
# 필요한 패키지를 로드합니다.
# import os
# os.environ['KERAS_BACKEND'] = 'theano'
import numpy as np
import numpy.random as nr
from keras.models import Sequential, save_model
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.utils import np_utils


#아래의 Conv2D 함수의 인자인 input_shape 는 (samples, rows, cols, chennels) 의 4D Tensor 형태여야 하므로
#Train data 를 모델에 넣기 위해서 (60000, 28, 28, 1) 의 4D Tensor 로 Reshape 해 줍니다.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
#Train data 를 dloat 형태로 바꿔 줍니다.
x_train = x_train.astype('float32')
#255로 나눠 주어 pixel의 grayscale 값이 0에서 1 사이의 값을 가지게 해 줍니다.
x_train /= 255
#np.util에 있는 to_categorical은 주어진 레이블을 one-hot encoding 해 줍니다.
y_train = np_utils.to_categorical(y_train, 10)

if __name__ == '__main__':
    # 모델 구성(Input -> CONV(ReLU) -> CONV(ReLU) -> Pool(2) -> FC(relu) -> FC(softmax))
    model = Sequential()
    #필터가 10개이고, 필터 사이즈가 (3,3), Stride가 1인 Conv Layer를 추가합니다.
    model.add(Convolution2D(input_shape = (28, 28, 1), filters = 10, kernel_size = (3,3), strides = 1))
    #Activation Function 으로 Relu를 사용하였습니다.
    model.add(Activation('relu'))
    #필터가 20개이고, 필터 사이즈가 (3,3), Stride가 1인 Conv Layer를 추가합니다.
    model.add(Convolution2D(filters = 20, kernel_size = (3,3), strides = 1))
    model.add(Activation('relu'))
    #이미지의 차원을 줄이기 위해서 Pooling Layer를 넣었습니다.
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #0.25의 확률의 Dropout 시행
    model.add(Dropout(0.25))
    #FC Layer에 넣기 위해서 2차원 배열로 되어 있는 이미지 인풋을 Flattern 해 줍니다.
    model.add(Flatten())
    #Output Node 가 128개인 FC Layer
    model.add(Dense(128))
    model.add(Activation('relu'))
    #Output Node 가 10개인 FC Layer
    model.add(Dense(10))
    #Output 벡터의 합을 1로 만들어주기 위해서 softmax 함수를 사용했습니다.
    model.add(Activation('softmax'))
    # #Keras Documentation 을 참고하여 Adam Optimizer 를 이용하였습니다.
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics = ['accuracy'])
    #CNN 모델을 학습시킵니다. 학습 결과는 History object에 저장됩니다.
    history = model.fit(x_train,y_train , epochs=30, batch_size=100, verbose = 1)
    #best parameter를 저장합니다.
    save_model(model, 'best_param.h5')
    #Train Log 를 텍스트 파일로 저장합니다.
    filename = 'train_log.txt'
    f = open(filename, 'w')
    f.write('=============Train Result==========\n')
    
    log_list = history.history['acc']
    for i in range(len(log_list)):
        f.write('Epoch ' + str(i+1) + ' - Error Rate : ' + str(log_list[i]) + '\n')
    f.close()