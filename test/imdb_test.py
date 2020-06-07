from keras.datasets import imdb
import numpy as np

from keras import layers
from keras import models
from keras import optimizers
from keras import losses
from keras import metrics

import matplotlib.pyplot as plt


def imdb_test():
    (train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)
    print("train_data:",train_data[0])
    print("train_labels:",train_labels[0])
    # 将某条评论解码为英文单词
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value,key) for (key,value) in  word_index.items()]) # 将整数索引映射为单词
    decoded_review = ' '.join([reverse_word_index.get(i -3,'?') for i in train_data[0]]) # 解码评论 
    print("reverse_word_index:",reverse_word_index)
    print("decoded_review:",decoded_review)

def vectorize_sequences(sequences,dimesion=10000):
    results = np.zeros((len(sequences),dimesion),dtype='float32') # 创建一个shape为（len(sequece),dimesion）的0矩阵
    for i ,sequences in enumerate(sequences):
        results[i,sequences] = 1.
    return results

def train_model():
    (train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)
    x_train = vectorize_sequences(train_data) # 训练数据向量化
    x_test = vectorize_sequences(test_data) # 测试数据向量化

    y_train = np.asanyarray(train_labels).astype('float32')
    y_test = np.asanyarray(test_labels).astype('float32')

    # 构建网络
    model = models.Sequential()
    model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
    model.add(layers.Dense(16,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))

    
    
    # 留出验证集
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]

    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc']
                 )
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,
                        batch_size=512,
                        validation_data=(x_val,y_val))
    print(history)

    result = model.predict(x_test)

    return history,result

def show_loss(history):

    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    epochs = range(1,len(loss_values)+1)

    plt.plot(epochs,loss_values,'bo',label='Trainning loss')
    plt.plot(epochs,val_loss_values,'b',label='Validation loss')
    plt.title("Trainning and validation loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def show_accuracy(history):
    history_dict = history.history
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    val_loss_values = history_dict['val_loss']

    epochs = range(1,len(val_acc)+1)

    plt.plot(epochs,acc,'bo',label='Trainning acc')
    plt.plot(epochs,val_acc,'b',label='Validation acc')
    plt.title("Trainning and validation loss")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

if __name__ == '__main__':
	
    history,result = train_model()
    print(result)
    # show_loss(history)
    # show_accuracy(history)
