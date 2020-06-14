from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt


def vectorize_sequences(sequences,dimension=10000):
    ""
    # 创建一个shape为（len(sequece),dimesion）的0矩阵
    results = np.zeros((len(sequences),dimension),dtype='float32') 
    for i ,sequences in enumerate(sequences):
        results[i,sequences] = 1.
    return results

def to_one_hot(labels,dimesion=46):
    results = np.zeros((len(labels),dimesion),dtype='float32') 
    for i ,sequences in enumerate(labels):
        results[i,sequences] = 1.
    return results

def train_model():
    # 获取数据集
    (train_data,train_lable),(test_data,test_labels)= reuters.load_data(num_words=10000)

    # 编码数据
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    one_hot_train_labels = to_categorical(train_lable)
    one_hot_test_labels = to_categorical(test_labels)

    # 定义model
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
    model.add(layers.Dense(128,activation='relu'))
    model.add(layers.Dense(128,activation='relu')) # z再添加一个隐藏层
    model.add(layers.Dense(46,activation='softmax'))

    # 编译model
    model.compile(optimizer='rmsprop',
                  loss = 'categorical_crossentropy',
                  metrics=['accuracy']
                 )

    # 留出验证集
    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]

    y_val = one_hot_train_labels[:1000]
    partial_y_train = one_hot_train_labels[1000:]

    # train model 
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,
                        batch_size=512,
                        validation_data=(x_val,y_val))
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

    
    # for item in history_dict.items():
    #     print(item)
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    # val_loss_values = history_dict['val_loss']

    epochs = range(1,len(val_acc)+1)

    plt.plot(epochs,acc,'bo',label='Trainning acc')
    plt.plot(epochs,val_acc,'b',label='Validation acc')
    plt.title("Trainning and validation accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    history,result = train_model()
    print(result)
    show_accuracy(history)
    # show_loss(history)