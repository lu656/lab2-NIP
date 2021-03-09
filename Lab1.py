
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
import matplotlib.pyplot as plt


random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ALGORITHM = "guesser"
# ALGORITHM = "tf_net"
ALGORITHM = "tf_conv"

DATASET = "mnist_d"
# DATASET = "mnist_f"
# DATASET = "cifar_10"
# DATASET = "cifar_100_f"
# DATASET = "cifar_100_c"

if DATASET == "mnist_d":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "mnist_f":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "cifar_10":
    # pass                                 # TODO: Add this case.
    NUM_CLASSES = 10
    IH = 32
    IW = 32
    IZ = 3
    IS = 1024
elif DATASET == "cifar_100_f":
    # pass                                 # TODO: Add this case.
    NUM_CLASSES = 100
    IH = 32
    IW = 32
    IZ = 3
    IS = 1024
elif DATASET == "cifar_100_c":
    # pass                                 # TODO: Add this case.
    NUM_CLASSES = 20
    IH = 32
    IW = 32
    IZ = 3
    IS = 1024


#=========================<Classifier Functions>================================

def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0] * NUM_CLASSES
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


def buildTFNeuralNet(x, y, eps = 6):
    # pass        #TODO: Implement a standard ANN here.
    model = None
    if DATASET == 'cifar_100_f':
        model = keras.Sequential([
                    keras.layers.Dense(512, activation='sigmoid'),
                    keras.layers.Dense(512, activation='sigmoid'),
                    keras.layers.Dense(100)
                ])
    elif DATASET == 'cifar_100_c':
        model = keras.Sequential([
                    keras.layers.Dense(512, activation='sigmoid'),
                    keras.layers.Dense(512, activation='sigmoid'),
                    keras.layers.Dense(20)
                ])
    else:
        model = keras.Sequential([
                    keras.layers.Dense(512, activation='sigmoid'),
                    keras.layers.Dense(512, activation='sigmoid'),
                    keras.layers.Dense(10)
                ])

    model.compile(optimizer='adam', 
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    model.fit(x, y, epochs=eps)
    return model


def buildTFConvNet(x, y, eps = 10, dropout = True, dropRate = 0.2):
    # pass        #TODO: Implement a CNN here. dropout option is required.
    model = tf.keras.Sequential()
    # print(x, y)
    if 'cifar' in DATASET:
        model.add(tf.keras.layers.Conv2D(32, (3, 3), use_bias=False, input_shape=(IH, IW, IZ)))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('elu'))
        if dropout:
            model.add(tf.keras.layers.Dropout(dropRate))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), use_bias=False))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('elu'))
        if dropout:
            model.add(tf.keras.layers.Dropout(dropRate))
        model.add(tf.keras.layers.Conv2D(256, (3, 3), use_bias=False))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('elu'))
        if dropout:
            model.add(tf.keras.layers.Dropout(dropRate))
        model.add(tf.keras.layers.Flatten())
        if DATASET == 'cifar_100_c':
            model.add(tf.keras.layers.Dense(1000, activation='elu'))
            model.add(tf.keras.layers.Dense(20, activation='softmax'))
        elif DATASET == 'cifar_100_f':
            model.add(tf.keras.layers.Dense(1000, activation='elu'))
            model.add(tf.keras.layers.Dense(100, activation='softmax'))
        else:
            model.add(tf.keras.layers.Dense(100, activation='elu'))
            model.add(tf.keras.layers.Dense(10, activation='softmax'))
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        model.fit(x, y, epochs=eps)
    elif DATASET == 'mnist_d':
        model.add(tf.keras.layers.Conv2D(32, (3, 3), use_bias=False, input_shape=(IH, IW, IZ)))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('elu'))
        if dropout:
            model.add(tf.keras.layers.Dropout(dropRate))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), use_bias=False))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('elu'))
        if dropout:
            model.add(tf.keras.layers.Dropout(dropRate))
        model.add(tf.keras.layers.Conv2D(256, (3, 3), use_bias=False))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('elu'))
        if dropout:
            model.add(tf.keras.layers.Dropout(dropRate))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1000, activation='elu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        model.fit(x, y, epochs=eps)
    else:
        model.add(tf.keras.layers.Conv2D(32, (3, 3), use_bias=False, input_shape=(IH, IW, IZ)))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), use_bias=False))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('elu'))
        if dropout:
            model.add(tf.keras.layers.Dropout(dropRate))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), use_bias=False))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), use_bias=False))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('elu'))
        if dropout:
            model.add(tf.keras.layers.Dropout(dropRate))
        model.add(tf.keras.layers.Conv2D(256, (3, 3), use_bias=False))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('elu'))
        if dropout:
            model.add(tf.keras.layers.Dropout(dropRate))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(500, activation='elu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        model.fit(x, y, epochs=eps + 10)
    return model

#=========================<Pipeline Functions>==================================

def getRawData():
    if DATASET == "mnist_d":
        mnist = tf.keras.datasets.mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "mnist_f":
        mnist = tf.keras.datasets.fashion_mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "cifar_10":
        # pass      # TODO: Add this case.
        cifar = tf.keras.datasets.cifar10
        (xTrain, yTrain), (xTest, yTest) = cifar.load_data()
    elif DATASET == "cifar_100_f":
        # pass      # TODO: Add this case.
        cifar = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar.load_data(label_mode='fine')
    elif DATASET == "cifar_100_c":
        # pass      # TODO: Add this case.
        cifar = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar.load_data(label_mode='coarse')
    else:
        raise ValueError("Dataset not recognized.")
    print("Dataset: %s" % DATASET)
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    if ALGORITHM != "tf_conv":
        xTrainP = xTrain.reshape((xTrain.shape[0], IS))
        xTestP = xTest.reshape((xTest.shape[0], IS))
    else:
        xTrainP = xTrain.reshape((xTrain.shape[0], IH, IW, IZ))
        xTestP = xTest.reshape((xTest.shape[0], IH, IW, IZ))
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrainP, yTrainP), (xTestP, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        return buildTFNeuralNet(xTrain, yTrain)
    elif ALGORITHM == "tf_conv":
        print("Building and training TF_CNN.")
        return buildTFConvNet(xTrain, yTrain)
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    elif ALGORITHM == "tf_conv":
        print("Testing TF_CNN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()
    return accuracy * 100



#=========================<Main>================================================

def main():
    # nn_type = ['tf_net', 'tf_conv']
    # dataset_type = ['mnist_d' , 'mnist_f', 'cifar_10', 'cifar_100_f', 'cifar_100_c']
    # global ALGORITHM
    # global DATASET
    # global IH
    # global IW
    # global IZ
    # global IS
    # global NUM_CLASSES
    # for item in nn_type:
    #     ALGORITHM = item 
    #     accuracies = {}
    #     for t in dataset_type:
    #         DATASET = t
    #         if DATASET == 'mnist_d':
    #             NUM_CLASSES = 10
    #             IH = 28
    #             IW = 28
    #             IZ = 1
    #             IS = 784
    #         elif DATASET == "mnist_f":
    #             NUM_CLASSES = 10
    #             IH = 28
    #             IW = 28
    #             IZ = 1
    #             IS = 784
    #         elif DATASET == "cifar_10":
    #             # pass                                 # TODO: Add this case.
    #             NUM_CLASSES = 10
    #             IH = 32
    #             IW = 32
    #             IZ = 3
    #             IS = 1024 * 3
    #         elif DATASET == "cifar_100_f":
    #             # pass                                 # TODO: Add this case.
    #             NUM_CLASSES = 100
    #             IH = 32
    #             IW = 32
    #             IZ = 3
    #             IS = 1024 * 3
    #         elif DATASET == "cifar_100_c":
    #             # pass                                 # TODO: Add this case.
    #             NUM_CLASSES = 20
    #             IH = 32
    #             IW = 32
    #             IZ = 3
    #             IS = 1024 * 3
    #         raw = getRawData()
    #         data = preprocessData(raw)
    #         model = trainModel(data[0])
    #         preds = runModel(data[1][0], model)
    #         acc = evalResults(data[1], preds)
    #         accuracies[t] = acc
    #     plt.bar(range(len(accuracies)), list(accuracies.values()), align='center')
    #     plt.xticks(range(len(accuracies)), list(accuracies.keys()))
    #     plt.xlabel('Dataset')
    #     plt.ylabel('Accuracy')
    #     plt.title(item)
    #     if item == 'tf_net':
    #         plt.savefig('./ANN_Accuracy_Plot.pdf')
    #     else:
    #         plt.savefig('./CNN_Accuracy_Plot.pdf')
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    acc = evalResults(data[1], preds)



if __name__ == '__main__':
    main()
