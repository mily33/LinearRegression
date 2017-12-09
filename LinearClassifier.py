from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


data = load_svmlight_file("/home/mily/australian_scale")
x_train, x_validation, y_train, y_validation = train_test_split(data[0].toarray(), data[1], test_size=0.33)
x_train = x_train
y_train = y_train.reshape(x_train.shape[0], 1)
x_validation = x_validation
y_validation = y_validation.reshape(x_validation.shape[0], 1)
x_train = np.concatenate((np.ones((x_train.shape[0], 1), dtype = 'float'), x_train), axis = 1)
x_validation = np.concatenate((np.ones((x_validation.shape[0], 1), dtype = 'float'), x_validation), axis = 1)


def process(x_train, y_train, x_validation, y_validation, alpha, C, delta):
    loss_train = []
    loss_validation = []
    accuracy_train = []
    accuracy_validation = []
    w = np.random.randn(x_train.shape[1], 1)
    # w = np.zeros((x_train.shape[1], 1), dtype='float')
    for i in range(500):
        epsilon_train = np.max(np.concatenate((np.zeros((y_train.shape[0], 1), dtype='float'),
                                               1 - y_train * x_train.dot(w)), axis=1), axis=1)

        epsilon_validation = np.max(np.concatenate((np.zeros((y_validation.shape[0], 1), dtype='float'),
                                                    1 - y_validation * x_validation.dot(w)), axis=1), axis=1)

        loss_train.append(((1 / 2 * np.dot(w.T, w)[0, 0]) + C * np.sum(epsilon_train)) / x_train.shape[0])
        loss_validation.append(
            ((1 / 2 * np.dot(w.T, w)[0, 0]) + C * np.sum(epsilon_validation)) / x_validation.shape[0])

        predict_train = 2 * (x_train.dot(w) > delta) - 1
        accuracy_train.append(np.sum(predict_train == y_train) / y_train.shape[0])

        predict_validation = 2 * (x_validation.dot(w) > delta) - 1
        accuracy_validation.append(np.sum(predict_validation == y_validation) / y_validation.shape[0])

        delta_w = w - C * ((y_train * x_train).T.dot(epsilon_train > 0)).reshape(x_train.shape[1], 1)
        w = w - alpha * delta_w / x_train.shape[0]
    return loss_train, loss_validation, accuracy_train, accuracy_validation


if __name__ == '__main__':
    loss_train, loss_validation, accuracy_train, accuracy_validation = process(x_train, y_train, x_validation, y_validation, alpha=0.08, C=1, delta=0)
    plt.figure(1)
    plot1,  = plt.plot(np.arange(0,len(loss_train)), loss_train, 'r', label='loss_train')
    plot2,  = plt.plot(np.arange(0,len(loss_validation)), loss_validation, 'b', label='loss_validation')
    plt.title('loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend([plot1, plot2], ['loss_train', 'loss_validation'])

    plt.figure(2)
    plot1, = plt.plot(np.arange(0, len(accuracy_train)), accuracy_train, 'r', label='accuracy_train')
    plot2, = plt.plot(np.arange(0, len(accuracy_validation)), accuracy_validation, 'b', label='accuracy_validation')
    plt.title('accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('accuracy')
    plt.legend([plot1, plot2], ['accuracy_train', 'accuracy_validation'])
    plt.show()