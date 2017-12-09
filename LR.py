from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


data = load_svmlight_file("/home/mily/housing_scale")
train_data, validation_data = train_test_split(data[0], test_size=0.33, random_state=42)
train_data = train_data.toarray()
validation_data = validation_data.toarray()

x_train = train_data[:, 1:]
y_train = train_data[:, 0]
x_validation = validation_data[:, 1:]
y_validation = validation_data[:, 0]
x_train = np.concatenate((np.ones((x_train.shape[0], 1), dtype='float'), x_train), axis=1)
x_validation = np.concatenate((np.ones((x_validation.shape[0], 1), dtype='float'), x_validation), axis=1)


def process(x, y, alpha=0.00001):
    loss = []
    w = np.zeros((x.shape[1], 1))
    for i in range(900):
        error = y - x.dot(w)
        loss.append((1/2 * np.dot(error.T, error)[0, 0])/x.shape[0])
        delta_w = x.T.dot(-error)
        w = w - alpha * delta_w
    return loss


if __name__ == '__main__':
    loss_train = process(x_train, y_train, 0.00001)
    loss_validation = process(x_validation, y_validation, 0.00001)
    plot1,  = plt.plot(np.arange(0, len(loss_train)), loss_train, 'r', label='loss_train')
    plot2,  = plt.plot(np.arange(0, len(loss_validation)), loss_validation, 'b', label='loss_validation')
    plt.title('loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend([plot1, plot2], ['loss_train', 'loss_validation'])
    plt.show()
