import my_net as mn
import pandas as pd
import numpy as np

if __name__ == '__main__':
    np.random.seed(137)
    square_train = pd.read_csv('data/regression/square-simple-training.csv', index_col=0)
    steps_train = pd.read_csv('data/regression/steps-large-training.csv', index_col=0)
    square_test = pd.read_csv('data/regression/square-simple-test.csv', index_col=0)
    steps_test = pd.read_csv('data/regression/steps-large-test.csv', index_col=0)

    square_train_X = square_train[['x']].values
    square_train_Y = square_train[['y']].values
    square_test_X = square_test[['x']].values
    square_test_Y = square_test[['y']].values

    print(np.shape(steps_train))

    # net = mn.Net(1)
    # net.add(mn.DenseLayer(15, "sigmoid"))
    # net.add(mn.DenseLayer(15, "sigmoid"))
    # net.add(mn.DenseLayer(1, "identity"))
    # net.kernel_init("xavier")
    #
    # mse, iters = net.backpropagate(square_train_X, square_train_Y, eta=0.001, n_epochs=5000,
    #                                required_mse=3.7, batch_size=25)
    # print(mse)
    # print(iters)

