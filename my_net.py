import numpy as np

sigmoid = lambda a: 1 / (1 + np.exp(-a))
sigmoid_prim = lambda a: sigmoid(a) * (1 - sigmoid(a))
identity = lambda a: a
identity_prim = lambda a: 1


class DenseLayer:
    def __init__(self, n_neurons: int, activation: str, weights=None, biases=None):
        if n_neurons < 1:
            raise ValueError("Layer should have at least 1 neuron")
        self.n_neurons = n_neurons
        self.activation = activation
        self.weights = weights
        self.biases = biases

        if activation == "sigmoid":
            self.f = sigmoid
            self.f_prim = sigmoid_prim
        elif activation == "identity":
            self.f = identity
            self.f_prim = identity_prim
        else:
            raise ValueError("Incorrect activation function specified. Should be \"sigmoid\" or \"identity\"")

        if weights is not None:
            if np.shape(weights)[1] != self.n_neurons:
                raise ValueError

            self.weights_acc_deltas = np.zeros(np.shape(weights))
            self.weights_momentum = np.zeros(np.shape(weights))

        if biases is not None:
            if np.shape(biases)[0] != 1:
                raise ValueError
            if np.shape(biases)[1] != self.n_neurons:
                raise ValueError

            self.biases_acc_deltas = np.zeros(np.shape(biases))
            self.biases_momentum = np.zeros(np.shape(biases))

    def kernel_init_uniform(self, n_inputs: int, bound: float):
        if n_inputs < 1:
            raise ValueError("Layer should have at least 1 input")
        self.weights = np.random.uniform(-bound, bound, (n_inputs, self.n_neurons))
        self.biases = np.random.uniform(-bound, bound, (1, self.n_neurons))

        self.weights_acc_deltas = np.zeros((n_inputs, self.n_neurons))
        self.biases_acc_deltas = np.zeros((1, self.n_neurons))
        self.weights_momentum = np.zeros((n_inputs, self.n_neurons))
        self.biases_momentum = np.zeros((1, self.n_neurons))

    def kernel_init_xavier(self, n_inputs: int):
        bound = np.sqrt(6 / (self.n_neurons + n_inputs))
        self.kernel_init_uniform(n_inputs, bound)

    def kernel_init(self, n_inputs: int, initializer: str):
        if initializer == "uniform":
            self.kernel_init_uniform(n_inputs, 0.5)
        elif initializer == "xavier":
            self.kernel_init_xavier(n_inputs)
        else:
            raise ValueError("Incorrect initializer specified. Should be \"uniform\" or \"xavier\".")

    def is_initialized(self):
        return self.weights is not None and self.biases is not None

    def n_inputs(self):
        if not self.is_initialized():
            raise ValueError("Layer weights or biases are not initialized yet.")

        return np.shape(self.weights)[0]

    def calculate(self, input):
        if not self.is_initialized():
            raise ValueError("Layer weights or biases are not initialized yet.")

        if np.shape(input)[1] != self.n_inputs():
            raise ValueError("Input size does not match layer's expected input size.")
        return self.f(input @ self.weights + self.biases)

    def propagate_error(self, differences, calculated_input):
        if not self.is_initialized():
            raise ValueError("Layer weights or biases are not initialized yet.")

        if np.shape(calculated_input)[1] != self.n_inputs():
            raise ValueError
        if np.shape(differences)[1] != self.n_neurons:
            raise ValueError

        next_errors = self.f_prim(calculated_input @ self.weights + self.biases) * differences  # shape = [1, n_neurons]
        delta_weights = -calculated_input.transpose() @ next_errors  # shape = [n_inputs, n_neurons]
        delta_biases = -next_errors  # shape = [1, n_neurons]
        next_differences = next_errors @ self.weights.transpose()  # shape = [1, n_neurons] * [n_neurons, n_inputs] = [1, n_inputs]
        self.weights_acc_deltas += delta_weights
        self.biases_acc_deltas += delta_biases
        return next_differences

    def apply_learned_changes(self, eta: float = 0.001, normalization_method: str = None, ext_factor: float = 0.9):
        if normalization_method is None:
            self.weights_momentum = self.weights_acc_deltas
            self.biases_momentum = self.biases_acc_deltas
            self.weights += eta * self.weights_momentum
            self.biases += eta * self.biases_momentum
        elif normalization_method is "momentum":
            if ext_factor < 0 or ext_factor >= 1:
                raise ValueError("ext_factor (lambda) should be between 0 and 1.")
            self.weights_momentum = self.weights_acc_deltas + ext_factor * self.weights_momentum
            self.biases_momentum = self.biases_acc_deltas + ext_factor * self.biases_momentum
            self.weights += eta * self.weights_momentum
            self.biases += eta * self.biases_momentum
        elif normalization_method is "RMSProp":
            if ext_factor < 0 or ext_factor >= 1:
                raise ValueError("ext_factor (beta) should be between 0 and 1.")
            self.weights_momentum = (1 - ext_factor) * self.weights_acc_deltas ** 2 + ext_factor * self.weights_momentum
            self.biases_momentum = (1 - ext_factor) * self.biases_acc_deltas ** 2 + ext_factor * self.biases_momentum
            self.weights += eta * self.weights_acc_deltas / np.sqrt(self.weights_momentum)
            self.biases += eta * self.biases_acc_deltas / np.sqrt(self.biases_momentum)
        else:
            raise ValueError("normalization_method should be either None or \"momentum\" or \"RMSProp\".")


class Net:
    def __init__(self, input_size):
        if input_size < 1:
            raise ValueError
        self.input_size = input_size
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def backpropagate(self, input, output, eta: float = 0.001, required_mse: float = None, batch_size: int = None,
                      n_epochs: int = 1, verbose: int = 0,
                      validation_input=None, validation_output=None):
        input = np.asarray(input)
        output = np.asarray(output)
        if validation_input is None:
            validation_input = input
        else:
            validation_input = np.asarray(validation_input)
        if validation_output is None:
            validation_output = output
        else:
            validation_output = np.asarray(validation_output)

        print_mses = verbose == 1 or verbose == 3
        print_absolute_weights = verbose == 2 or verbose == 3

        if np.shape(input)[1] != self.layers[0].n_inputs():
            raise ValueError("Incorrect X shape, got " + str(np.shape(input)[1]),
                             "expected: " + str(self.layers[0].n_inputs()))

        if np.shape(output)[1] != self.layers[-1].n_neurons:
            raise ValueError("Incorrect Y shape")

        if np.shape(input)[0] != np.shape(output)[0]:
            raise ValueError("X and Y shapes not matching. Different n of observations.")

        if n_epochs < 1:
            raise ValueError("There should be at least 1 epoch")

        n_obs = np.shape(input)[0]

        current_mse = required_mse
        mses_list = []
        n_iters = 0
        for epoch in range(n_epochs):
            permutation = np.random.permutation(n_obs)
            for i in permutation:
                n_iters += 1
                x = input[i, :]
                x = x.reshape((1, np.shape(input)[1]))
                y = output[i, :]
                y = y.reshape((1, np.shape(output)[1]))

                self.backpropagate_once(x, y, eta)

                if (batch_size is not None and n_iters % batch_size == 0) or i == permutation[-1]:
                    self.apply_learned_changes(eta=eta)

                    if required_mse is not None or print_mses:
                        current_mse = self.mse(validation_input, validation_output)
                        mses_list.append((n_iters, current_mse))
                    if print_mses:
                        print("MSE after {0} iterations (epoch {1}): {2}".format(str(n_iters), str(epoch),
                                                                                 str(current_mse)))
                    if print_absolute_weights:
                        abs_weights_string = "Absolute weights sum on {0} iteration (epoch {1}): \n".format(str(n_iters), str(epoch + 1))
                        for layer in self.layers:
                            layer_weights_sum = np.sum(np.abs(layer.weights)) + np.sum(np.abs(layer.biases))
                            abs_weights_string += "{:.2f}".format(layer_weights_sum) + ", "
                        print(abs_weights_string)

                    if required_mse is not None and current_mse <= required_mse:
                        return current_mse, n_iters, mses_list

        current_mse = self.mse(validation_input, validation_output)
        return current_mse, n_iters, mses_list

    def mse(self, input, output):
        return np.mean((self.predict(input) - output) ** 2)

    def backpropagate_once(self, input, output, eta: float = 0.001):
        if np.shape(input) != (1, self.layers[0].n_inputs()):
            raise ValueError(
                "Incorrect X shape, got:" + str(np.shape(input)) + ", expected: " + str((1, self.layers[0].n_inputs())))

        if np.shape(output) != (1, self.layers[-1].n_neurons):
            raise ValueError("Incorrect Y shape")

        # feed forward
        inputs = [input]
        for i in range(len(self.layers) - 1):
            layer = self.layers[i]
            input = layer.calculate(input)
            inputs.append(input)
        predicted_output = self.layers[-1].calculate(input)

        # backpropagate
        differences = predicted_output - output
        for layer, calculated_input in zip(reversed(self.layers), reversed(inputs)):
            differences = layer.propagate_error(differences, calculated_input)

    def predict(self, input_values):
        input_values = np.asarray(input_values)
        for layer in self.layers:
            input_values = layer.calculate(input_values)
        return input_values

    def kernel_init(self, initializer: str):
        n_inputs = self.input_size
        for layer in self.layers:
            layer.kernel_init(n_inputs, initializer)
            n_inputs = layer.n_neurons

    def apply_learned_changes(self, eta: float = 0.001):
        for layer in self.layers:
            layer.apply_learned_changes(eta=eta)
