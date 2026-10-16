import cupy as cp
from scipy.special import expit

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        # He Initialization
        self.weights = cp.random.randn(n_inputs, n_neurons) * cp.sqrt(2.0 / n_inputs)
        self.biases = cp.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs):
        self.inputs = inputs
        self.output = cp.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = cp.dot(self.inputs.T, dvalues)
        self.dbiases = cp.sum(dvalues, axis=0, keepdims=True)

        if self.weight_regularizer_l1 > 0:
            dL1 = cp.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        if self.bias_regularizer_l1 > 0:
            dL1 = cp.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        self.dinputs = cp.dot(dvalues, self.weights.T)


class Layer_Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = cp.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = cp.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_SoftMax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = cp.exp(inputs - cp.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / cp.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = cp.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = cp.diagflat(single_output) - cp.dot(single_output, single_output.T)
            self.dinputs[index] = cp.dot(jacobian_matrix, single_dvalues)


class Activation_Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = expit(-inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output


class Optimizer_SGD:
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = cp.zeros_like(layer.weights)
                layer.bias_momentums = cp.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adagrad:
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = cp.zeros_like(layer.weights)
            layer.bias_cache = cp.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        layer.weights += -self.current_learning_rate * layer.dweights / (cp.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (cp.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class Optimizer_RMSprop:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = cp.zeros_like(layer.weights)
            layer.bias_cache = cp.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2

        layer.weights += -self.current_learning_rate * layer.dweights / (cp.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (cp.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = cp.zeros_like(layer.weights)
            layer.weight_cache = cp.zeros_like(layer.weights)
            layer.bias_momentums = cp.zeros_like(layer.biases)
            layer.bias_cache = cp.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (cp.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (cp.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class Loss:
    def regularization_loss(self, layer):
        regularization_loss = 0

        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * cp.sum(cp.abs(layer.weights))

        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * cp.sum(layer.weights**2)

        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * cp.sum(cp.abs(layer.biases))

        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * cp.sum(layer.biases**2)

        return regularization_loss

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = cp.mean(sample_losses)
        return data_loss


class Loss_CatCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = cp.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = cp.sum(y_pred_clipped * y_true, axis=1)

        negative_log_lh = -cp.log(correct_confidences)
        return negative_log_lh

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = cp.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


class Activation_SoftMax_Loss_CatEntropy():
    def __init__(self):
        self.activation = Activation_SoftMax()
        self.loss = Loss_CatCrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = cp.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


class Loss_BinaryCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = cp.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * cp.log(y_pred_clipped) + (1 - y_true) * cp.log(1 - y_pred_clipped))
        sample_losses = cp.mean(sample_losses, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        clipped_dvalues = cp.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        self.dinputs = self.dinputs / samples


class Layer_Conv2D:
    def __init__(self, n_filters, filter_size, input_depth, stride=1, padding=0, weight_scale=0.01):
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.input_depth = input_depth
        self.stride = stride
        self.padding = padding
        
        # He Initialization
        fan_in = input_depth * filter_size * filter_size
        self.weights = cp.random.randn(n_filters, fan_in) * cp.sqrt(2.0 / fan_in)
        self.biases = cp.zeros((n_filters, 1))

    def im2col(self, inputs):
        batch_size, depth, H, W = inputs.shape
        f = self.filter_size
        s = self.stride

        out_H = (H - f + 2*self.padding)//s + 1
        out_W = (W - f + 2*self.padding)//s + 1

        if self.padding > 0:
            padded = cp.pad(inputs, ((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)), mode='constant')
        else:
            padded = inputs

        shape = (batch_size, depth, out_H, out_W, f, f)
        strides = (padded.strides[0], padded.strides[1], padded.strides[2]*s, padded.strides[3]*s, padded.strides[2], padded.strides[3])
        
        cols = cp.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
        cols = cols.transpose(0, 2, 3, 1, 4, 5).reshape(batch_size * out_H * out_W, -1)
        return cols, out_H, out_W

    def col2im(self, dcols, input_shape, out_H, out_W):
        batch_size, depth, H, W = input_shape
        f = self.filter_size
        s = self.stride

        if self.padding > 0:
            H_p, W_p = H + 2*self.padding, W + 2*self.padding
        else:
            H_p, W_p = H, W
            
        dinputs_padded = cp.zeros((batch_size, depth, H_p, W_p), dtype=dcols.dtype)
        dcols_reshaped = dcols.reshape(batch_size, out_H, out_W, depth, f, f)
        dcols_reshaped = dcols_reshaped.transpose(0, 3, 1, 2, 4, 5)

        for i in range(f):
            for j in range(f):
                dinputs_padded[:, :, i:i+out_H*s:s, j:j+out_W*s:s] += dcols_reshaped[:, :, :, :, i, j]

        if self.padding > 0:
            return dinputs_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            return dinputs_padded

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, depth, H, W = inputs.shape
        self.cols, self.out_H, self.out_W = self.im2col(inputs)

        out = self.cols @ self.weights.T + self.biases.T
        out = out.reshape(batch_size, self.out_H, self.out_W, self.n_filters)
        self.output = out.transpose(0, 3, 1, 2)
        return self.output

    def backward(self, dvalues):
        dvalues_reshaped = dvalues.transpose(0,2,3,1).reshape(-1, self.n_filters)

        self.dweights = dvalues_reshaped.T @ self.cols
        self.dweights = self.dweights.reshape(self.weights.shape)

        self.dbiases = cp.sum(dvalues_reshaped, axis=0).reshape(self.n_filters,1)

        dcols = dvalues_reshaped @ self.weights
        self.dinputs = self.col2im(dcols, self.inputs.shape, self.out_H, self.out_W)
        return self.dinputs


class Layer_MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def im2col(self, inputs):
        batch_size, depth, H, W = inputs.shape
        p = self.pool_size
        s = self.stride

        out_H = (H - p)//s + 1
        out_W = (W - p)//s + 1

        shape = (batch_size, depth, out_H, out_W, p, p)
        strides = (inputs.strides[0], inputs.strides[1], inputs.strides[2]*s, inputs.strides[3]*s, inputs.strides[2], inputs.strides[3])
        
        cols = cp.lib.stride_tricks.as_strided(inputs, shape=shape, strides=strides)
        cols = cols.reshape(batch_size * depth * out_H * out_W, p * p)
        return cols, out_H, out_W

    def col2im(self, dcols, input_shape, out_H, out_W):
        batch_size, depth, H, W = input_shape
        p = self.pool_size
        s = self.stride

        dinputs = cp.zeros((batch_size, depth, H, W), dtype=dcols.dtype)
        dcols_reshaped = dcols.reshape(batch_size, depth, out_H, out_W, p, p)

        for i in range(p):
            for j in range(p):
                dinputs[:, :, i:i+out_H*s:s, j:j+out_W*s:s] += dcols_reshaped[:, :, :, :, i, j]
        return dinputs

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, depth, H, W = inputs.shape

        self.cols, self.out_H, self.out_W = self.im2col(inputs)

        self.max_idx = cp.argmax(self.cols, axis=1)
        out = self.cols[cp.arange(self.cols.shape[0]), self.max_idx]

        self.output = out.reshape(batch_size, depth, self.out_H, self.out_W)
        return self.output

    def backward(self, dvalues):
        #batch_size, depth, H, W = self.inputs.shape
        dcols = cp.zeros_like(self.cols)

        dcols[cp.arange(self.cols.shape[0]), self.max_idx] = dvalues.flatten()

        self.dinputs = self.col2im(dcols, self.inputs.shape, self.out_H, self.out_W)
        return self.dinputs


class Layer_BatchNorm2D:
    def __init__(self, channels, eps=1e-5, momentum=0.9):
        self.channels = channels
        self.eps = eps
        self.momentum = momentum
        self.gamma = cp.ones((1, channels, 1, 1))
        self.beta = cp.zeros((1, channels, 1, 1))
        self.running_mean = cp.zeros((1, channels, 1, 1))
        self.running_var = cp.ones((1, channels, 1, 1))

    def forward(self, inputs, training=True):
        self.inputs = inputs
        if training:
            mean = cp.mean(inputs, axis=(0, 2, 3), keepdims=True)
            var = cp.var(inputs, axis=(0, 2, 3), keepdims=True)
            self.norm = (inputs - mean) / cp.sqrt(var + self.eps)
            out = self.gamma * self.norm + self.beta

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            norm = (inputs - self.running_mean) / cp.sqrt(self.running_var + self.eps)
            out = self.gamma * norm + self.beta

        self.output = out
        return self.output

    def backward(self, dvalues):
        batch_size = self.inputs.shape[0]
        mean = cp.mean(self.inputs, axis=(0, 2, 3), keepdims=True)
        var = cp.var(self.inputs, axis=(0, 2, 3), keepdims=True)

        self.dgamma = cp.sum(dvalues * self.norm, axis=(0, 2, 3), keepdims=True)
        self.dbeta = cp.sum(dvalues, axis=(0, 2, 3), keepdims=True)

        dnorm = dvalues * self.gamma
        dvar = cp.sum(dnorm * (self.inputs - mean) * -0.5 * (var + self.eps) ** (-1.5),
                      axis=(0, 2, 3), keepdims=True)
        dmean = cp.sum(dnorm * -1 / cp.sqrt(var + self.eps),
                       axis=(0, 2, 3), keepdims=True) + dvar * cp.mean(-2 * (self.inputs - mean),
                                                                       axis=(0, 2, 3), keepdims=True)
        self.dinputs = (dnorm / cp.sqrt(var + self.eps)) + (dvar * 2 * (self.inputs - mean) / batch_size) + (dmean / batch_size)
        return self.dinputs
