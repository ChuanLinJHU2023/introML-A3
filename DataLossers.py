from DataLayers import *


class Loss:
    def __init__(self, output_size):
        self.output_size = output_size
        self.predicted = None
        self.groundtruth = None

    def init_input_size(self, input_size):
        assert self.output_size is None
        self.output_size = input_size

    def value(self, predicted, groundtruth, need_reshape=False, need_encode=False):
        raise NotImplementedError

    def grad(self):
        raise NotImplementedError

    def shape_check(self, predicted, groundtruth):
        assert predicted.shape == groundtruth.shape == (self.output_size, 1)

    def reshape(self,predicted,groundtruth):
        predicted = np.array(predicted).reshape(-1, 1)
        groundtruth = np.array(groundtruth).reshape(-1, 1)
        return predicted,groundtruth


class SSE(Loss):
    def value(self, predicted, groundtruth, need_reshape=False, need_encode=False):
        if need_reshape:
            predicted, groundtruth= self.reshape(predicted,groundtruth)
        self.shape_check(predicted, groundtruth)
        self.predicted=predicted
        self.groundtruth=groundtruth
        value = np.sum(np.square(predicted - groundtruth))
        return value

    def grad(self):
        grad = 2 * (self.predicted - self.groundtruth)
        return grad


class SoftmaxCrossEntropy(Loss):
    def value(self, predicted, groundtruth, need_reshape=False, need_encode=False):
        if need_encode:
            groundtruth = self.one_hot_encode(groundtruth)
        if need_reshape:
            predicted, groundtruth = self.reshape(predicted, groundtruth)
        self.shape_check(predicted, groundtruth)
        self.predicted=predicted
        self.groundtruth=groundtruth
        predicted = self.softmax(predicted)
        return self.cross_entropy(predicted,groundtruth)

    def softmax(self, predicted):
        print("!!!!!!!!!!!")
        print(predicted)
        exp = np.exp(predicted)
        exp_sum = np.sum(exp)
        probabilities = exp / exp_sum
        return probabilities

    def cross_entropy(self, probabilities_pred, probabilities_true):
        assert probabilities_pred.shape == probabilities_true.shape == (self.output_size, 1)
        return -np.sum(np.log2(probabilities_pred + 1e-30) * probabilities_true)

    def one_hot_encode(self, class_k):
        number_of_classes = self.output_size
        shape = (number_of_classes, 1)
        probabilities = np.zeros(shape)
        probabilities[class_k][0] = 1
        return probabilities

    def grad(self):
        grad=self.softmax(self.predicted) - self.groundtruth
        return grad


