# DataLossers.py implements the MSE loss, the softmax cross-entropy loss, and other losses.
from DataLayers import *


class Loss:
    """
    Base class for defining loss functions.

    Args:
        output_size (int): Size of the output.

    Attributes:
        output_size (int): Size of the output.
        predicted (np.ndarray): Predicted values.
        groundtruth (np.ndarray): Ground truth values.
    """

    def __init__(self, output_size):
        """
        Initializes the Loss object. output_size is the size of the output layer which feeds the loss function

        Args:
            output_size (int): Size of the output.
        """
        self.output_size = output_size
        self.predicted = None
        self.groundtruth = None


    def value(self, predicted, groundtruth, need_reshape=False, need_encode=False):
        """
        Computes the loss value.
        """
        raise NotImplementedError

    def grad(self):
        """
        Computes the gradient of the loss.
        """
        raise NotImplementedError

    def shape_check(self, predicted, groundtruth):
        """
        Checks if the shapes of predicted and groundtruth match.
        """
        assert predicted.shape == groundtruth.shape == (self.output_size, 1)

    def reshape(self,predicted,groundtruth):
        """
        Reshapes predicted and groundtruth arrays if needed.
        """
        predicted = np.array(predicted).reshape(-1, 1)
        groundtruth = np.array(groundtruth).reshape(-1, 1)
        return predicted,groundtruth


class SSE(Loss):
    """
    Sum of Squared Errors loss function.
    """

    def value(self, predicted, groundtruth, need_reshape=False, need_encode=False):
        """
         Computes the Sum of Squared Errors loss value.

         Args:
             predicted (np.array): Predicted values.
             groundtruth (np.array): Ground truth values.
             need_reshape (bool): Whether to reshape the input arrays. Defaults to False.
             need_encode (bool): Whether to encode the input arrays. Defaults to False.

         Returns:
             float: Sum of Squared Errors loss value.
         """
        if need_reshape:
            predicted, groundtruth= self.reshape(predicted,groundtruth)
        self.shape_check(predicted, groundtruth)
        self.predicted=predicted
        self.groundtruth=groundtruth
        value = np.sum(np.square(predicted - groundtruth))
        return value

    def grad(self):
        """
        Computes the gradient of the Sum of Squared Errors loss.

        Returns:
            np.array: Gradient of the loss.
        """
        grad = 2 * (self.predicted - self.groundtruth)
        return grad


class SoftmaxCrossEntropy(Loss):
    """
    Softmax Cross Entropy loss function.
    """

    def value(self, predicted, groundtruth, need_reshape=False, need_encode=False):
        """
         Computes the Softmax Cross Entropy loss value.
        """

        if need_encode:
            groundtruth = self.one_hot_encode(groundtruth)
        if need_reshape:
            predicted, groundtruth = self.reshape(predicted, groundtruth)
        self.shape_check(predicted, groundtruth)
        self.predicted=predicted
        self.groundtruth=groundtruth
        predicted = self.softmax(predicted)
        loss_value = self.cross_entropy(predicted,groundtruth)
        return loss_value

    def softmax(self, predicted):
        """
        Softmax the input value
        """
        exp = np.exp(predicted)
        exp_sum = np.sum(exp)
        probabilities = exp / exp_sum
        return probabilities

    def cross_entropy(self, probabilities_pred, probabilities_true):
        """
        Calculates the cross entropy.
        """
        assert probabilities_pred.shape == probabilities_true.shape == (self.output_size, 1)
        return -np.sum(np.log2(probabilities_pred + 1e-30) * probabilities_true)

    def one_hot_encode(self, class_k):
        """
        Encode a number into a one-hot vector
        """
        class_k = int(class_k)
        number_of_classes = self.output_size
        shape = (number_of_classes, 1)
        probabilities = np.zeros(shape)
        probabilities[class_k][0] = 1
        return probabilities

    def grad(self):
        """
        Compute the gradient of Softmax Cross Entropy loss
        """
        grad=self.softmax(self.predicted) - self.groundtruth
        return grad


