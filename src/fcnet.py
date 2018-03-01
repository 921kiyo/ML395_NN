import numpy as np
import copy

from src.classifiers import softmax
from src.layers import (linear_forward, linear_backward, relu_forward,
                        relu_backward, dropout_forward, dropout_backward)


def random_init(n_in, n_out, weight_scale=5e-2, dtype=np.float32):
    """
    Weights should be initialized from a normal distribution with standard
    deviation equal to weight_scale and biases should be initialized to zero.

    Args:
    - n_in: The number of input nodes into each output.
    - n_out: The number of output nodes for each input.
    """
    W = None
    b = None
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
    b = np.zeros(n_out)
    W = np.random.normal(size=(n_in, n_out), scale=weight_scale, loc=0)

    W = W.astype(dtype=dtype)

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return W, b



class FullyConnectedNet(object):
    """
    Implements a fully-connected neural networks with arbitrary size of
    hidden layers. For a network with N hidden layers, the architecture would
    be as follows:
    [linear - relu - (dropout)] x (N - 1) - linear - softmax
    The learnable params are stored in the self.params dictionary and are
    learned with the Solver.
    """
    def __init__(self, hidden_dims, input_dim=32*32*3, num_classes=10,
                 dropout=0, reg=0.0, weight_scale=1e-2, dtype=np.float32,
                 seed=None):
        """
        Initialise the fully-connected neural networks.
        Args:
        - hidden_dims: A list of the size of each hidden layer
        - input_dim: A list giving the size of the input
        - num_classes: Number of classes to classify.
        - dropout: A scalar between 0. and 1. determining the dropout factor.
        If dropout = 0., then dropout is not applied.
        - reg: Regularisation factor.

        """
        self.dtype = dtype
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.use_dropout = True if dropout > 0.0 else False
        if seed:
            np.random.seed(seed)
        self.params = dict()
        """
        TODO: Initialise the weights and bias for all layers and store all in
        self.params. Store the weights and bias of the first layer in keys
        W1 and b1, the weights and bias of the second layer in W2 and b2, etc.
        Weights and bias are to be initialised according to the Xavier
        initialisation (see manual).
        """
        #######################################################################
        #                           BEGIN OF YOUR CODE                        #
        #######################################################################
        # Set first hidden layer from the input dimensions and the first hidden layer dimensions
        self.params["W1"], self.params["b1"] = random_init(input_dim, n_out= hidden_dims[0], weight_scale = weight_scale)

        # Set remaining hidden layers using input is previous hidden layers output and output is size
        for item in range(1, len(hidden_dims)):
            W_keyword = "W" + str(item+1)
            b_keyword = "b" + str(item+1)
            self.params[W_keyword], self.params[b_keyword] = random_init(hidden_dims[item-1], hidden_dims[item], weight_scale = weight_scale)

        # Set output layer using the final hidden layer and the number of classes
        W_keyword = "W" + str(self.num_layers)
        b_keyword = "b" + str(self.num_layers)
        self.params[W_keyword], self.params[b_keyword] = random_init(hidden_dims[-1], num_classes)
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        # When using dropout we need to pass a dropout_param dictionary to
        # each dropout layer so that the layer knows the dropout probability
        # and the mode (train / test). You can pass the same dropout_param to
        # each dropout layer.
        self.dropout_params = dict()
        if self.use_dropout:
            self.dropout_params = {"train": True, "p": dropout}
            if seed is not None:
                self.dropout_params["seed"] = 0
            # else:
            #     self.dropout_params["seed"] = None
        # Cast all parameters to the correct datatype

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
        Args:
        - X: Input data, numpy array of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
        Returns:
        If y is None, then run a test-time forward pass of the model and
        return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
        If y is not None, then run a training-time forward and backward pass
        and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping
        parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        X = X.astype(self.dtype)
        linear_cache = dict()
        relu_cache = dict()
        dropout_cache = dict()
        """
        TODO: Implement the forward pass for the fully-connected neural
        network, compute the scores and store them in the scores variable.
        """
        #######################################################################
        #                           BEGIN OF YOUR CODE                        #
        #######################################################################
        activations = []
        z_values = []
        masks = dict()
        output_cache = dict()

        for i in range(1,self.num_layers):
            W_keyword = "W" + str(i)
            b_keyword = "b" + str(i)

            if i == 1:
                linear_cache[i] =  linear_forward(X, self.params[W_keyword], self.params[b_keyword])
            else:
                linear_cache[i] =  linear_forward(output_cache[i-1], self.params[W_keyword], self.params[b_keyword])
            relu_cache[i] = relu_forward(linear_cache[i])

            # Put in a dropout on the last layer - Test
            if self.use_dropout:
                s = None
                if 'seed' in self.dropout_params.keys():
                    s = self.dropout_params['seed']
                # if self.dropout_params["seed"] is not None:
                dropout_cache["d" + str(i)], masks["m" + str(i)] = dropout_forward(X = relu_cache[i], p= self.dropout_params["p"], train = self.dropout_params["train"], seed=0)
                # else:
                    # dropout_cache["d" + str(i)], masks["m" + str(i)] = dropout_forward(X = relu_cache[i], p= self.dropout_params["p"], train = self.dropout_params["train"])

                output_cache[i] = dropout_cache["d" + str(i)]
            # #Linear final layer
            # scores = linear_forward(activation, self.params["W" + str(self.num_layers)], self.params["b" + str(self.num_layers)]) # For dropout in last layer

        if self.use_dropout:
            dropout_cache["d" + str(self.num_layers)], masks["m" + str(self.num_layers)] = dropout_forward(output_cache[i-1], p= self.dropout_params["p"], train = self.dropout_params["train"])

            #Linear final layer
            scores = linear_forward(output_cache[i-1], self.params["W" + str(self.num_layers)], self.params["b" + str(self.num_layers)]) # For dropout in last layer
        else:
            scores = linear_forward(output_cache[i-1], self.params["W"+str(self.num_layers)], self.params["b"+str(self.num_layers)])
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores
        loss, grads = 0, dict()

        """
        TODO: Implement the backward pass for the fully-connected net. Store
        the loss in the loss variable and all gradients in the grads
        dictionary. Compute the loss with softmax. grads[k] has the gradients
        for self.params[k]. Add L2 regularisation to the loss function.
        NOTE: To ensure that your implementation matches ours and you pass the
        automated tests, make sure that your L2 regularization includes a
        factor of 0.5 to simplify the expression for the gradient.
        """
        #######################################################################
        #                           BEGIN OF YOUR CODE                        #
        #######################################################################
        loss, dx = softmax(scores,y)

        # Apply L2 Regularisation
        for i in range(1, self.num_layers+1):
            loss += (0.5*self.reg) * np.sum(np.square(self.params["W{}".format(i)]))

        if self.use_dropout:
            dx, dW, db = linear_backward(dx, dropout_cache["d" + str(self.num_layers)],self.params["W" + str(self.num_layers)], self.params["b" + str(self.num_layers)]) # Version for with dropout in final layer
        else:
            dx, dW, db = linear_backward(dx, relu_cache[self.num_layers-1], self.params["W" + str(self.num_layers)],self.params["b" + str(self.num_layers)])
        grads["W" + str(self.num_layers)] = dW
        grads["W" + str(self.num_layers)] += self.reg * self.params["W" + str(self.num_layers)]
        grads["b" + str(self.num_layers)] = db

        for j in range(self.num_layers-1, 0, -1):
            # Relu pass with incoming Z value
            dx = relu_backward(dx, linear_cache[j])
            # Linear pass with activation value of incoming layer
            # Backwards pass: last layer should be linear:
            if self.use_dropout and j > 1:
                dx, dW, db = linear_backward(dx, dropout_cache["d{}".format(j)], self.params["W{}".format(j)], self.params["b{}".format(j)])
                # Then reverse the dropout
                dx = dropout_backward(dout = dx, mask=masks["m{}".format(j)], p = self.dropout_params["p"], train = self.dropout_params["train"])
            else:
                if(j == 1):
                    dx, dW, db = linear_backward(dx, X, self.params["W{}".format(j)], self.params["b{}".format(j)])
                else:
                    dx, dW, db = linear_backward(dx, relu_cache[j], self.params["W{}".format(j)], self.params["b{}".format(j)])
            grads["W{}".format(j)] = dW
            # Regularisation
            grads["W{}".format(j)] += self.reg * self.params["W{}".format(j)]
            grads["b{}".format(j)] = db
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        return loss, grads
