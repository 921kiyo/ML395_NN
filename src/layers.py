import numpy as np

def linear_forward(X, W, b):
    """
    Computes the forward pass for a linear (fully-connected) layer.

    Args:
    - X: A numpy array of shape (N, d_1, ..., d_K) incoming data
    - W: Anumpy array of shape (K, M) of weights
    - b: A numpy array of shape (M, ) of biases

    Returns:
    - out: linear transformation to the incoming data

    """

    """
    TODO: Implement the linear forward pass. Store your result in `out`.
    Tip: Think how X needs to be reshaped.
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################

    X_reshaped = np.reshape(X, newshape=(X.shape[0],-1))
    a = np.matmul(X_reshaped,W)
    out = np.add(a,b)

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return out


def linear_backward(dout, X, W, b):
    """
    Computes the forward pass for a linear (fully-connected) layer.

    Args:
    - dout: Upstream derivative, of shape (N, M)
    - X: A numpy array of shape (N, d_1, ..., d_K) incoming data
    - W: Anumpy array of shape (K, M) of weights
    - b: A numpy array of shape (M, ) of biases

    Returns (as tuple):
    - dX: A numpy array of shape (N, d1, ..., d_k), gradient with respect to x
    - dW: A numpy array of shape (D, M), gradient with respect to W
    - db: A nump array of shape (M,), gradient with respect to b
    """

    dX, dW, db = None, None, None
    """
    TODO: Implement the linear backward pass. Store your results of the
    gradients in `dX`, `dW`, `db`.
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################

    db = np.sum(dout, axis=0)
    dX = dout.dot(W.T).reshape(X.shape)
    dW = X.reshape(X.shape[0], W.shape[0]).T.dot(dout)

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return dX, dW, db


def relu_forward(X):
    """
    Computes the forward pass for rectified linear unit (ReLU) layer.
    Args:
    - X: Input, an numpy array of any shape
    Returns:
    - out: An numpy array, same shape as X
    """
    out = None
    """
    TODO: Implement the ReLU forward pass. Store your result in out.
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
    out = X.copy()  # Must use copy in numpy to avoid pass by reference.
    # ReLU activation function
    relu = lambda z: np.maximum(0, z)
    out = relu(out)
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return out


def relu_backward(dout, X):
    """
    Computes the backward pass for rectified linear unit (ReLU) layer.
    Args:
    - dout: Upstream derivative, an numpy array of any shape
    - X: Input, an numpy array with the same shape as dout
    Returns:
    - dX: A numpy array, derivative with respect to X
    """
    dX = None
    """
    TODO: Implement the ReLU backward pass. Store your result in out.
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################

    # Backprop: ReLU
    dX = dout
    dX[X <= 0] = 0
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return dX


def dropout_forward(X, p=0.5, train=True, seed=42):
    """
    Compute f
    Args:
    - X: Input data, a numpy array of any shape.
    - p: Dropout parameter. We drop each neuron output with probability p.
    Default is p=0.5.
    - train: Mode of dropout. If train is True, then perform dropout;
      Otherwise train is False (= test mode). Default is train=True.

    Returns (as a tuple):
    - out: Output of dropout applied to X, same shape as X.
    - mask: In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    out = None
    mask = None
    if seed:
        np.random.seed(seed)
    """
    TODO: Implement the inverted dropout forward pass. Make sure to consider
    both train and test case. Pay attention scaling the activation function.
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################

    out = X
    if train:
        q = 1-p
        mask = np.random.binomial(n = 1, p=q, size = X.shape)
        out = X*mask
        out = out/q

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return out, mask


def dropout_backward(dout, mask, p=0.5, train=True):
    """
    Compute the backward pass for dropout
    Args:
    - dout: Upstream derivative, a numpy array of any shape
    - mask: In training mode, mask is the dropout mask that was used to
      multiply the input; in test mode, mask is None.
    - p: Dropout parameter. We drop each neuron output with probability p.
    Default is p=0.5.
    - train: Mode of dropout. If train is True, then perform dropout;
      Otherwise train is False (= test mode). Default is train=True.

    Returns:
    - dX: A numpy array, derivative with respect to X
    """
    dX = None
    """
    TODO: Implement the inverted backward pass for dropout. Make sure to
    consider both train and test case.
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
    dX = np.array(dout, copy=True)
    if train:
        dX = dX * mask
        dX = dX/(1-p)
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return dX
