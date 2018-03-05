import numpy as np


def softmax(logits, y):
    """
    Computes the loss and gradient for softmax classification.

    Args:
    - logits: A numpy array of shape (N, C)
    - y: A numpy array of shape (N,). y represents the labels corresponding to
    logits, where y[i] is the label of logits[i], and the value of y have a
    range of 0 <= y[i] < C

    Returns (as a tuple):
    - loss: Loss scalar
    - dlogits: Loss gradient with respect to logits
    """
    loss, dlogits = None, None
    """
    TODO: Compute the softmax loss and its gradient using no explicit loops
    Store the loss in loss and the gradient in dW. If you are not careful
    here, it is easy to run into numeric instability. Don't forget the
    regularization!
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
<<<<<<< HEAD
    # # Convert scores to probabilities
    # # Softmax Layer
    # Add normalisation factor C for numerical stability
    C = - np.max(logits, axis=1, keepdims=True)
    P = np.exp(logits + C)
    # Calculate probabilities
    probs = P / np.sum(P, axis=1, keepdims=True)
=======
    # Convert scores to probabilities
    # Softmax Layer
    #-np.max(logits)
    softmax_ = lambda s: s / np.sum(s, axis=1, keepdims=True)
    exp_sc = np.exp(logits - np.max(logits))
    probs = softmax_(exp_sc)
    N = y.shape[0]
    # Cross-Entropy Error
    corect_logprobs = -np.log(probs[range(N), y])

    print ("***********")
    print(corect_logprobs)
    loss = np.sum(corect_logprobs) / N
    # Backward pass: compute gradients
    dscores = probs
    dscores[range(N), y] -= 1
    dlogits = dscores / N
>>>>>>> netnet_merge

    K = logits.shape[0]
    # # Backward pass: compute gradients
    dlogits = np.array(probs, copy = True)
    # Subtract one for the attribute that matches the label
    dlogits[np.arange(K), y] -= 1
    # Normalise by the number of classes
    dlogits /= K

    # Add epsilon to avoid divide by zero error in loss
    epsilon = np.finfo(np.float64).eps
    loss = -np.sum(np.log(probs[np.arange(K), y] + epsilon)) / K
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return loss, dlogits
