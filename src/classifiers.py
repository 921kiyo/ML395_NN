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
    loss = np.exp(logits-np.max(logits)) / np.sum(np.exp(logits-np.max(logits)), axis=0)
    SM = logits.reshape((5,1,-1))
    print(SM.shape)
    print("****************")
    
    print(np.transpose(SM,(0,2,1)).shape)
    tenbyten = [np.dot(x,y) for y,x in zip(SM,np.transpose(SM,(0,2,1)))]
    diags = [np.diag(y) for y in logits]
    print(diags[0].shape)
    #tenbyten = np.dot(SM,np.transpose(SM,(0,2,1)))
    next_array = [x - y for x,y in zip(diags,tenbyten)]
    print(next_array[0].shape)
    print(logits.shape)
    dlogits = np.array([np.dot(y,x) for x,y in zip(next_array,logits)])
    print(dlogits.shape)

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return loss, dlogits
