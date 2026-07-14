from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights. (3073, 10)
    - X: A numpy array of shape (N, D) containing a minibatch of data. (500, 3073)
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1] #column: 10 classes
    num_train = X.shape[0] #row: Of size N (no of input images)
    for i in range(num_train):
        scores = X[i].dot(W) #(3073,).(3073,10)-> (10,)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores) #to prevent memory overflow caused be (e^ large number)
        p = np.exp(scores)
        p /= p.sum()  # normalize (also to get probability ditribution)
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss, consider only correct class

        for j in range(num_classes):
          if j == y[i]:                 # if correct clas predicted
            grad_score = (p[j]-1) * X[i] # chain rule (loss is 1- pred as true class prob should be 1)
            dW[:,j] += grad_score #gradient for all weights of class j
          else:
            grad_score = p[j] * X[i] # chain rule
            dW[:,j] += grad_score

    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W) #Average loss + L2 Regularization

    #############################################################################
    # TODO:   [done]                                                                  #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    #dW has dimentions (D,C)
    dW /= num_train
    dW += 2 * reg* W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
        Inputs:
    - W: A numpy array of shape (D, C) containing weights. (3073, 10)
    - X: A numpy array of shape (N, D) containing a minibatch of data. (500, 3073)
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0] #row: Of size N (no of input images)

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################
    scores = np.dot(X, W) #(N,D)*(D,C)=(N,C)
    scores -= np.max(scores, axis=1, keepdims=True) #numeric stability, also keep column dim for broascasting
    p= np.exp(scores)
    p /= np.sum(p, axis=1, keepdims=True) #softmax prob
    correct_p = p[np.arange(num_train),y] #consider prob of only correct class, returns (N,) 1D array
    logp = -(np.log(correct_p))
    loss = np.mean(logp) + reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    #need to subtract 1 from the prob of correct label in each row in matrix p
    p[np.arange(num_train),y] -= 1
    #p has dimensions (N,C), X has dimensions (N,D), we need grad_score with dim (D,C)
    X = np.transpose(X)
    grad_score = np.dot(X,p)
    dW += grad_score
    dW = (dW/num_train) + 2* reg * W #averaged and we added the L2 term

    return loss, dW
