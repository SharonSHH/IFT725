# Code adapté de projets académiques de la professeur Fei Fei Li et de ses étudiants Andrej Karpathy, Justin Johnson et autres.
# Version finale rédigée par Carl Lemaire, Vincent Ducharme et Pierre-Marc Jodoin

import numpy as np


def softmax_naive_loss_function(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Calculez la perte softmax (entropie croisée) moyenne et son         #
    #  gradient moyen avc des boucles explicites sur chaque paire (X[i], y[i]). #
    #  N'oubliez pas que l'entropie-croisée pour une paire (X[i], y[i]) est     #
    #  -log(SM[y[i]), où SM est le vecteur softmax à 10 classes de X[i]         #
    #  Pour ce qui est du gradient, vous pouvez utiliser l'equation 4.109       #
    #  du livre de Bishop.                                                      #
    # Stockez la perte dans la variable "loss" et le gradient dans "dW".        #
    # N'oubliez pas la régularisation! Afin d'éviter toute instabilité          #
    # numérique, soustrayez le score maximum de la classe de tous les scores    #
    # d'un échantillon.                                                         #
    # TODO: Calculate the average softmax loss(Cross entropy) and its average gradient with loops
    # on each pair(X[i], y[i]). The cross-entropy for a pair (X[i], y[i]) is -log(SM[y[i]]),
    # where SM is the 10-class softmax vector of X[i] for the gradient, you can use equation 4.109.
    # To avoid numerical instability, substract the maximum class score from all scores in a sample.
    #############################################################################
    loss = loss*0
    dW = dW*0

    #############################################################################
    #                         FIN DE VOTRE CODE                                 #
    #############################################################################

    return loss, dW


def softmax_vectorized_loss_function(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Calculez la perte softmax et son gradient en n'utilisant aucune     #
    #  boucle explicite.                                                        #
    # Stockez la perte dans la variable "loss" et le gradient dans "dW".        #
    # N'oubliez pas la régularisation! Afin d'éviter toute instabilité          #
    # numérique, soustrayez le score maximum de la classe de tous les scores    #
    # d'un échantillon.
    # TODO: To avoid numerical instability, subtract the maximum class score from all scores in a sample.                                                         #
    #############################################################################
    loss = loss * 0
    dW = dW * 0
    result = np.dot(W, X)
    score = np.exp(result)/np.sum(np.exp(result))
    W_norm = np.dot(W, W.transpose())
    loss = -np.log(score) + reg * W_norm

    dW = np.dot(score, X) + 2*reg*np.array(W)


    #############################################################################
    #                         FIN DE VOTRE CODE                                 #
    #############################################################################

    return loss, dW
