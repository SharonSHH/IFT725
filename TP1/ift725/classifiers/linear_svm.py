# Code adapté de projets académiques de la professeur Fei Fei Li et de ses étudiants Andrej Karpathy, Justin Johnson et autres.
# Version finale rédigée par Carl Lemaire, Vincent Ducharme et Pierre-Marc Jodoin

import numpy as np


def svm_naive_loss_function(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    loss = 0.0
    #############################################################################
    # TODO: Calculez le gradient "dW" et la perte "loss" et stockez le résultat #
    #  dans "dW et dans "loss".                                                 #
    #  Pour cette implementation, vous devez naivement boucler sur chaque pair  #
    #  (X[i],y[i]), déterminer la perte (loss) ainsi que le gradient (voir      #
    #  exemple dans les notes de cours).  La loss ainsi que le gradient doivent #
    #  être par la suite moyennés.  Et, à la fin, n'oubliez pas d'ajouter le    #
    #  terme de régularisation L2 : reg*||w||^2                                 #
    #############################################################################
    delta = 1
    num_train = X.shape[0]
    num_class = W.shape[1]
    for i in range(num_train):
        scores = np.dot(X[i], W)
        count_loss = 0
        for j in range(num_class):
            if j == y[i]:
                continue
            margin = scores[j] - scores[y[i]] + delta
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                count_loss += 1
        dW[:, y[i]] += (-1) * count_loss * X[i]

    # calculate the average loss and gradient
    loss /= num_train
    dW /= num_train

    # Add regularization term L2:reg*||w||^2 to the loss and gradient
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    #############################################################################
    #                            FIN DE VOTRE CODE                              #
    #############################################################################

    return loss, dW


def svm_vectorized_loss_function(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO: Implémentez une version vectorisée de la fonction de perte SVM.     #
    # Veuillez mettre le résultat dans la variable "loss".                      #
    # NOTE : Cette fonction ne doit contenir aucune boucle                      #
    #############################################################################
    num_train = X.shape[0]
    scores = X.dot(W)
    delta = np.ones(scores.shape)
    correct_class_score = scores[range(num_train), y].reshape(num_train, 1)
    margin = np.maximum(0, scores - correct_class_score + delta)

    # Do not consider correct class in loss
    margin[range(num_train), y] = 0
    loss = margin.sum()/num_train
    loss += reg * np.sum(W * W)
    #############################################################################
    #                            FIN DE VOTRE CODE                              #
    #############################################################################

    #############################################################################
    # TODO: Implémentez une version vectorisée du calcul du gradient de la      #
    #  perte SVM.                                                               #
    # Stockez le résultat dans "dW".                                            #
    #                                                                           #
    # Indice: Au lieu de calculer le gradient à partir de zéro, il peut être    #
    # plus facile de réutiliser certaines des valeurs intermédiaires que vous   #
    # avez utilisées pour calculer la perte.                                    #
    #############################################################################
    dW = dW*0
    margin[margin > 0] = 1
    margin_count = margin.sum(axis=1)
    margin[range(X.shape[0]), y] -= margin_count
    dW = X.T.dot(margin)/num_train + 2 * reg * W
    #############################################################################
    #                            FIN DE VOTRE CODE                              #
    #############################################################################

    return loss, dW
