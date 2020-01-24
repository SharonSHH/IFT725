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
    # TODO: Calculate dW and loss.  The loss and gradient must be subsequently averaged.
    # And at the end, don't forget to add the regularization term L2: reg*||w||^2
    #############################################################################
    delta = 1
    for i in range(X.shape[0]):
        y_predict = np.dot(X[i, :], W)
        sum_loss = 0
        for j in range(W.shape[1]):
            if j != y[i]:
                score = y_predict[j] - y_predict[y[i]] + delta
                loss += score if score > 0 else 0

    loss = loss/X.shape[0] + 0.5 * reg * np.sum(W*W)

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
    # TODO: Please put the result in the variable "loss".  NOTE: This function must not contain any loop
    #   Implement a vectorized version of the SVM loss function.
    #############################################################################
    loss = 0.0
    y_predict = X.dot(W)
    score = np.maximum(0, y_predict)


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
    # avez utilisées pour calculer la perte.
    #
    # TODO: Index: Instead of calculating the gradient from zero, it may be easier to
    #  reuse some of the intermediate values that you used to calculate the loss.                                 #
    #############################################################################

    dW = dW*0

    #############################################################################
    #                            FIN DE VOTRE CODE                              #
    #############################################################################

    return loss, dW
