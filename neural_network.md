Neural network: the neurons are connected together by synapses which are nothing but the connections across which a neuron can send an impulse to another neuron. When a neuron sends an excitatory signal to another neuron, then this signal will be added to all of the other inputs of that neuron.
The output: If the connection exceeds a given threshold then it will cause the target neuron to fire an action signal forward.

In computer science:
create 'networks' on a computer using matrix.
## Elements of a Neural Network:
__Input Layer__: The layer accepts input features.  
__Hidden Layer__: Nodes of the layers are not exposed to the outer world. It performs all sort of __computation on the features__ entered through the input layer and __transfer the result__ to the output layer.  
__Output Layer__: The layer bring up the information learned by the network to the outer world.  


## The training process consists of the following steps:

## Forward Propagation:
Take the inputs, multiply by the weights (just use random numbers as weights)
Let Y = WiIi = W1I1+W2I2+W3I3
Pass the result through a sigmoid formula to calculate the neuron’s output.
The Sigmoid function is used to __normalise the result between 0 and 1__:
1/1 + exp(-y)
## Back Propagation:
Calculate the error i.e the difference between the actual output and the expected output. Depending on the error, adjust the weights by multiplying the error with the input and again with the gradient of the Sigmoid curve:
Weight += Error Input Output (1-Output) ,here Output (1-Output) is derivative of sigmoid curve.

## Activation Function:
Activation function decides, whether a neuron should be activated or not by calculating weighted sum and further adding bias with it.  
The purpose of the AF(activation function) is to introduce non-linearity into the output of a neuron. If a neural network without activation function is essentially just a linear regression model.

2). __Sigmoid Function__ :
It is a function which is plotted as ‘S’ shaped graph.
- Equation : A = 1/(1 + exp(-x))
- Nature : Non-linear. Notice that X values lies between -2 to 2, Y values are very steep. This means, small changes in x would also bring about large changes in the value of Y.
- Value Range : 0 to 1
- Uses : Usually used in output layer of a binary classification, where result is either 0 or 1, as value for sigmoid function lies between 0 and 1 only so, result can be predicted easily to be 1 if value is greater than 0.5 and 0 otherwise.

3). __Tanh Function__ :- The activation that works almost always __better than sigmoid function__ is Tanh function also knows as Tangent Hyperbolic function. It’s actually mathematically shifted version of the sigmoid function. Both are similar and can be derived from each other.
Equation:

        f(x) = tanh(x) = 2/(1 + e-2x) - 1
        OR tanh(x) = 2 * sigmoid(2x) - 1
- Value Range: -1 to +1
- Nature : non-linear
- Uses : Usually used in hidden layers of a neural network as it’s values lies between -1 to 1 hence the mean for the hidden layer comes out be 0 or very close to it, hence helps in centering the data by bringing mean close to 0. This makes learning for the next layer much easier.

4). __RELU__ : Stands for Rectified linear unit. It is the most widely used activation function. Chiefly implemented in hidden layers of Neural network.

- Equation: A(x) = max(0,x). It gives an output x if x is positive and 0 otherwise.
- Value Range : [0, inf)
- Nature: non-linear, which means we can easily backpropagate the errors and have multiple layers of neurons being activated by the ReLU function.
- Uses: ReLu is less computationally expensive than tanh and sigmoid because it involves simpler mathematical operations. At a time only a few neurons are activated making the network sparse making it efficient and easy for computation.
In simple words, RELU learns much faster than sigmoid and Tanh function.

5). __Softmax Function__ :- The softmax function is also a type of sigmoid function but is handy when we are trying to handle classification problems.

- Nature :- non-linear
- Uses :- Usually used when trying to handle multiple classes. The softmax function would squeeze the outputs for each class between 0 and 1 and would also divide by the sum of the outputs.
- Ouput:- The softmax function is ideally used in the output layer of the classifier where we are actually trying to attain the probabilities to define the class of each input.
CHOOSING THE RIGHT ACTIVATION FUNCTION

The basic rule of thumb is if you really don’t know what activation function to use, then simply use __RELU as it is a general activation function__ and is used in most cases these days.
If your output is for __binary classification then, sigmoid function__ is very natural choice for output layer.
