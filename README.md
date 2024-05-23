# MNIST
Use MNIST database

Two layer neural network that classifies images of handwritten digits and tell us what digit is written in that image 

28 x 28 training images 

A^0 = X  (784 x m)

Z^1 = W^1 A^0 + b^1 (Multiplying by corresponding weights and adding constant bias)

Activation function is needed because based on above equation, the second layer would just be a linear combiantion of the first (input) layer and does not give any interesting function : Sigmoid, Tanh

Applying activation functions like tanh or sigmoid makes the function non linear making it a hidden layer. 

Relu : Rectified linear unit.

If Relu(x) > 0 then x otherwise 0.

Softmax activation function gives probability of the output between 0-1 of it being the particular digit. 

Backward propagation 

Network might learn to predict correctly for particular example (training data) but not generalised data so set us aside cross validation data that you don't train on and check hyper parameteres on that data. Eliminates overfitting. 










