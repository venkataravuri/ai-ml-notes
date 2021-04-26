
# Analytic Vidhya Course
import torch

# Input tensor
X = torch.Tensor([[1,0,1,0], [1,0,1, 1], [0,1,0,1]])

#Output
y = torch.Tensor([[1], [1], [0]])

print(X, '\n')
print(y, '\n')

# Sigmoid Function
def sigmoid(x):
    return 1/(1+torch.exp(-x))

# Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Variable initialization

epoch = 7000 #Setting training iterations
lr = 0.1 # Learning rate

inputlayer_neurons = X.shape[1] #number of features in data set
print('Input layer neurons: ', X.shape[1])
hiddenlayer_neurons = 3 # number of hidden layer neurons
output_neurons = 1 #number of neurons in output layer

# weight and bias initialization
wh = torch.randn(inputlayer_neurons, hiddenlayer_neurons).type(torch.FloatTensor)
print('Random input weights: ', wh)
bh = torch.randn(1, hiddenlayer_neurons).type(torch.FloatTensor)
wout = torch.randn(hiddenlayer_neurons, output_neurons)
bout = torch.randn(1, output_neurons)

for i in range(epoch):

    #Forward Propagation
    hidden_layer_input1 = torch.mm(X, wh)
    hidden_layer_input = hidden_layer_input1 + bh
    hidden_layer_activations = sigmoid(hidden_layer_input)

    output_layer_input1 = torch.mm(hidden_layer_activations, wout)
    output_layer_input = output_layer_input1 + bout
    output = sigmoid(output_layer_input)

    # Backpropagation
    E = y-output
    slope_output_layer = derivatives_sigmoid(output)
    slope_hidden_layer = derivatives_sigmoid(hidden_layer_activations)
    d_output = E * slope_output_layer
    Error_at_hidden_layer = torch.mm(d_output, wout.t())
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer

    wout += torch.mm(hidden_layer_activations.t(), d_output) * lr
    bout += d_output.sum() * lr
    wh += torch.mm(X.t(), d_hiddenlayer) * lr
    bh += d_output.sum() * lr

print('Calculated weights: ', wh, '\n')
print('actual: \n', y, '\n')
print('predicted: \n', output)
