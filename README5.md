# Machine Learning - Neural Networks for Dynamic Systems
Author: Chetana Iyer

This notebook provides basic applications of Neural Networks v in Python, following HW5 in EE 399A, Introduction to Machine Learning for Science and Engineering, a UW course designed by J. Nathan Kutz for Spring 2023.

## Abstract: 
In this assignment, we will be looking at how neural networks can be used for future state predictions of dynamic systems - based off the Lorenz' system of differential equations

## 1. Introduction

We will begin by training a neural network to advance the solution from t to t + delta t, and a given set of rho values , and test the model for future state predictions on a different set of rho values. Then this will be repeated using feed-forward,LSTM, RNN and Echo State Networks to compare the level to which the dynamics are forecasted fitting the given data to a 3-layer Feed Forward Neural Network. Then, using the first 20 points as training data, we will compute the least square error oc the models on the test data. Then we will repeat this with an even splitting of the data into training and test data. 

Through these exercises, we aim to gain a deeper understanding of the underlying similarities across different approaches for machine learning

## 2. Theoretical Background

This assignment is based off the concept of the **Lorenz Equations**. The Lorenz equations are a set of mathematical equations that describe a simplified model of weather patterns and fluid flow. They were first introduced by the meteorologist Edward Lorenz in 1963.

The equations consist of three variables: x, y, and z. These variables represent different properties of the fluid or system being modeled. The equations describe how these variables change over time.

In simple terms, the Lorenz equations tell us how the fluid or system evolves based on a few key factors: the current state of the system and how it interacts with itself. The equations are nonlinear, meaning that small changes in the initial conditions can lead to significantly different outcomes over time. This is known as the "butterfly effect" and is one of the most famous aspects of the Lorenz equations.


One of the key concepts that underlies this assignment is the concept of a **Neural Network**. Neural Networks are mathematical models that were originally inspired by the structure of the visual cortex of cats discovered by Nobel Prize scientists Hubel and Wiesel. The scientists discovered that neuronal networks are organized hierarchically to process visual information, and since then Neural networks have been used in the fields of Machine Learning and Artificial Intelligence to learn and predict patterns based on input data. 

Neural Networks consist of several key components, here are some of the essential ones.

*Neurons:* These are the input processors of the network. They receive the inputs, and apply the weights to them and produce the appropriate output. They are typically organized in layers, commonly in the format of an input layer, one or more hidden layers and an output layer

*Weights:* Each connection between neurons in the network carries an associated weight. The weights determine the strength and priority of the connection. During training, the network adjusts these weights to optimize its performance and improve it accuracy

*Activation Function:* An activation function is applied to the weighted sum of the inputs to introduce non linearity into the network. More generally it applies the transformation that maps the input to the output 

*Layers:* Neurons are traditionally organized into layers in a neural network. The input layer receives the initial data, the hidden layers process and transform the data while the output layer produces the final output/prediction

*Loss function:* Measures the difference between the network’s predicted vs true output. During training, the network adjusts the weight values to minimize this loss. Examples include mean squared error etc. 

In this assignment, we will be comparing the performance of  *Feed Forward Neural Network, LSTM, RNN and Echo State Networks s* - which is are all types of Neural Network. Feed Forward Neural Networks are a type of network in which information only flows in one direction: through the input layers, hidden layers and to the output layers.



## 3. Algorithm Implementation and Visualizations

### Part I : Train a NN to advance the solution from t to t + ∆t for ρ = 10, 28 and 40. Now see how well your NN works for future state prediction for ρ = 17 and ρ = 35.

``` 
dt = 0.01
T = 8
t = np.arange(0,T+dt,dt)
sequence_length = len(t) - 1  # Define the sequence length
beta = 8/3
sigma = 10
rho = 28


nn_input = np.zeros((100*(len(t)-1),3))
nn_output = np.zeros_like(nn_input)

fig,ax = plt.subplots(1,1,subplot_kw={'projection': '3d'})


def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho):
    x, y, z = x_y_z
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

np.random.seed(123)
x0 = -15 + 30 * np.random.random((100, 3))

x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t)
                  for x0_j in x0])

for j in range(100):
    nn_input[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_t[j,:-1,:]
    nn_output[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_t[j,1:,:]
    x, y, z = x_t[j,:,:].T
    ax.plot(x, y, z,linewidth=1)
    ax.scatter(x0[j,0],x0[j,1],x0[j,2],color='r')
             
ax.view_init(18, -113)
plt.show()

def advance_lorenz(x_y_z, t0, delta_t, sigma=sigma, beta=beta, rho=rho):
    x, y, z = x_y_z
    dx_dt = lorenz_deriv(x_y_z, t0, sigma, beta, rho)
    x_new = x + dx_dt[0] * delta_t
    y_new = y + dx_dt[1] * delta_t
    z_new = z + dx_dt[2] * delta_t
    return [x_new, y_new, z_new]

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define activation functions
def logsig(x):
    return 1 / (1 + torch.exp(-x))

def radbas(x):
    return torch.exp(-torch.pow(x, 2))

def purelin(x):
    return x

# Define the model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(in_features=3, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=10)
        self.fc3 = nn.Linear(in_features=10, out_features=3)
        
        # Initialize the weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
        # Initialize the biases
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

        
    def forward(self, x):
        x = logsig(self.fc1(x))
        x = radbas(self.fc2(x))
        x = purelin(self.fc3(x))
        return x

def generate_data(p):
    # Set the desired ρ value
    rho = p

    # Generate the initial conditions
    np.random.seed(123)
    x0 = -15 + 30 * np.random.random((100, 3))

    # Generate the time vector
    dt = 0.01
    T = 8
    t = np.arange(0, T + dt, dt)

    # Generate the input and output arrays
    nn_input = np.zeros((100 * (len(t) - 1), 3))
    nn_output = np.zeros_like(nn_input)

    # Generate the Lorenz system trajectories
    x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t)
                      for x0_j in x0])

    for j in range(100):
        nn_input[j * (len(t) - 1):(j + 1) * (len(t) - 1), :] = x_t[j, :-1, :]
        nn_output[j * (len(t) - 1):(j + 1) * (len(t) - 1), :] = x_t[j, 1:, :]

    # Convert numpy arrays to PyTorch tensors
    nn_input = torch.from_numpy(nn_input).float()
    nn_output = torch.from_numpy(nn_output).float()

    return nn_input, nn_output


p_values = [10, 28, 40]
epochs = 30

for p in p_values:
    # Generate the training data for the current ρ value
    nn_input, nn_output = generate_data(p)

    # Create model instance
    model = MyModel()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Train the model
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(nn_input)
        loss = criterion(outputs, nn_output)
        loss.backward()

        # Apply gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Adjust max_norm as needed

        optimizer.step()
        print(f"ρ={p}, Epoch {epoch + 1}, loss={loss.item():.4f}")

# Evaluate the model on future state
criterion = nn.MSELoss(reduction='mean')  # Change the reduction mode to 'mean'

for p_future in [17, 35]:
    nn_input_future, nn_output_future = generate_data(p_future)
    predicted_output = model(nn_input_future)

    # Calculate the loss
    loss_future = criterion(predicted_output, nn_output_future)
    print(f"Loss for future state (ρ = {p_future}): {loss_future.item():.4f}")

```

### Part II : Compare feed-forward, LSTM, RNN and Echo State Networks for forecasting the dynamics.
```# Feed-forward neural network model
class FeedForwardModel(nn.Module):
    def __init__(self):
        super(FeedForwardModel, self).__init__()
        self.fc1 = nn.Linear(in_features=3, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=10)
        self.fc3 = nn.Linear(in_features=10, out_features=3)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=10, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=10, out_features=3)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = self.fc(h_n[-1])
        return x

# RNN model
class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=3, hidden_size=10, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=10, out_features=3)

    def forward(self, x):
        _, h_n = self.rnn(x)
        x = self.fc(h_n[-1])
        return x

p_values = [10, 28, 40]
epochs = 30

model_types = {
    "Feed-Forward": FeedForwardModel(),
    "LSTM": LSTMModel(),
    "RNN": RNNModel(),
}

for model_name, model in model_types.items():
    print(f"Training {model_name} model")

    for p in p_values:
        # Generate the training data for the current ρ value
        nn_input, nn_output = generate_data(p)

        # Create model instance
        model = model_types[model_name]

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


    # Train the model
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(nn_input)
        loss = criterion(outputs, nn_output)
        loss.backward()

        # Apply gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        print(f"ρ={p}, Epoch {epoch + 1}, loss={loss.item():.4f}")

# Evaluate the model on future state
criterion = nn.MSELoss(reduction='mean')

for p_future in [17, 35]:
    nn_input_future, nn_output_future = generate_data(p_future)
    predicted_output = model(nn_input_future)

    # Calculate the loss
    loss_future = criterion(predicted_output, nn_output_future)
    print(f"{model_name} - Loss for future state (ρ = {p_future}): {loss_future.item():.4f}")

```

## 4. Results & Conclusion

**problem 1)**

Loss for future state (ρ = 17): 242.5168
Loss for future state (ρ = 35): 242.5168

**problem 2)**

RNN - Loss for future state (ρ = 17): 188.2867
RNN - Loss for future state (ρ = 35): 188.2867
