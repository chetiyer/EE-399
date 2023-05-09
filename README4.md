# Machine Learning - Classification
Author: Chetana Iyer

This notebook provides basic applications of Feed Forward Neural Networks  v in Python, following HW4 in EE 399A, Introduction to Machine Learning for Science and Engineering, a UW course designed by J. Nathan Kutz for Spring 2023.

## Abstract: 
In this assignment, we will do classification using a feed forward Neural Network on a test data set, and the MNIST dataset. The MNIST dataset has been used extensively in research for developing and evaluating image recognition algorithms. It is often used as a benchmark dataset to compare the performance of different machine learning models for tasks such as classification and object detection. We will also compare the performance of the Feedforward Neural Network against LSTM,SVM and Decision tree classifiers. Through these exercises, we aim to gain a deeper understanding of the underlying similarities across different approaches for machine learning

## 1. Introduction

We will begin by fitting the given data to a 3-layer Feed Forward Neural Network. Then, using the first 20 points as training data, we will compute the least square error oc the models on the test data. Then we will repeat this with an even splitting of the data into training and test data. 

In additiion, we will traing a feedforward neural network on the MNIST data to classify the digits, and compare the performance against the LSTM,SVM and Decision tree classifiers. Through these exercises, we aim to gain a deeper understanding of the underlying similarities across different approaches for machine learning

## 2. Theoretical Background

One of the key concepts that underlies this assignment is the concept of a **Neural Network**. Neural Networks are mathematical models that were originally inspired by the structure of the visual cortex of cats discovered by Nobel Prize scientists Hubel and Wiesel. The scientists discovered that neuronal networks are organized hierarchically to process visual information, and since then Neural networks have been used in the fields of Machine Learning and Artificial Intelligence to learn and predict patterns based on input data. 

Neural Networks consist of several key components, here are some of the essential ones.

*Neurons:* These are the input processors of the network. They receive the inputs, and apply the weights to them and produce the appropriate output. They are typically organized in layers, commonly in the format of an input layer, one or more hidden layers and an output layer

*Weights:* Each connection between neurons in the network carries an associated weight. The weights determine the strength and priority of the connection. During training, the network adjusts these weights to optimize its performance and improve it accuracy

*Activation Function:* An activation function is applied to the weighted sum of the inputs to introduce non linearity into the network. More generally it applies the transformation that maps the input to the output 

*Layers:* Neurons are traditionally organized into layers in a neural network. The input layer receives the initial data, the hidden layers process and transform the data while the output layer produces the final output/prediction

*Loss function:* Measures the difference between the network’s predicted vs true output. During training, the network adjusts the weight values to minimize this loss. Examples include mean squared error etc. 

In this assignment, we will be examining *Feed Forward Neural Networks* - which is a type of Network. Feed Forward Neural Networks are a type of network in which information only flows in one direction: through the input layers, hidden layers and to the output layers. It is called feedforward since the data flows through the network without any loops or feedback connections. 

## 3. Algorithm Implementation and Visualizations

### Part I : Fit data to feedforward neural network, split into training/test two ways
Here, we first defined the neural network to be feedforward, and defined the loss and optimization functions. 

```# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create an instance of the network and define the loss and optimizer
net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

# Convert the data to PyTorch tensors
X = torch.from_numpy(np.arange(0,31).reshape(-1,1).astype(np.float32))
Y = torch.from_numpy(np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41,
                                40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53]).astype(np.float32))
```
Next, we split the data into test and training two different ways and calculated the least squared error of the models on the test data [which was withheld from training]

```# (ii) Using the first 20 data points as training data, fit the neural network
X_train = X[:20]
Y_train = Y[:20]

for epoch in range(1000):
    optimizer.zero_grad()
    output = net(X_train)
    loss = criterion(output, Y_train)
    loss.backward()
    optimizer.step()

# Compute the least-square error for each of these over the training points
with torch.no_grad():
    train_loss = criterion(net(X_train), Y_train)
    print('Training loss:', train_loss)

# Compute the least square error of these models on the test data which are the remaining 10 data points
X_test = X[20:]
Y_test = Y[20:]
with torch.no_grad():
    test_loss = criterion(net(X_test), Y_test)
    print('Test loss:', test_loss)


# (iii) Repeat (iii) but use the first 10 and last 10 data points as training data
X_train2 = torch.cat((X[:10], X[-10:]), dim=0)
Y_train2 = torch.cat((Y[:10], Y[-10:]), dim=0)

for epoch in range(1000):
    optimizer.zero_grad()
    output = net(X_train2)
    loss = criterion(output, Y_train2)
    loss.backward()
    optimizer.step()

# Compute the least-square error for each of these over the training points
with torch.no_grad():
    train_loss2 = criterion(net(X_train2), Y_train2)
    print('Training loss (ii):', train_loss2)

# Fit the model to the test data (which are the 10 held out middle data points)
X_test2 = X[10:20]
Y_test2 = Y[10:20]
with torch.no_grad():
    test_loss2 = criterion(net(X_test2), Y_test2)
    print('Test loss (ii):', test_loss2)
```
### Part II : Trained a feedforward neural network on the MNIST data set
```# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Flatten the images
train_images = mnist_trainset.data.reshape(mnist_trainset.data.shape[0], -1)
test_images = mnist_testset.data.reshape(mnist_testset.data.shape[0], -1)

# Perform PCA on the training images
pca = PCA(n_components=20)
train_images_pca = pca.fit_transform(train_images)

# Use the PCA model to transform the test images
test_images_pca = pca.transform(test_images)


```
##### (ii) Build a feed-forward neural network to classify the digits. 
```# Define the neural network
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of the network and define the loss and optimizer
input_size = 20 # size of PCA output
hidden_size = 50
output_size = 10 # number of classes
net = Net(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

# Convert the data to PyTorch tensors
train_labels = mnist_trainset.targets
test_labels = mnist_testset.targets
X_train = torch.from_numpy(train_images_pca.astype(np.float32))
y_train = torch.from_numpy(train_labels.numpy())
X_test = torch.from_numpy(test_images_pca.astype(np.float32))
y_test = torch.from_numpy(test_labels.numpy())

# Train the network
num_epochs = 10
batch_size = 32
for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        optimizer.zero_grad()
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        output = net(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

# Evaluate the network
net.eval()
with torch.no_grad():
    train_preds = torch.argmax(net(X_train), axis=1)
    test_preds = torch.argmax(net(X_test), axis=1)
    train_acc = torch.sum(train_preds == y_train).item() / len(y_train)
    test_acc = torch.sum(test_preds == y_test).item() / len(y_test)

print("Training accuracy:", train_acc)
print("Test accuracy:", test_acc)

```
#### Compare the results of the neural network against LSTM, SVM (support vector machines) and decision tree classifiers.

```
# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Flatten the images
train_images = mnist_trainset.data.reshape(mnist_trainset.data.shape[0], -1)
test_images = mnist_testset.data.reshape(mnist_testset.data.shape[0], -1)

# Perform PCA on the training images
pca = PCA(n_components=20)
train_images_pca = pca.fit_transform(train_images)

# Use the PCA model to transform the test images
test_images_pca = pca.transform(test_images)

# Load labels
train_labels = mnist_trainset.targets.numpy()
test_labels = mnist_testset.targets.numpy()

# Convert the data to PyTorch tensors
train_images_pca_tensor = torch.from_numpy(train_images_pca.astype(np.float32))
train_labels_tensor = torch.from_numpy(train_labels)
test_images_pca_tensor = torch.from_numpy(test_images_pca.astype(np.float32))
test_labels_tensor = torch.from_numpy(test_labels)

# Create PyTorch datasets
train_dataset = TensorDataset(train_images_pca_tensor, train_labels_tensor)
test_dataset = TensorDataset(test_images_pca_tensor, test_labels_tensor)

# Set up the DataLoader objects
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Set up the model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 20
hidden_size = 128
num_layers = 2
num_classes = 10
learning_rate = 0.001

model_lstm = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_lstm.parameters(), lr=learning_rate)

# Train the LSTM model
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = 0.0
    total = 0
    correct = 0
    for images, labels in train_loader:
        images = images.view(-1, 1, input_size).to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model_lstm(images)
        loss = criterion(outputs, labels)
        loss.backward()
       
# Evaluate the LSTM model using scikit-learn accuracy score
with torch.no_grad():
    predicted_labels = []
    true_labels = []
    for images, labels in test_loader:
        images = images.view(-1, 1, input_size).to(device)
        outputs = model_lstm(images)
        _, predicted = torch.max(outputs.data, 1)
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(true_labels, predicted_labels)
print(f'Test accuracy (LSTM): {accuracy:.4f}')
```
Calculate which digits are easiest to distinguish by looping through all combinations of digits from the data set

## 4. Results & Conclusion

**problem i)**

Training loss: tensor(1401.8444), Test loss: tensor(2446.0713)

**problem ii)**

Training loss: tensor(1333.6720), Test loss: tensor(2355.3342)

**problem iii)**

Training loss: tensor(1661.1974), Test loss: tensor(1485.2930)


**problem iv)**

Homework 1 looked at fitting data to a line,parabola and 19th degree polynomial - for the first splitting 20 points of the data each evenly into training and test, the errors were respectively

    training: 
    line fitting - 100.59
    parabola - 90.35
    19th degree polynomial - 0.016

    test:
    line fitting - 118.28
    parabola - 816.33
    19th degree polynomial - 9.014

And on this splitting, the neural networks performed worse with a training loss of 1333.67, and a test loss of 2355.33

For the second division of the data, the respective values were 

    training: 
    line fitting - 68.57
    parabola - 68.51
    19th degree polynomial - 0.536

    test: 
    line fitting - 86.64
    parabola - 84.70
    19th degree polynomial - 2576550.371

The training and test loss for the neural network was 1661.19 and 1485.2930 respectively. It performed better than the 19th degree polynomial on the testing data, but worse compared to both line fitting and parabola fitting 


**problem 2** 

The Test accuracy (LSTM) of 0.0999 is significantly lower than the Training accuracy of 0.9094666666666666 for the LSTM model, and also much lower than the Test accuracy of 0.9106 BY the Feedforward Neural Network.
    
This indicates that the LSTM model is overfitting to the training data and is not generalizing well to the test data. This could be due to various reasons such as suboptimal hyperparameters, insufficient training data, or insufficient regularization.

In the last homework assignment, we calculated the accuracy of SVM and decision tree classifiers in overall classification. The overall accuracy for classification for the SVM and Decision tree methods were 0.97628 and 0.8666. 
    
From this homework, we identified that the classification accuracy for the feed-forward neural network was 0.9106, and 0.0999 for the LSTM model.  From this we can note the order of performance in classifying MNIST digits,  as 1) SVM, 2) FFNN 3) Decision tree and lastly LSTM 
