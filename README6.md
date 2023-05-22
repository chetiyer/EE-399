# Machine Learning - SHRED
Author: Chetana Iyer

This notebook looks at basic applications of SHRED(Shallow Recurrent Decode) following HW6 in EE 399A, Introduction to Machine Learning for Science and Engineering, a UW course designed by J. Nathan Kutz for Spring 2023.

## Abstract: 
In this assignment, we used SHRED (SHallow REcurrent Decoder), a novel neural network approach that combines recurrent and shallow decoding models to achieve accurate reconstructions and forecasting with a limited number of sensors, overcoming challenges such as noisy measurements and sensor placement.Through these exercises, we aim to gain a deeper understanding of the underlying similarities across these different approaches for machine learning

## 1. Introduction
In this homework assignment, we explore the performance of the SHRED (SHallow REcurrent Decoder) neural network model for sea-surface temperature analysis. Firstly, we download the example code from the provided GitHub repository, which includes both the code and data. Secondly, we train the SHRED model using an LSTM/decoder architecture and visualize the results to assess its effectiveness. Next, we conduct a comprehensive analysis to evaluate the model's performance in relation to the time lag variable, noise levels (by adding Gaussian noise to the data), and the number of sensors employed. Through these analyses, we gain insights into the capabilities and limitations of the SHRED model for various scenarios in sea-surface temperature analysis.

## 2. Theoretical Background

Sensing plays a vital role in scientific and engineering domains, presenting challenges when dealing with limited sensors, noisy measurements, and incomplete data. Traditional techniques rely on current sensor measurements and require carefully placed sensors or a large number of randomly placed sensors. However, we propose an innovative approach called SHRED (SHallow REcurrent Decoder) that overcomes these limitations.

SHRED utilizes a recurrent neural network (LSTM) to capture the temporal dynamics of sensors and learn a latent representation of the data over time. This enables accurate reconstructions with a reduced number of sensors and outperforms existing methods when more measurements are available. One key advantage of SHRED is its agnosticism towards sensor placement, allowing for flexibility and adaptability in various scenarios. Additionally, SHRED extracts a compressed representation of the high-dimensional state directly from sensor measurements, providing efficient on-the-fly compression for modeling physical and engineering systems. It also enables accurate forecasting based solely on sensor time-series data, making it a valuable tool for predicting temporal evolution with a minimal number of sensors.

## 3. Algorithm Implementation and Visualizations
Load in given data set:

```num_sensors = 3 
lags = 52
load_X = load_data('SST')
n = load_X.shape[0]
m = load_X.shape[1]
sensor_locations = np.random.choice(m, size=num_sensors, replace=False)
```
### Problem 1: Train the Model and plot the results

```
#divide into test and train
train_indices = np.random.choice(n - lags, size=1000, replace=False)
mask = np.ones(n - lags)
mask[train_indices] = 0
valid_test_indices = np.arange(0, n - lags)[np.where(mask!=0)[0]]
valid_indices = valid_test_indices[::2]
test_indices = valid_test_indices[1::2]
```
sklearn's MinMaxScaler is used to preprocess the data for training and we generate input/output pairs for the training, validation, and test sets. 

```sc = MinMaxScaler()
sc = sc.fit(load_X[train_indices])
transformed_X = sc.transform(load_X)

### Generate input sequences to a SHRED model
all_data_in = np.zeros((n - lags, lags, num_sensors))
for i in range(len(all_data_in)):
    all_data_in[i] = transformed_X[i:i+lags, sensor_locations]

### Generate training validation and test datasets both for reconstruction of states and forecasting sensors
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

### -1 to have output be at the same time as final sensor measurements
train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
test_dataset = TimeSeriesDataset(test_data_in, test_data_out)
```
Train the model using training & validation sets

```shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=1000, lr=1e-3, verbose=True, patience=5)
```

### Analysis of Performance as a function for the time lag variable

```

# Define a range of time lag values to evaluate
lag_values = [10, 20, 30, 40, 50]

validation_errors = []

# Calculate the validation errors (already available in the `validation_errors` list)
for lag in lag_values:
    lags = lag

    # Generate input sequences to a SHRED model
    all_data_in = np.zeros((n - lags, lags, num_sensors))
    for i in range(len(all_data_in)):
        all_data_in[i] = transformed_X[i:i+lags, sensor_locations]

    # Generate training, validation, and test datasets both for reconstruction of states and forecasting sensors
    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
    valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
    test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

    # -1 to have output be at the same time as final sensor measurements
    train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
    valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
    test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

    # Create new instances of the TimeSeriesDataset
    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)

    # Create and train the model
    shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=1000, lr=1e-3, verbose=True, patience=5)
    
    # Obtain predictions from the model
    shred.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predicted_valid_out = shred(valid_data_in)
    
    # Calculate validation error (RMSE)
    mse = torch.mean((predicted_valid_out - valid_data_out)**2)
    rmse = torch.sqrt(mse)
    validation_error = rmse.item()
    
    # Store the validation error
    validation_errors.append(validation_error)
    
# Plot the validation errors against the time lag values
plt.plot(lag_values, validation_errors)
plt.xlabel('Time Lag')
plt.ylabel('Validation Error (RMSE)')
plt.title('Performance vs Time Lag')
plt.show()

```
### Analysis of performance as a function of noise (with Gaussian Noise added) 

```
# Define the range of noise levels to evaluate
noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]  # Standard deviation of the Gaussian noise

validation_errors = []

# Calculate the validation errors (already available in the `validation_errors` list)
for noise_level in noise_levels:
    # Generate noisy input data
    noisy_X = transformed_X + np.random.normal(0, noise_level, size=transformed_X.shape)

    # Generate training, validation, and test datasets both for reconstruction of states and forecasting sensors
    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
    valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
    test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

    # -1 to have output be at the same time as final sensor measurements
    train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
    valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
    test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

    # Create new instances of the TimeSeriesDataset with noisy input data
    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)

    # Create and train the model
    shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=1000, lr=1e-3, verbose=True, patience=5)
    
    # Obtain predictions from the model using noisy input data
    shred.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predicted_valid_out = shred(valid_data_in)
    
    # Calculate validation error (RMSE) using noisy input data
    mse = torch.mean((predicted_valid_out - valid_data_out)**2)
    rmse = torch.sqrt(mse)
    validation_error = rmse.item()
    
    # Store the validation error
    validation_errors.append(validation_error)
    
# Plot the validation errors against the noise levels
plt.plot(noise_levels, validation_errors)
plt.xlabel('Noise Level (Standard Deviation)')
plt.ylabel('Validation Error (RMSE)')
plt.title('Performance vs Noise Level')
plt.show()

```

### Analysis of performance as a function of the number of sensors

```
# Define the range of number of sensors to evaluate
sensor_counts = [2, 4, 6, 8, 10]

validation_errors = []

# Calculate the validation errors (already available in the `validation_errors` list)
for sensor_count in sensor_counts:
    # Randomly select sensor locations
    sensor_locations = np.random.choice(m, size=sensor_count, replace=False)

    # Generate input sequences based on the selected sensor locations
    all_data_in = np.zeros((n - lags, lags, sensor_count))
    for i in range(len(all_data_in)):
        all_data_in[i] = transformed_X[i:i+lags, sensor_locations]

    # Generate training, validation, and test datasets based on the selected sensor locations
    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
    valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
    test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

    # Create new instances of the TimeSeriesDataset based on the selected sensor locations
    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)

    # Create and train the model
    shred = models.SHRED(sensor_count, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=1000, lr=1e-3, verbose=True, patience=5)
    
    # Calculate validation errors
    predicted_valid_out = shred.predict(valid_dataset)
    validation_error = calculate_error(predicted_valid_out, valid_data_out)  # Replace with appropriate error metric calculation

    # Store the validation error
    validation_errors.append(validation_error)
    
# Plot the validation errors against the number of sensors
plt.plot(sensor_counts, validation_errors)
plt.xlabel('Number of Sensors')
plt.ylabel('Validation Error')
plt.title('Performance vs Number of Sensors')
plt.show()

```

## 4. Results & Conclusion

**problem 2** ![here](https://github.com/chetiyer/EE-399/blob/main/3_1.png)

The R value is 343, which is the rank of digit space

**problem 4** ![here](https://github.com/chetiyer/EE-399/blob/main/3_2.png)  

**problem A** the LDA's classifcation accuracy for distinguishing 4 and 9 is 0.96 

**problem B** the LDA's classifcation accuracy for distinguishing 2,3,5 is 0.93

**problem C** 5 and 8 were calculated to be the hardest to distinguish with an accuracy of .95

**problem D** 6 and 9 were calculated to be the easiest to distinguish with an accuracy of 1.0

**problem E** The overall accuracy for classification between the SVM and Decision tree methods were 0.97628 and 0.8666

**problem F** On the hard-to-distinguish digits: the accuracy for SVM was 96.80%, and 95.97% for decision tree
![here](https://github.com/chetiyer/EE-399/blob/main/3_3.png)
![here](https://github.com/chetiyer/EE-399/blob/main/3_4.png)

For the easy to distinguish digits: the accuracy for SVM was 99.89%, and 99.24% for decision tree
![here](https://github.com/chetiyer/EE-399/blob/main/3_5.png)
![here](https://github.com/chetiyer/EE-399/blob/main/3_6.png)
