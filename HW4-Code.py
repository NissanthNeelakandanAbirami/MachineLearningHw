import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

df = pd.read_csv('train_data.csv') 
df_test = pd.read_csv('test_data.csv')

df.describe()

df.head()

# .values is needed here to transfer a Pandas Series into numpy array

X_train_unnorm=df[['X1 transaction date','X2 house age','X3 distance to the nearest MRT station','X4 number of convenience stores','X5 latitude','X6 longitude']].values
X_test_unnorm = df_test[['X1 transaction date','X2 house age','X3 distance to the nearest MRT station','X4 number of convenience stores','X5 latitude','X6 longitude']].values


y_train=df['Y house price of unit area'].values
y_test = df_test['Y house price of unit area'].values

print('there are {} of samples in training data'.format(len(y_train)))
print('there are {} of samples in  test data'.format(len(y_test)))

from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()
X_train = standard_scaler.fit_transform(X_train_unnorm)
X_test = standard_scaler.transform(X_test_unnorm)

'''
Augment the data for both X_train and X_test
'''
####################################
########### Your code here #########
####################################

X_train_aug = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_aug = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

print('there are {} of samples in training data'.format(len(X_train_aug)))
print('there are {} of samples in  test data'.format(len(X_test_aug)))


import time
start_time = time.time()
####################################
########### Your code here #########
####################################

W = np.dot(np.linalg.pinv(X_train_aug), y_train)

end_time = time.time()
print('calculating the Moore Penrose pseudoinverse need {} seconds'.format(end_time-start_time))

MSE_cost_train = 0.0
MAPE_train = 0.0

MSE_cost_test = 0.0
MAPE_test = 0.0

def mse_cost(X,y,w):
  cost = 0.0
  """
    Compute mean square loss function
     Args:
      X (ndarray (N,D+1)): augmented data, m samples and each sample has dimension D+1
      y (ndarray (N,)) :   ground-truth label y
      w (ndarray (D+1,)) : model parameters

    Returns:
      cost (scalar): cost
    """
    #######################################
    ########### Your code here ############
    #######################################

  N = X.shape[0]  # Number of samples
  predictions = X.dot(w)
  errors = predictions - y
  cost = (1/N) * np.dot(errors.T, errors)


  return cost

def MAPE_value(X,y,w):
  mape = 0.0
  """
    Compute mape function
    Args:
      X (ndarray (N,D+1)): augmented data, m samples and each sample has dimension D+1
      y (ndarray (N,)) :   ground-truth label y
      w (ndarray (D+1,)) : model parameters
    Returns:
      mape (scalar): cost
    """
    #######################################
    ########### Your code here ############
    #######################################

  N = X.shape[0]  # Number of samples
  predictions = X.dot(w)
  errors = np.abs((y - predictions) / y)
  mape = 100 * np.mean(errors)

  return mape

# Calculate MSE and MAPE for training data
MSE_cost_train = mse_cost(X_train_aug, y_train, W)
MAPE_train = MAPE_value(X_train_aug, y_train, W)

# Calculate MSE and MAPE for test data
MSE_cost_test = mse_cost(X_test_aug, y_test, W)
MAPE_test = MAPE_value(X_test_aug, y_test, W)


print('training mse cost is {}'.format(MSE_cost_train))
print('training MAPE value is {}'.format(MAPE_train))

print('test mse cost is {}'.format(MSE_cost_test))
print('test MAPE value is {}'.format(MAPE_test))

class MultiVariate_Linear_Regression():
  def __init__(self,X_train,y_train):
    '''
    Args:
      X_train: (ndarray (N,D+1)): augmented train data features, which has N samples and each sample has D+1 dimension
      y_train: (ndarry (N,)): train data labels
    '''
    self.X_train = X_train
    self.y_train = y_train
    self.data_length = len(X_train)
    self.data_dimension = X_train.shape[1]
    self.w = np.zeros(self.data_dimension) # the weight of linear regression model


  def fit_train(self,epochs=40,learning_rate=0.01,batch_size=20):
    """
    To implement minibatch gradient variant 2, you need:
    for each epoch
      1. shuffle whole dataset
      2. get the mini batch data
      3. compute the gradient with regard to each sample in the mini batch
      4. Caculated the gradient needed for update weight and bias
      4. apply the gradient descent to the weight (self.w) and bias (self.b)
    5. calculate and restore the mse_loss using the new weight (just for visualization)
    """

    """
    Tips:
    A proper routine for writing the code is: mse_mape_cost() --> gradient_per_sample() --> fit_train() --> predict()
    """

    """
    implements minibatch gradient descent variant2 for linear regression
    Args:
      epochs (scalar): number of total epochs
      learning_rate (scalar): learning rate for updating the weight
      batch_size (scalar)

    Returns:
      J_hist (list): a list containing the mse loss for each epoch
      MAPE_hist (list): a list containing the MAPE for each epoch
    """

    """
    Attention:
    The computation time should only record the gradient descent!!!!
     Other parts, including data reading, plot or MSE/MAPE calculation should not be included.
    """

    J_hist = [] # history of mse_loss, used for plot the learning curve
    MAPE_hist = []  # history of MAPE
    epoch_time=[]
    batch_num = int(np.ceil(self.data_length / batch_size))
    for epoch_idx in range(epochs):
      #########################################
      ###############Your code here###########
      #########################################
      start_time = time.time()  

      indices = np.arange(self.data_length)
      np.random.shuffle(indices)
      X_shuffled = self.X_train[indices]
      y_shuffled = self.y_train[indices]
            
      for i in range(0, self.data_length, batch_size):
                # Extract minibatches
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
                
        gradients = np.zeros_like(self.w)
            
        for j in range(len(X_batch)):
          x = X_batch[j]
          y = y_batch[j]
                # Compute gradient for the single sample
          gradients += self._compute_gradient_per_sample(x, y)
            
            # Average the gradients over the minibatch and update the weights
        self.w -= (learning_rate / len(X_batch)) * gradients



      # at the end of each epoch, call mse_loss
      J_wb, MAPE = self.mse_mape_cost(self.X_train,self.y_train)
      J_hist.append(J_wb)
      MAPE_hist.append(MAPE)

      end_time = time.time()  # End time for the epoch
      epoch_time.append(end_time - start_time)

      # Print the weight vector after the last epoch
    print("The resulting weight vector ŵ from the training set:", self.w)

        # Compute MSE and MAPE on the training and test sets after the last epoch
    final_mse_train, final_mape_train = self.mse_mape_cost(self.X_train, self.y_train)

        # Print the final MSE and MAPE for both training and test sets
    print("The resulting MSE on the training set after the last epoch:", final_mse_train)
    print("The resulting MAPE on the training set after the last epoch:", final_mape_train)

        # Determine the number of epochs to reach MSE < 85 on the training set
    epochs_to_85 = next((i for i, mse in enumerate(J_hist) if mse < 85), None)
    if epochs_to_85 is not None:
        print("Number of epochs to reach MSE < 85 on the training set:", epochs_to_85 + 1)
    else:
        print("MSE < 85 was not reached during the training.")

        # Print the total computation time to the last epoch
    total_time = sum(epoch_time)
    print("The total computation time to the last epoch:", total_time)

        # If MSE < 85 was reached, print the computation time up to that point
    if epochs_to_85 is not None:
        time_to_85 = sum(epoch_time[:epochs_to_85+1])
        print("The computation time to reach MSE < 85 on the training set:", time_to_85)
    else:
        print("The computation time to reach MSE < 85 was not applicable since the condition was not met.")


    return J_hist, MAPE_hist




  def _compute_gradient_per_sample(self,x,y):
    """
    Computes the single-sample gradient for linear regression

    Args:
      x (ndarray (D+1,)): augmented data, a sample with D+1 dimension
      y (scaler) : target values
        w (ndarray (D+1,)) : model parameters (using self.w)

    Returns:
      dj_dw (ndarray (n,)): the gradient of the cost w.r.t. the parameters w.

      
    """
    prediction = np.dot(self.w, x)  
    error = prediction - y  # Compute the prediction error
    dj_dw = error * x  # Compute the gradient of the cost w.r.t. the parameters w
    return dj_dw
   


  def mse_mape_cost(self,X, y):
    """
    Compute mean square and MAPE loss functions

    Args:
      X (ndarray (N,D+1)): augmented data, m samples and each sample has dimension D+1
      y (ndarray (N,)) : target values
        w (ndarray (D+1,)) : model parameters (using self.w)

    Returns:
      mse_loss (scalar): cost
      MAPE (scalar): mean absolute percentage error
    """

    #########################################
    ###############Your code here###########
    #########################################

    predictions = np.dot(X, self.w)  # Compute predictions for all samples
    errors = predictions - y  # Compute errors
    mse_loss = np.mean(errors ** 2)  # Compute MSE
    mape = np.mean(np.abs(errors / y)) * 100  # Compute MAPE
    return mse_loss, mape


  def predict(self,X_test):
    """
      Compute output prediction for each data point in X_test
      Args:
        X_test (ndarray (M,D+1)): augmented data, M examples and each sample has dimension D+1
      Returns:
        y_pred (ndarray (M,)): prediction
    """
    #########################################
    ###############Your code here###########
    #########################################
    y_pred = np.dot(X_test, self.w)
    return y_pred



import numpy as np

model = MultiVariate_Linear_Regression(X_train_aug, y_train)

J_hist, MAPE_hist = model.fit_train(epochs=40, learning_rate=0.01, batch_size=20)

"""
Tips:
Based on the output of fit_train(), plot the figure use plt.plot()
"""
#########################################
###############Your code here###########
#########################################

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(J_hist, label='MSE Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training MSE Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(MAPE_hist, label='MAPE')
plt.xlabel('Epoch')
plt.ylabel('MAPE (%)')
plt.title('Training MAPE')
plt.legend()

plt.show()


"""
Predict the test data by calling predict()
"""
#########################################
###############Your code here###########
#########################################

y_pred = model.predict(X_test_aug)



# Compute the Mean Squared Error on the test set
mse_test = np.mean((y_test - y_pred) ** 2)

# Compute the Mean Absolute Percentage Error on the test 
mape_test = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Print the test MSE and MAPE
print(f'Test MSE: {mse_test:.3f}')
print(f'Test MAPE: {mape_test:.2f}%')


import numpy as np
import matplotlib.pyplot as plt
import time

class MultiVariate_Linear_Regression():
  def __init__(self,X_train,y_train):
    '''
    Args:
      X_train: (ndarray (N,D+1)): augmented train data features, which has N samples and each sample has D+1 dimension
      y_train: (ndarry (N,)): train data labels
    '''
    self.X_train = X_train
    self.y_train = y_train
    self.data_length = len(X_train)
    self.data_dimension = X_train.shape[1]
    self.w = np.zeros(self.data_dimension) # the weight of linear regression model


  def fit_train(self,epochs=40,learning_rate=0.01,batch_size=20):
    """
    To implement minibatch gradient variant 2, you need:
    for each epoch
      1. shuffle whole dataset
      2. get the mini batch data
      3. compute the gradient with regard to each sample in the mini batch
      4. Caculated the gradient needed for update weight and bias
      4. apply the gradient descent to the weight (self.w) and bias (self.b)
    5. calculate and restore the mse_loss using the new weight (just for visualization)
    """

    """
    Tips:
    A proper routine for writing the code is: mse_mape_cost() --> gradient_per_sample() --> fit_train() --> predict()
    """

    """
    implements minibatch gradient descent variant2 for linear regression
    Args:
      epochs (scalar): number of total epochs
      learning_rate (scalar): learning rate for updating the weight
      batch_size (scalar)

    Returns:
      J_hist (list): a list containing the mse loss for each epoch
      MAPE_hist (list): a list containing the MAPE for each epoch
    """

    """
    Attention:
    The computation time should only record the gradient descent!!!!
     Other parts, including data reading, plot or MSE/MAPE calculation should not be included.
    """

    J_hist = [] # history of mse_loss, used for plot the learning curve
    MAPE_hist = []  # history of MAPE
    epoch_time=[]
    batch_num = int(np.ceil(self.data_length / batch_size))
    for epoch_idx in range(epochs):
      #########################################
      ###############Your code here###########
      #########################################
      start_time = time.time()  

      indices = np.arange(self.data_length)
      np.random.shuffle(indices)
      X_shuffled = self.X_train[indices]
      y_shuffled = self.y_train[indices]
            
      for i in range(0, self.data_length, batch_size):
                # Extract minibatches
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
                
        gradients = np.zeros_like(self.w)
            
        for j in range(len(X_batch)):
          x = X_batch[j]
          y = y_batch[j]
                # Compute gradient for the single sample
          gradients += self._compute_gradient_per_sample(x, y)
            
            # Average the gradients over the minibatch and update the weights
        self.w -= (learning_rate / len(X_batch)) * gradients



      # at the end of each epoch, call mse_loss
      J_wb, MAPE = self.mse_mape_cost(self.X_train,self.y_train)
      J_hist.append(J_wb)
      MAPE_hist.append(MAPE)

      end_time = time.time()  # End time for the epoch
      epoch_time.append(end_time - start_time)

      # Print the weight vector after the last epoch
    print("The resulting weight vector ŵ from the training set:", self.w)

        # Compute MSE and MAPE on the training and test sets after the last epoch
    final_mse_train, final_mape_train = self.mse_mape_cost(self.X_train, self.y_train)

        # Print the final MSE and MAPE for both training and test sets
    print("The resulting MSE on the training set after the last epoch:", final_mse_train)
    print("The resulting MAPE on the training set after the last epoch:", final_mape_train)

        # Determine the number of epochs to reach MSE < 85 on the training set
    epochs_to_85 = next((i for i, mse in enumerate(J_hist) if mse < 85), None)
    if epochs_to_85 is not None:
        print("Number of epochs to reach MSE < 85 on the training set:", epochs_to_85 + 1)
    else:
        print("MSE < 85 was not reached during the training.")

        # Print the total computation time to the last epoch
    total_time = sum(epoch_time)
    print("The total computation time to the last epoch:", total_time)

        # If MSE < 85 was reached, print the computation time up to that point
    if epochs_to_85 is not None:
        time_to_85 = sum(epoch_time[:epochs_to_85+1])
        print("The computation time to reach MSE < 85 on the training set:", time_to_85)
    else:
        print("The computation time to reach MSE < 85 was not applicable since the condition was not met.")


    return J_hist, MAPE_hist




  def _compute_gradient_per_sample(self,x,y):
    """
    Computes the single-sample gradient for linear regression

    Args:
      x (ndarray (D+1,)): augmented data, a sample with D+1 dimension
      y (scaler) : target values
        w (ndarray (D+1,)) : model parameters (using self.w)

    Returns:
      dj_dw (ndarray (n,)): the gradient of the cost w.r.t. the parameters w.

      
    """
    prediction = np.dot(self.w, x)  # Compute the model's prediction
    error = prediction - y  # Compute the prediction error
    dj_dw = error * x   # Compute the gradient of the cost w.r.t. the parameters w
    return dj_dw
   


  def mse_mape_cost(self,X, y):
    """
    Compute mean square and MAPE loss functions

    Args:
      X (ndarray (N,D+1)): augmented data, m samples and each sample has dimension D+1
      y (ndarray (N,)) : target values
        w (ndarray (D+1,)) : model parameters (using self.w)

    Returns:
      mse_loss (scalar): cost
      MAPE (scalar): mean absolute percentage error
    """

    #########################################
    ###############Your code here###########
    #########################################

    predictions = np.dot(X, self.w)  # Compute predictions for all samples
    errors = predictions - y  # Compute errors
    mse_loss = np.mean(errors ** 2)  # Compute MSE
    mape = np.mean(np.abs(errors / y)) * 100  # Compute MAPE
    return mse_loss, mape


  def predict(self,X_test):
    """
      Compute output prediction for each data point in X_test
      Args:
        X_test (ndarray (M,D+1)): augmented data, M examples and each sample has dimension D+1
      Returns:
        y_pred (ndarray (M,)): prediction
    """
    #########################################
    ###############Your code here###########
    #########################################
    y_pred = np.dot(X_test, self.w)
    return y_pred



learning_rates = [10**-3, 5*10**-3, 10**-2, 10**-1, 1]

for lr in learning_rates:
    print(f"\nTraining with learning rate: {lr}")
    model = MultiVariate_Linear_Regression(X_train_aug, y_train)
    J_hist, MAPE_hist = model.fit_train(epochs=40, learning_rate=lr, batch_size=20)

    # After training, you might want to plot or print results specific to this learning rate
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(J_hist, label='MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training MSE Loss (lr={lr})')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(MAPE_hist, label='MAPE')
    plt.xlabel('Epoch')
    plt.ylabel('MAPE (%)')
    plt.title(f'Training MAPE (lr={lr})')
    plt.legend()

    plt.show()

    # Predicting on the test set with the current model
    y_pred = model.predict(X_test_aug)

    # Calculate the MSE and MAPE for the test set predictions
    mse_test = np.mean((y_test - y_pred) ** 2)
    mape_test = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Print out the MSE and MAPE for the test set
    print(f"Test MSE for learning rate {lr}: {mse_test:.3f}")
    print(f"Test MAPE for learning rate {lr}: {mape_test:.2f}%")

