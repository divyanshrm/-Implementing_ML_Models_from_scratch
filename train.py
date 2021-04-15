#!/usr/bin/env python
# coding: utf-8

# ## SEMINAR IMPLEMENTATION
# ### Name and Roll no-Divyansh Singh,18ucs127
# ### Name and Roll no-Lakshay Bhagtani,18ucs132

# Note- This pdf is searchable and is generated using pyppeteer. ipynb to pdf converter

# In[4]:


import numpy as np  
import collections


# In[5]:


#HERE WE ARE LOADING THE DATASET FROM RAW TEXT FILES USING NUMPY
x_train=np.loadtxt('p1_a_X.dat')
y_train=np.loadtxt('p1_a_y.dat')
x_test=np.loadtxt('p1_b_X.dat')
y_test=np.loadtxt('p1_b_y.dat')


# In[6]:


#getting the minimum and maximum elements for each feature in the dataset.
max_0=max(np.hsplit(x_train,2)[0])
max_1=max(np.hsplit(x_train,2)[1])


# In[7]:


min_1=min(np.hsplit(x_train,2)[1])
min_0=min(np.hsplit(x_train,2)[0])


# In[8]:


#This normalizer performs MIN-MAX scaling on the dataset.

def normalizer(data,min_x,min_y,range_x,range_y):
    x=np.hsplit(data,2)[0]
    y=np.hsplit(data,2)[1]
    x=(x-min_x)/range_x
    y=(y-min_y)/range_y
    return np.hstack((x,y))
    


# In[9]:


#normalizing the dataset
x_train=normalizer(x_train,min_0,min_1,max_0-min_0,max_1-min_1)
x_test=normalizer(x_test,min_0,min_1,max_0-min_0,max_1-min_1)


# In[10]:


#THIS FUNCTION CONVERTS THE CLASSES INTO 0 and 1,
def label_binarizer(data):
    for x in range(len(data)):
        if data[x]==1:
            data[x]=1
        if data[x]==-1:
            data[x]=0
    return data
#we are now binarizing our training and testing labels
y_train=label_binarizer(y_train)
y_test=label_binarizer(y_test)


# ### This function calculates the performance metrics and returns Accuracy, Precision and Recall

# In[11]:


#This funciton returns the Accuracy, Precision and recall from the predicted and true labels.
def metrics(y_true,predicted):
    true_positives=0
    correct=0
    false_positives=0
    false_negatives=0
    for x in range(len(y_true)):
        if y_true[x]==predicted[x]:
            correct+=1.0
        if y_true[x]==1 and predicted[x]==1:
            true_positives+=1.0
        if y_true[x]==0 and predicted[x]==1:
            false_positives+=1.0
        if y_true[x]==1 and predicted[x]==0:
            false_negatives+=1.0
    accuracy=(correct*100)/len(y_true)
    preci=(true_positives)/(true_positives+false_positives)
    recall=(true_positives)/(true_positives+false_negatives)
    print('Accuracy is '+str(accuracy)+' ,Precision is '+str(preci)+' ,Recall is '+str(recall))


# # LOGISTIC REGRESSION

# We will Implement Logistic Regression below.

# This function performs the sigmoid activation on an array/scalar

# In[12]:


def sigmoid(x):
    s = 1.0/(1.0 + np.exp(-1.0 * x))
    return s


# We will initialize the weights and then use gradient descent to minimize the logistic loss function

# In[13]:


def Logistic_regression_train(X, y):
#Here we initialize the learning rate as 0.1 and epochs to 10000
    learning_rate=0.1
    epochs=10000
#we are initializing the theta and bias, the number of theta would be 2 as we have 2 dimensional input and 1 bias weight
    theta = np.zeros(2)
    b=0
    #total number of examples in triaining
    m=X.shape[0]
   
    #This function returns the activation given the inputs
    def forward(theta,b, inputs):
        #This code applies matrix dot product to theta and input and adds the bias term
        Z= np.dot(inputs,theta)+b
        A = sigmoid(Z)
        return A
       
#Here we are training the model and initializeing the for loop to run for the number of epochs and updates as 0
    
    for i in range(epochs):
        #First we compute the predicted output from the input 
        A = forward(theta,b,X)
#we calculate the cost
        cost =(-1.0)*np.mean(np.multiply(y, np.log(A))+np.multiply(1-y, np.log(1- A)))
#we calculate the gradient for each data point
        dw = np.dot(X.T, (A - y)) * (1.0/m)
        db = np.mean((A - y))
#Then we update the theta with the gradient
        theta = theta-learning_rate * dw
        b = b-learning_rate * db
#printing loss every 100 epochs
        if (i)%1000==0:
            print('The Loss at epoch '+str(i)+' is '+str(cost))
    print('Final Loss is '+ str(cost))
    return theta,b


# This function will perform prediction on the test data by using the weights.

# In[14]:


def logistic_test(theta,b,x_test):
#This function predicts the labels using the weights
    prediction=[]
    Z= np.matmul(x_test, theta) + b
    A = sigmoid(Z)
    y_predicted=np.asarray(A)
    for x in A:
        if x>=0.5:
            prediction.append(1.0)
        else:
            prediction.append(0.)
    return np.asarray(prediction)


# We will now use the above functions to train our model on the train data

# In[15]:


#We train our perceptron and then then initialise the theta for testing the model.
theta,b=Logistic_regression_train(x_train,y_train)
print('The theta are '+ str(theta)+' and bias is '+str(b))


# We will now evaluate the performance of the model on test data.

# In[16]:


print('Performance of Logistic Regression Model on the Test Data: ')
metrics(y_train,logistic_test(theta,b,x_train))


# # KNN MODEL
# 

# We will now implement KNN algorithm.

# In[17]:


def KNN_implementation(x_train,labels,test_data,k):    
#function to calculate the distance between two points
    def calculate_distance(point1,point2):
        y2=point2[1]
        y1=point1[1]
        x2=point2[0]
        x1=point1[0]
        return pow(pow(y2-y1,2)+pow(x2-x1,2),1/2)
#this function returns the distances of all the elements in a vector from  a given point
    def get_distances(point,data):
        distance=[]
        for x in data:
            s=calculate_distance(point,x)
            distance.append(s)
        return np.asarray(distance)
#this function sorts the distances and returns the labels of nearest data points
    def get_nearest_points(point,data,labels,nearest_points=1):
        distances=get_distances(point,data)
        sorted_index=np.argsort(distances) 
        return labels[sorted_index[:nearest_points]]
#returns the frequency distribution of 0s and 1s in the array
    def counter(data):
        counter=[0,0]
        for x in data:
            if x==1:
                counter[1] +=1
            if x==0:
                counter[0] +=1 
        return counter
#this function predicts the values based on counter and breaks the ties if there are any.
    def KNN(test_data,training_data,labels,k):
        predictions=[]
        for x in test_data:
            nearest_point_labels=get_nearest_points(x,training_data,labels,k)
            counter_prediction=counter(nearest_point_labels)
            #In case of a tie we break the tie by searching for a lower k value
            while(counter_prediction[0]==counter_prediction[1]):
                nearest_point_labels=get_nearest_points(x,training_data,labels,k-1)
                counter_prediction=counter(nearest_point_labels)
            if counter_prediction[0]>counter_prediction[1]:
                predictions.append(0)
            else:
                predictions.append(1)
        return np.asarray(predictions)
            
    return(KNN(x_test,x_train,y_train,k))


# We will now evaluate the performance of the model on test data.

# In[18]:


print('Performance of KNN model on the Test Data: ')
for x in range(1,5):
    print('The Metrics of performance on the test set for K value:'+str(x)+ ' are: ')
    metrics(y_test,KNN_implementation(x_train,y_train,x_test,x))


# # NEURAL NETWORK

# We will now implement an Artificial Neural Network
# This model will have only 1 hidden layer to maintain simplicity.
# The size of this layer is variable.

# In[19]:


#We will initialize the weights as random numbers to avoid explosion of weights.
def parameters_initialize(hidden_layer):
    w1= np.random.randn(hidden_layer,2)*0.01
    b1 = np.zeros((hidden_layer,1))
    w2 = np.random.randn(1,hidden_layer)*0.01
    b2 = np.zeros((1,1))
    return w1,b1,w2,b2
#This function returns the activations by forward propagation in the network
def forward(X,w1,b1,w2,b2):
    z1=np.dot(w1,X.T)
    z1=z1+b1
    a1=np.tanh(z1)
    z2=np.dot(w2,a1)
    z2=z2+b2
    a2=sigmoid(z2)
    return z1,a1,z2,a2


#this function calculates the cost.
def compute_cost(a2,y):
    cost = (-1.0) * np.mean(np.multiply(y.T, np.log(a2)) + np.multiply(1-y.T, np.log(1- a2)))
    return cost
#this function calculates the gradients for each weight and then updates them given the learning rate
def back_prop_and_update(w1,b1,w2,b2,X,y,learning_rate):
    z1,a1,z2,a2=forward(X,w1,b1,w2,b2)
    m=X.shape[0]
    
    dz2 = a2-y.T
    dw2 = 1/m*np.dot(dz2, a1.T)
    db2 = 1/m*np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.dot(w2.T, dz2)*(1 - np.power(a1, 2))
    dw1 = 1/m*np.dot(dz1, X)
    db1 = 1/m*np.sum(dz1, axis=1, keepdims=True)
#updating the parameters
    w1=w1-learning_rate*dw1
    w2=w2-learning_rate*dw2
    b1=b1-learning_rate*db1
    b2=b2-learning_rate*db2
    return w1,w2,b1,b2
    
#This function performs forward and backprop till the number of epochs        

def nn_model(X,y,hidden_layer,learning_rate=0.1,epochs=1000):
    w1,b1,w2,b2=parameters_initialize(hidden_layer)
    for x in range(epochs):
        w1,w2,b1,b2=back_prop_and_update(w1,b1,w2,b2,X,y,learning_rate)
#printing loss every 100 epochs
        if(x%100==0):
            _,_,_,a2=forward(X,w1,b1,w2,b2)
            print('The Loss in Epoch- '+str(x)+ ' is '+str(compute_cost(a2,y)))
    _,_,_,a2=forward(X,w1,b1,w2,b2)
    print('Final loss is '+ str(compute_cost(a2,y)))
    return w1,b1,w2,b2

#this function performs predictions given the weights and test data

def nn_predict(x_test,w1,w2,b1,b2):
    predictions=[]
    _,_,_,a2=forward(x_test,w1,b1,w2,b2)
    a2=a2.T
    for x in a2:
        if(x>=0.5):
            predictions.append(1)
        else:
            predictions.append(0)
    return np.asarray(predictions)
    


# In[20]:


w1,b1,w2,b2=nn_model(x_train,y_train,9)


# In[21]:


print('Performance of Neural Network Model on the Test Data: ')
metrics(y_test,nn_predict(x_test,w1,w2,b1,b2))






