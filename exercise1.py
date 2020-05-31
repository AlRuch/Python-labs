import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#import the data set
df = pd.read_csv("Boston_Housing.csv")

#select features and Response
features = ["RM","LSTAT","PTRATIO"]
X = df[features]
Y = df["MEDV"]

#devide a 80% test set and training set
x_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=1)

#Get the X(train) matrix
data=np.array(x_train)
n=len(data)
a=np.full ([n,1] , 1)
xx_train=np.c_[ a,data ]

xt=xx_train.T

#Get the X(test) matrix
data_test=np.array(X_test)
n1=len(data_test)
a1=np.full ([n1,1] , 1)
xx_test=np.c_[ a1,data_test ]

# Y set for the train set
u_train=np.array(y_train).reshape(-1,1)

#Y set for the test set
u_test=np.array(y_test).reshape(-1,1)




# 1sr matrix multiplication
result_1 =  np.dot(xt,xx_train)

#Matrix inverting
x_inv=np.linalg.inv(result_1)

# 2nd matrix multiplication
result_2 =np.dot(x_inv,xt)

#3rd matrix multiplication Get the (Theta)
result_f =np.dot(result_2,u_train)


# Predict the results(test)
y_pred =np.dot(xx_test,result_f)

# Predict the results(train)
y_pred2 =np.dot(xx_train,result_f)


ytest=np.array(y_test)
ytrain=np.array(y_train)

# x-axis values 
x =y_pred
x1=y_pred2

# y-axis values 
y =ytest
y1=ytrain

# plotting points as a scatter plot test set
plt.scatter(x, y, label= "test set", color= "green",  
            marker= "*")
# plotting points as a scatter plot train set
plt.scatter(x1, y1, label= "train set", color= "red",  
            marker= ".")

x2=y2=[0,800000]
plt.plot(x2, y2)

# x-axis label 
plt.xlabel('predicted values') 
# y-axis label 
plt.ylabel('actual values') 
# plot title 
plt.title('Bostan_Housing') 
# showing legend 
plt.legend() 
  
# function to show the plot 
plt.show() 
