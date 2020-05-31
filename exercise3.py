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

#devide a 20% test set and training set
x_train, X_test = train_test_split(X, test_size=0.2,random_state=1)
y_train, y_test = train_test_split(Y, test_size=0.2,random_state=1)


matrix=np.full ([len(y_test),1] , 0)

#loop for the 50 samples 
for i in range(50):

    #Creating samples with n=100
    xx=x_train.sample(n=100, random_state= i)
    yy=y_train.sample(n=100, random_state= i)

    #making the arrays
    xx_matrix=np.array(xx)
    yy_matrix=np.array(yy)

    #make the array with 1
    n=len(xx_matrix)
    a=np.full ([n,1] , 1)
    XX=np.c_[ a,xx_matrix ]

    #transpose
    Xt=XX.T

    #test set
    data_test=np.array(X_test)
    n1=len(data_test)
    a1=np.full ([n1,1] , 1)
    xx_test=np.c_[ a1,data_test ]
    


    # Y set for the train set
    u_train=np.array(yy).reshape(-1,1)

    #Y set for the test set
    u_test=np.array(y_test).reshape(-1,1)


    # 1sr matrix multiplication
    result_1 =  np.dot(Xt,XX)

    #Matrix inverse
    x_inv=np.linalg.inv(result_1)

    # 2nd matrix multiplication
    result_2 =np.dot(x_inv,Xt)

    #3rd matrix multiplication Get the (Theta)
    result_f =np.dot(result_2,u_train)

    # Predict the results(test)
    y_pred =np.dot(xx_test,result_f)
 
    #addition of the samples 
    matrix=matrix+y_pred

#end of loop and getting the average of the matrix
final_test=matrix/50

# x-axis values 
x =final_test

ytest=np.array(y_test)
# y-axis values 
y =ytest

# plotting points as a scatter plot test set
plt.scatter(x, y, label= "test set", color= "green",  
            marker= "*")

x1=y1=[0,800000]
plt.plot(x1, y1)

# x-axis label 
plt.xlabel('predicted values') 
# frequency label 
plt.ylabel('actual values') 
# plot title
plt.title('Bostan_Housing Ensemble Learning') 
# showing legend 
plt.legend() 
  
# function to show the plot 
plt.show() 
    

    
