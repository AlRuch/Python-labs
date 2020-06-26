#import standard data sets
from sklearn import datasets
#split data set into a train and test set
from sklearn.model_selection import train_test_split
#import matplot library
import matplotlib.pyplot as plt   
from mpl_toolkits import mplot3d
#immport numpy
import numpy as np
#immport pandas
import pandas as pd
#immport Scaler
from sklearn.preprocessing import StandardScaler
#Import PCA
from sklearn.decomposition import PCA

dataset =datasets.load_wine() #load 'Wine' data set from standard data sets
x=dataset["data"] #defining features values
y =dataset["target"] #defining target variable values

#Standardized the data
X = StandardScaler().fit_transform(x)

#PCA transformation
pca = PCA(n_components=3)
components = pca.fit_transform(X)
df = pd.DataFrame(data = components
             , columns = ['component 1', 'component 2','component 3'])

Y=y.reshape(178,1)
principal = pd.DataFrame(data = Y
             , columns = ['target'])


fig = plt.figure() #creating a figure
ax = fig.add_subplot(111, projection='3d') #creating 3D subplot

#Label the axises in the plot
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

#concat the features columns with the targecolumn to get a single table
final_df = pd.concat([df,principal], axis = 1)

#Map the target numerical values with the classes
a1 = {0:'class_0',1:'class_1',2:'class_2'}
final_df['target'] = final_df['target'].map(a1)

#Define the targer classes
targets = ['class_0','class_1','class_2']
#Define the colurs of the target classes
colors = ['r', 'g', 'b']

#Plot the values witth colurin the classes according to its colour index 
for target,color in zip(targets,colors):
    indicesToKeep = final_df['target'] == target
    ax.scatter(final_df.loc[indicesToKeep, 'component 1']
               , final_df.loc[indicesToKeep, 'component 2']
               , final_df.loc[indicesToKeep, 'component 3']
               , c = color
               )
    
ax.legend(targets)
ax.grid()

plt.show()
















