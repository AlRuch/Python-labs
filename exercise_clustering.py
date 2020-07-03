from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import pandas as pd


iris = datasets.load_iris()

x=iris.data[:,:]


#within cluster sum of squares
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#from Elbow method we identified n_clusters=3

#K Means algorithem
kmeans = KMeans(n_clusters=3, random_state=0) 
closest_cluster_index = kmeans.fit_predict(x)
cluster_centers=kmeans.cluster_centers_

#kmeans.cluster_centers_  returns the coordinates of the centers of the clusters

df = pd.DataFrame(data = x, columns = ['Variable 1', 'Variable 2','Variable 3','Variable 4'])

Y=closest_cluster_index.reshape(150,1)
df_target = pd.DataFrame(data = Y,columns = ['Target'])

a1 = {0:'cluster_1',1:'cluster_2',2:'cluster_3'}
df_target['Target'] = df_target['Target'].map(a1)

#Concat the varables and targets to a single table 
final_df = pd.concat([df,df_target], axis = 1)

#Define the targer classes
targets = ['cluster_1','cluster_2','cluster_3']
#Define the colurs of the target classes
colors = ['r', 'b', 'g']

#creating a figure
fig = plt.figure()
#creating 3D subplot
ax = fig.add_subplot(111, projection='3d') 

#Label the axises in the plot
ax.set_xlabel('Variable 1')
ax.set_ylabel('Variable 2')
ax.set_zlabel('Variable 3')

#Plot the values witth colurin the classes according to its colour index 
for Target,color in zip(targets,colors):
    indicesToKeep = final_df['Target'] == Target
    ax.scatter(final_df.loc[indicesToKeep, 'Variable 1']
               , final_df.loc[indicesToKeep, 'Variable 2']
               , final_df.loc[indicesToKeep, 'Variable 3']
               , c = color
               )
    
ax.legend(targets)
ax.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,2],s=200,c='black',marker='*')
plt.show()
