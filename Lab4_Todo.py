#import matplot library
import matplotlib.pyplot as plt
#importing modules for 3D plotting
from mpl_toolkits import mplot3d
import numpy as np

t = np.arange(0., 5., 0.2)
fig = plt.figure(figsize =(10 , 10)) #creating a figure
fig.subplots_adjust(hspace =1.0)
axes_1 = plt.subplot (4,1, 1) #first axes in the figure
plt.plot(t, t,'r^',markersize=8,label='line1') #plotting with red marker '^'
legend = plt.legend(loc='upper right', shadow=True,fontsize='x-large') #adding the legend
plt.title('First Plot') #adding the title
plt.xlabel('t') #labeling x axis
plt.ylabel('t') #labeling y axis
plt.xlim([0,10]) #limits of x axis
axes_2 = plt.subplot (4,1,2) #second axes in the figure
plt.plot(t, t**2, 'b*',markersize=8) #plotting
axes_2.set_title('Second Plot') #adding the title
axes_2.set_xlabel('t') #labeling x axis
axes_2.set_ylabel('t squred') #labeling y axis
axes_2.set_ylim([0 ,40]) #limits of y axis

plt.show()
