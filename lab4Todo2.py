#import matplot library
import matplotlib.pyplot as plt
#importing modules for 3D plotting
from mpl_toolkits import mplot3d 

#creating a figure
fig = plt.figure()

#creating 3D subplot
ax = fig.add_subplot(111, projection='3d') 

xs=([29, 24, 25, 23, 30 ,31, 26, 26, 30, 28])
ys=([ 7, 53 , 33 , 66, 1 ,11, 91, 51, 83, 6])
zs=([-25, -25, -19, -23,-6, -9, -11 , -11,-5, 14])

ax.scatter(xs, ys, zs, c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.view_init(azim=0, elev=90)

plt.show()
