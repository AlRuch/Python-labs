

import numpy as np

#2.3#############################################################################

matrix=np.array([[0,1,2,3,4,5],
           [10,11,12,13,14,15],
           [20,21,22,23,24,25],
           [30,31,32,33,34,35],
           [40,41,42,43,44,45],
           [50,51,52,53,54,55]])

q=np.copy(matrix)
#[[ 0  1  2  3  4  5]
# [10 11 12 13 14 15]
# [20 21 22 23 24 25]
# [30 31 32 33 34 35]
# [40 41 42 43 44 45]
# [50 51 52 53 54 55]]

r=matrix.copy()
#[[ 0  1  2  3  4  5]
# [10 11 12 13 14 15]
# [20 21 22 23 24 25]
# [30 31 32 33 34 35]
# [40 41 42 43 44 45]
# [50 51 52 53 54 55]]

s=matrix.view()
#[[ 0  1  2  3  4  5]
# [10 11 12 13 14 15]
# [20 21 22 23 24 25]
# [30 31 32 33 34 35]
# [40 41 42 43 44 45]
# [50 51 52 53 54 55]]


qq=np.array([1,2,3,4,7,3,9,7])
t=qq.sort()


#2.3.1

a=matrix[1,0]
#10

b=matrix[0] == 42 
#[False False False False False False]

c=matrix[1:3]
#[[10 11 12 13 14 15]
# [20 21 22 23 24 25]]

#d=matrix[]

e=matrix[1:]
#[[10 11 12 13 14 15]
# [20 21 22 23 24 25]
# [30 31 32 33 34 35]
# [40 41 42 43 44 45]
# [50 51 52 53 54 55]]

f=matrix[1:100]
#[[10 11 12 13 14 15]
# [20 21 22 23 24 25]
# [30 31 32 33 34 35]
# [40 41 42 43 44 45]
# [50 51 52 53 54 55]]

g=matrix[:]
#[[ 0  1  2  3  4  5]
# [10 11 12 13 14 15]
# [20 21 22 23 24 25]
# [30 31 32 33 34 35]
# [40 41 42 43 44 45]
# [50 51 52 53 54 55]]

h=matrix[1: , :2]
#[[10 11]
# [20 21]
# [30 31]
# [40 41]
# [50 51]]

i=matrix[:2, 1:]
#[[ 1  2  3  4  5]
# [11 12 13 14 15]]

j=matrix.ravel ()
#[ 0  1  2  3  4  5 10 11 12 13 14 15 20 21 22 23 24 25 30 31 32 33 34 35
# 40 41 42 43 44 45 50 51 52 53 54 55]

k=matrix[: ,1].copy()
#[ 1 11 21 31 41 51]

l=matrix[1].tolist()
#[10, 11, 12, 13, 14, 15]

m=matrix.reshape(-1)
#[ 0  1  2  3  4  5 10 11 12 13 14 15 20 21 22 23 24 25 30 31 32 33 34 35
# 40 41 42 43 44 45 50 51 52 53 54 55]


#2.4#######################################################################################

aa=np.sqrt(matrix)
#[[0.         1.         1.41421356 1.73205081 2.         2.23606798]
# [3.16227766 3.31662479 3.46410162 3.60555128 3.74165739 3.87298335]
# [4.47213595 4.58257569 4.69041576 4.79583152 4.89897949 5.        ]
# [5.47722558 5.56776436 5.65685425 5.74456265 5.83095189 5.91607978]
# [6.32455532 6.40312424 6.4807407  6.55743852 6.63324958 6.70820393]
# [7.07106781 7.14142843 7.21110255 7.28010989 7.34846923 7.41619849]]

ab=np.exp(matrix)
#[[1.00000000e+00 2.71828183e+00 7.38905610e+00 2.00855369e+01
#  5.45981500e+01 1.48413159e+02]
# [2.20264658e+04 5.98741417e+04 1.62754791e+05 4.42413392e+05
#  1.20260428e+06 3.26901737e+06]
# [4.85165195e+08 1.31881573e+09 3.58491285e+09 9.74480345e+09
#  2.64891221e+10 7.20048993e+10]
# [1.06864746e+13 2.90488497e+13 7.89629602e+13 2.14643580e+14
#  5.83461743e+14 1.58601345e+15]
# [2.35385267e+17 6.39843494e+17 1.73927494e+18 4.72783947e+18
#  1.28516001e+19 3.49342711e+19]
# [5.18470553e+21 1.40934908e+22 3.83100800e+22 1.04137594e+23
#  2.83075330e+23 7.69478527e+23]]

ac=np.min(matrix)
#0

ad=np.max(matrix, axis=1)
#[ 5 15 25 35 45 55]

ae=np.min(np.maximum(np.random.randn(4),np.random.randn(4)))
# -0.35770057484762907

af=np.mean(matrix)
# 27.5

ag=np.sum(matrix)
# 990

ah=np.invert(matrix)
#[[ -1  -2  -3  -4  -5  -6]
# [-11 -12 -13 -14 -15 -16]
# [-21 -22 -23 -24 -25 -26]
# [-31 -32 -33 -34 -35 -36]
# [-41 -42 -43 -44 -45 -46]
# [-51 -52 -53 -54 -55 -56]]

ai=np.random.randn(5)
#[-2.50666065 -0.91281764 -1.33719724  0.99645734 -0.43676448]

aj=np.trace(matrix)
# 165



############################################################################


#ar=np.random.randn(5)
print(ah)

#def randomwalker():
    
   # start=np.random.randint(2)

    #print(start)

 #   for x in range(0, 500)
        
    #return

#randomwalker()






