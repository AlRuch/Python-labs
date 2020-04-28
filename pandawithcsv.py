import pandas as pd;
import numpy as np;
import random;
import matplotlib.pyplot as plt;



dff=pd.read_csv('sampleDataSet.csv');
#    5.1  0.222222222       3.5  ...   0.2  0.041666667     setosa
#0   4.9     0.166667  3.000000  ...  0.20  0.041666667     setosa
#1   4.7     0.111111  3.200000  ...  0.20  0.041666667     setosa
#2   4.6     0.083333  3.100000  ...  0.20  0.041666667     setosa
#3   NaN     0.194444  3.600000  ...  0.20  0.041666667     setosa
#4   NaN     0.305556  3.900000  ...  0.40        0.125     setosa
#..  ...          ...       ...  ...   ...          ...        ...
#94  7.2     0.805556  3.000000  ...  1.60        0.625  virginica
#95  7.4          NaN  0.333333  ...  0.75    virginica        NaN
#96  7.9     0.999900  3.800000  ...  2.00  0.791666667  virginica
#97  6.4     0.583333  2.800000  ...  2.20        0.875  virginica
#98  6.3     0.555556  2.800000  ...  1.50  0.583333333  virginica

df=pd.read_csv('sampleDataSet.csv',names=["a","b","c","d","e","f","g","h","i"] );
#      a         b         c         d  ...         f     g            h          i
#0   5.1  0.222222  3.500000  0.625000  ...  0.067797  0.20  0.041666667     setosa
#1   4.9  0.166667  3.000000  0.416667  ...  0.067797  0.20  0.041666667     setosa
#2   4.7  0.111111  3.200000  0.500000  ...       NaN  0.20  0.041666667     setosa
#3   4.6  0.083333  3.100000  0.458333  ...  0.084746  0.20  0.041666667     setosa
#4   NaN  0.194444  3.600000  0.666667  ...       NaN  0.20  0.041666667     setosa
#..  ...       ...       ...       ...  ...       ...   ...          ...        ...
#95  7.2  0.805556  3.000000  0.416667  ...  0.813559  1.60        0.625  virginica
#96  7.4       NaN  0.333333  6.100000  ...  1.900000  0.75    virginica        NaN
#97  7.9  0.999900  3.800000  0.750000  ...  0.915254  2.00  0.791666667  virginica
#98  6.4  0.583333  2.800000  0.333333  ...  0.779661  2.20        0.875  virginica
#99  6.3  0.555556  2.800000  0.333333  ...  0.694915  1.50  0.583333333  virginica

#[100 rows x 9 columns]

#So without the setting names it get the first woe as the column names so it dont show the
#actual data set information.It shows 99 rows but its actually 100 rows.It happened because
#without setting names it lost the first row of data.
x=df['a'];
plt(x,100);

a=df.isnull().g
#0     False
#1     False
#2     False
#3     False
#4     False
#      ...  
#95    False
#96    False
#97    False
#98    False
#99    False

#Name: g, Length: 100, dtype: bool

b=df.isnull().sum(0)
#a    4
#b    1
#c    0
#d    3
#e    2
#f    2
#g    1
#h    1
#i    1
#dtype: int64

c=df=df[df.isnull().a != True]
#      a         b         c         d  ...         f     g            h          i
#0   5.1  0.222222  3.500000  0.625000  ...  0.067797  0.20  0.041666667     setosa
#1   4.9  0.166667  3.000000  0.416667  ...  0.067797  0.20  0.041666667     setosa
#2   4.7  0.111111  3.200000  0.500000  ...       NaN  0.20  0.041666667     setosa
#3   4.6  0.083333  3.100000  0.458333  ...  0.084746  0.20  0.041666667     setosa
#7   5.0  0.194444  3.400000       NaN  ...  0.084746  0.20  0.041666667     setosa
#..  ...       ...       ...       ...  ...       ...   ...          ...        ...
#95  7.2  0.805556  3.000000  0.416667  ...  0.813559  1.60        0.625  virginica
#96  7.4       NaN  0.333333  6.100000  ...  1.900000  0.75    virginica        NaN
#97  7.9  0.999900  3.800000  0.750000  ...  0.915254  2.00  0.791666667  virginica
#98  6.4  0.583333  2.800000  0.333333  ...  0.779661  2.20        0.875  virginica
#99  6.3  0.555556  2.800000  0.333333  ...  0.694915  1.50  0.583333333  virginica

#[96 rows x 9 columns]

d=df.dropna(axis=0).isnull().sum()
e=df.dropna(axis=1)
f=df.dropna(axis=1, how='all')
g=df.dropna(axis=1, thresh=1)
h=df.drop('i',axis=1)
i=df.fillna(899)
j=df.fillna(method='ffill')
k=df.replace(6.3,600)
l=df.replace('.',np.nan)
#m=df[np.random.rand(df.shape[0]>0.5)]=1.5

print('\n')
print(a)
print('\n')
print(b)
print('\n')
print(c)
print('\n')
print(d)
print('\n')
print(e)
print('\n')
print(f)
print('\n')
print(g)
print('\n')
print(h)
print('\n')
print(i)
print('\n')
print(j)
print('\n')
print(k)
print('\n')
print(l)
print('\n')
print(m)
print('\n')



print(dff)
