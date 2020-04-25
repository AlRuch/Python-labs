import pandas as pd
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics



df=pd.read_csv('Book.csv',names=['Carname','Colour','Age','Speed','Auto Pass'] );
'''print(df);'''

d = {'BMW': 0, 'Volvo': 1, 'VW': 2,'Ford': 3,'Tesla': 4,'Toyota': 5}
df['Carname'] = df['Carname'].map(d)

e = {'red': 0, 'black': 1, 'gray': 2,'white': 3,'blue': 4}
df['Colour'] = df['Colour'].map(e)

f = {'Y': 0, 'N': 1}
df['Auto Pass'] = df['Auto Pass'].map(f)

features = ['Carname','Colour','Age','Speed']
X = df[features]
y = df['Auto Pass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print(y_pred)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
