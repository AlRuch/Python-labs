from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import numpy as np

#Import the data set 
df = pd.read_csv('grocc.csv', header=None)
# create an array with the data 
x=np.array(df)
# make a list of data
a=x.tolist()

#Remove the Nan values from the dataset
dataset  = [[y for y in k if y == y] for k in a  ]

#Create the frequent item dataset
te1 = TransactionEncoder()
te_ary = te1.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te1.columns_)

#Item set with minimum support greater than 10%(It dont get any rules with 10% So i used 0.05(5%))
sup = apriori(df, min_support=0.05, use_colnames=True)

#The set of Association rules using the metric 'lift'
rules= association_rules(sup, metric="lift", min_threshold=1)


#I take the rule no 4 
#4        (whole milk)            (yogurt)  ...  0.020379    1.102157
#From this rule it shoes the association with the Whole milk after yogurt.like
#buying whole milk first and then buy yogurt
#Independally whole milk have a support of 0.255516(25.51%) and yogurt have 0.139502(13.95%)
#It describes an indication of how frequently the item or itemcombination bought.For the both whole milk
#and yogurt have a support of 0.056024(5.64%).The confidence of buying according to
#the above scenario is 0.219260 (21.92%).Which means the number of times the the
#statements are found true.It also describes how frequently the rule head occurs among all the groups containing the rule body
#which is a indicates the reliability of the rule .The rule have a lift of 1.571735.
#Which can be used to compare confidence with expected confidence


#There are no rules when the ’lift’ is greater than 4 and the ’confidence’ is greater than 0.8
