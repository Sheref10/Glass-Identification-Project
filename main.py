import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas_profiling as pp
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


matplotlib.use( 'tkagg' )
#Data preparation
data=pd.read_csv('glass.csv')
X=data.iloc[:,:-1]
y=data.iloc[:,-1]

#Visualization

plt.figure(figsize=(40,25))

plt.subplot(3,3,1)
sns.histplot(data['RI'], color = 'red', kde = True).set_title('RI Interval and Counts')

plt.subplot(3,3,2)
sns.histplot(data['Na'], color = 'green', kde = True).set_title('Na Interval and Counts')

plt.subplot(3,3,3)
sns.histplot(data['Mg'], kde = True, color = 'blue').set_title('Mg Interval and Counts')

plt.subplot(3,3,4)
sns.histplot(data['Al'], kde = True, color = 'black').set_title('Al Interval and Counts')

plt.subplot(3,3,5)
sns.histplot(data['Si'], kde = True, color = 'yellow').set_title('Si Interval and Counts')

plt.subplot(3,3,6)
sns.histplot(data['K'], kde = True, color = 'orange').set_title('K Interval and Counts')

plt.subplot(3,3,7)
sns.histplot(data['Ca'], kde = True, color = 'brown').set_title('Ca Interval and Counts')

plt.subplot(3,3,8)
sns.histplot(data['Ba'], kde = True, color = 'cyan').set_title('Ba Interval and Counts')

plt.subplot(3,3,9)
sns.histplot(data['Fe'], kde = True, color = 'purple').set_title('Fe Interval and Counts')
#plt.show()
#Distribution
plt.figure(2)
plt.title("Distribution of Type")
data['Type'].value_counts().plot.pie(autopct="%1.1f%%")

plt.figure(3, figsize=(5,5))
sns.countplot(x=data['Type'])
#plt.show()
#ProfileReport
#profile = pp.ProfileReport(data)
#profile.to_file("output.html")

#Model

#Naive_Bayes
xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=4)

MNB=MultinomialNB()
MNB.fit(xtrain,ytrain)
ypre=MNB.predict(xtest)
print(" MNaive_Bayes accuracy is ={:.2%}".format(accuracy_score(ytest,ypre)))
#print("MNaive_Bayes :",classification_report(ytest,ypre))
GNB=GaussianNB()
GNB.fit(xtrain,ytrain)
ypre=GNB.predict(xtest)
print(" GNaive_Bayes accuracy is ={:.2%}".format(accuracy_score(ytest,ypre)))
#print("GNaive_Bayes :",classification_report(ytest,ypre))

#LogisticRegression
scale=StandardScaler()
scale_xtrain=scale.fit_transform(xtrain)
scale_xtest=scale.fit_transform(xtest)

LRe=LogisticRegression()
LRe.fit(scale_xtrain,ytrain)
ypre=LRe.predict(scale_xtest)
print("LogisticRegression accuracy is ={:.2%}".format(accuracy_score(ytest,ypre)))
#print("LogisticRegression",classification_report(ytest,ypre))

#KNN (k-nearest neighbors)

model=KNeighborsClassifier()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
print(" KNN accuracy is ={:.2%}".format(accuracy_score(ytest,ypred)))
#print('KNN',classification_report(ytest,ypre))

#RandomForestClassifier

model=RandomForestClassifier(n_estimators=20)
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
print(" RandomForestClassifier accuracy is ={:.2%}".format(accuracy_score(ytest,ypred)))
#print('RandomForestClassifier',classification_report(ytest,ypred))

#Gradient Boosting Classifier
model=GradientBoostingClassifier()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
print(" Gradient Boosting Classifier accuracy is ={:.2%}".format(accuracy_score(ytest,ypred)))
print('Gradient Boosting Classifier',classification_report(ytest,ypred))
