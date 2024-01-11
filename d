from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
 
Fp = '/Users/auanws/Desktop/data/'
Fn = 'Iris.xlsx'
apr = pd.read_excel(Fp + Fn)
 
apr.drop(columns=['Id'],inplace=True)
encoders = []
for i in range(0, len(apr.columns)-1):
    enc = LabelEncoder()
    apr.iloc[:,i] = enc.fit_transform(apr.iloc[:, i])
    encoders.append(enc)
x = apr.iloc[:,0:4]
y = apr['Species']
x_train,x_test,y_train,y_test =train_test_split(x,y)

model = DecisionTreeClassifier(criterion='entropy')
model.fit(x,y)
x_pred = [5.3,3.5,1.1,0.4]
for i in range(0,len(apr.columns)-1):
    x_pred[i] = encoders[i].transform([x_pred[i]])
x_pred_adj =np.array(x_pred).reshape(-1,4)
y_pred = model.predict(x_pred_adj)
print('Pred : ',(y_pred))
score = model.score(x,y)
print('Accuracy : ','{:.2f}'.format(score))
 
feature = x.columns.tolist()
Data_class = y.tolist()
plt.figure(figsize=(25,20))
_ = plot_tree(model,
              feature_names= feature,
              class_names = Data_class,
              label='all',
              impurity=True,
              precision=3,
              filled=True,
              rounded=True,
              fontsize=16)
plt.show()
import seaborn as sns
Feature_imp = model.feature_importances_
feature_names = feature
 
sns.set(rc = {'figure.figsize' : (11.7,8.7)})
sns.barplot(x = Feature_imp, y = feature_names)
print(Feature_imp)
