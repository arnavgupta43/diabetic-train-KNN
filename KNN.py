import pandas as pd
import numpy as np
# Import the Library
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# read the data
dataset=pd.read_csv('diabetes.csv')
print(len(dataset))
print(dataset.head())

# Clean the data
zero_not_accepted=['Glucose','BloodPressure','SkinThickness','BMI','Insulin']

for column in zero_not_accepted:
    dataset[column]=dataset[column].replace(0,np.nan)
    mean=int(dataset[column].mean(skipna=True))
    dataset[column]=dataset[column].replace(np.nan,mean)

# split dataset

X=dataset.iloc[:,0:8]
y=dataset.iloc[:,8]
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)

# Feature scaling
sc_X =StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

# Define the model :KNN
classifier= KNeighborsClassifier(n_neighbors=11, p=2,metric='manhattan')

# Training the model
classifier.fit(X_train,y_train)

# Making predictions
y_pred=classifier.predict(X_test)
y_pred


# Evaluate the Model: Confusion Metrics

cm=confusion_matrix(y_test,y_pred)
print(cm)
print("F1 score:",f1_score(y_test,y_pred))
print("Accuracy Score",accuracy_score(y_test,y_pred))