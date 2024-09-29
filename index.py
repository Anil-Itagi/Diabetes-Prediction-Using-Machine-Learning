import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import streamlit as st

# header part in stremlit
header = st.container()
with header:
    header.markdown("#### **This app is to predict The person is diabetic or Not **")
    header.markdown("---")
# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('diabetes.csv') 

# pd.read_csv?
# printing the first 5 rows of the dataset
diabetes_dataset.head()

# number of rows and Columns in this dataset
# diabetes_dataset.shape

# getting the statistical measures of the data
diabetes_dataset.describe()
diabetes_dataset['Outcome'].value_counts()
diabetes_dataset.groupby('Outcome').mean()

# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# print(X)
# print(Y)

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
# print(standardized_data)

X = standardized_data
Y = diabetes_dataset['Outcome']
# print(X)
# print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
classifier = svm.SVC(kernel='linear')
#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)
# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)
# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)


Pregnancies=st.slider("Pregnancies", min_value=0, max_value=17, label_visibility="visible")

Glucose=st.slider("Glucose", min_value=0, max_value=199, label_visibility="visible")

BloodPressure=st.slider("BloodPressure", min_value=0, max_value=122, label_visibility="visible")

SkinThickness=st.slider("SkinThickness in (mm)", min_value=0, max_value=99, label_visibility="visible")

Insulin=st.slider("Insulin  in (µU/mL)", min_value=0, max_value=846, label_visibility="visible")

BMI=st.slider("BMI", min_value=0, max_value=67, label_visibility="visible")

DiabetesPedigreeFunction=st.slider("DiabetesPedigreeFunction", min_value=0.078, max_value=2.42, label_visibility="visible")

Age=st.slider("Age", min_value=10, max_value=100, label_visibility="visible")

input_data = (Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)

# input 
# 6,148,72,35,0,33.6,0.627,50,1

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
# print(std_data)



prediction = classifier.predict(std_data)
# print(prediction)

if st.button('Test'):
      if (prediction[0] == 0):
             st.header("The person is not diabetic")
      else:
             st.header("The person is diabetic")
