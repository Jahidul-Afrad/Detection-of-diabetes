import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,KFold 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import auc, accuracy_score
#from sklearn import preprocessing
#dataset_diabetes\\diabetic_data.csv'diabetes.csv
diabetes_data = pd.read_csv('D:\\Back up hp 2000\\My File\\read\\4-2\\machine learning code\\diabetes.csv')
print(diabetes_data.shape)
#for i in range (7):
diabetes_data.fillna(str(0)) 
X=diabetes_data.iloc[:,:-1].values
Y=diabetes_data.iloc[:,8].values


#print(dataset.head())
#np.set_printoptions(edgeitems=10)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit_transform(X)
kfold = KFold(n_splits=3, shuffle=True, random_state=70)

scores = []

for train_index, test_index in kfold.split(X):   
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]



print(X_train.shape)

from sklearn import preprocessing
X_scaled = preprocessing.scale(X_train)
X_scaled.mean(axis=0)
X_scaled.std(axis=0)
scaler = preprocessing.StandardScaler().fit(X_train)
StandardScaler()
scaler.transform(X_test)


#Using GBM
from sklearn.ensemble import GradientBoostingClassifier
gb_clf = GradientBoostingClassifier(n_estimators=300, max_features=2, max_depth=9, random_state=0)
gb_clf.fit(X_train, y_train)
print("Accuracy score using GBM (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
print("Accuracy score using GGM(test): {0:.3f}".format(gb_clf.score(X_test, y_test)))


#Using K-NN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=2)
classifier.fit(X_train, y_train)
print("Accuracy score using K-NN(training): {0:.3f}".format(classifier.score(X_train, y_train)))
print("Accuracy score using K-NN(test): {0:.3f}".format(classifier.score(X_test, y_test)))

#Using Random forest
from sklearn.ensemble import RandomForestRegressor 
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 
regressor.fit(X_train, y_train)
print("Accuracy score using Random forest(training): {0:.3f}".format(regressor.score(X_train, y_train)))
print("Accuracy score using Random forest(test): {0:.3f}".format(regressor.score(X_test, y_test)))


#Using SVM
from sklearn import svm
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X_train, y_train)
print("Accuracy score using SVM(training): {0:.3f}".format(clf.score(X_train, y_train)))
print("Accuracy score using SVM(test): {0:.3f}".format(clf.score(X_test, y_test)))



diabetes_data_copy = diabetes_data.copy(deep = True)
diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

## showing the count of Nans
print(diabetes_data_copy.isnull().sum())

p = diabetes_data.hist(figsize = (10,10))

diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace = True)
diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace = True)
diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace = True)
diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace = True)
diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace = True)

p = diabetes_data_copy.hist(figsize = (10,10))

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X =  pd.DataFrame(sc_X.fit_transform(diabetes_data_copy.drop(["Outcome"],axis = 1),),columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age'])
y = diabetes_data_copy.Outcome
print(X.shape,y.shape)
trainX = X[:700]
trainY = y[:700] 
testX = X[600:]
testY = y[600:]
#Using GBM
from sklearn.ensemble import GradientBoostingClassifier
gb_clf = GradientBoostingClassifier(n_estimators=300, max_features=2, max_depth=9, random_state=0)
gb_clf.fit(trainX, trainY)
y_pred= gb_clf.predict(testX)
from sklearn.metrics import  f1_score, precision_score, recall_score, classification_report, confusion_matrix
cm=confusion_matrix(testY,y_pred)
print("Confusion Metrix for GBM:",cm)
print("Accuracy score using GBM (training): {0:.3f}".format(gb_clf.score(trainX, trainY)))
print("Accuracy score using GGM(test): {0:.3f}".format(accuracy_score(testY, y_pred)))
print('Precision for GBM:',precision_score(testY,y_pred))
print('Recall for GBM:',recall_score(testY,y_pred))
print('F1_score for GBM:',f1_score(testY,y_pred))
#Using K-NN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=2)
classifier.fit(trainX, trainY)
y_pred= classifier.predict(testX)
cm=confusion_matrix(testY,y_pred)
print("Confusion Metrix for K-NN:",cm)
print("Accuracy score using K-NN(training): {0:.3f}".format(classifier.score(trainX, trainY)))
print("Accuracy score using K-NN(test): {0:.3f}".format(accuracy_score(testY, y_pred)))
print('Precision for K-NN:',precision_score(testY,y_pred))
print('Recall for K-NN:',recall_score(testY,y_pred))
print('F1_score for K-NN:',f1_score(testY,y_pred))

#Using SVM
from sklearn import svm
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(trainX, trainY)
y_pred= clf.predict(testX)
cm=confusion_matrix(testY,y_pred)
print("Confusion Metrix for SVM:",cm)
print("Accuracy score using SVM (training): {0:.3f}".format(clf.score(trainX, trainY)))
print("Accuracy score using SVM(test): {0:.3f}".format(accuracy_score(testY, y_pred)))
print('Precision for SVM:',precision_score(testY,y_pred))
print('Recall for SVM:',recall_score(testY,y_pred))
print('F1_score for SVM:',f1_score(testY,y_pred))