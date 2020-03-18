# Detection-of-diabetes
# Introduction
Diabetes is a chronic health problem with devastating, yet preventable consequences. It is characterized by high blood glucose levels resulting from defects in insulin production, insulin action, or both. Globally, rates of type 2 diabetes were 15.1 million in 2000, the number of people with diabetes worldwide is projected to increase to 36.6 million by 2030.Along with the increase in incidence of diabetes, both individual and societal expectations concerning the management of diabetes have also increased, with many reports from The Centers for Disease Control (CDC), United States Department of Health and Human Services (USDHHS), and the National Institutes of Health (NIH) urging patients to “Take Charge of Your Diabetes” and “Conquer Diabetes”. One of the main goals of USDHHS’s report, Healthy People 2010, is to improve the quality of life for persons with diabetes.Despite the technological and scientific advances made toward the treatment of diabetes, the American Association of Clinical Endocrinologists reports that only 1 in 3 patients with type 2 diabetes is well controlled.

# Motivation
As a research group, we wanted to do our undergraduate thesis on a research that will assist a huge amount of people in their healthy lives. The number of people with diabetic retinopathy is growing higher day by day. It is estimated that the number will grow from 126.6 million to 191.0 million by 2030 and the number with vision-threatening diabetic retinopathy (VTDR) will increase from 37.3 million to 56.3 million, if any proper action is not taken. Despite growing evidence documenting the effectiveness of routine DR screening and early treatment, it is frequently leads to poor visual functioning and represents the leading cause of blindness. Most of the time it has been neglected in health care and in many low income countries because of inadequate medical service. While researching about these factors we get motivated to work with this topic. As there is insufficient ways to detect about diabetic retinopathy, we will build a system which will give prediction about diabetic retinopathy. Thus, we decided to use Machine Learning Algorithms for the prediction of this disease.

# Objectives 
This thesis mainly focuses on the prediction of diabetic and analysis performed of different algorithm for the prediction. Machine learning algorithms such as KNN, GBM, SVM, Random Forest etc. can be trained by providing training datasets to them and then these algorithms can predict the data by comparing the provided data with the training datasets. Our objective is to train our algorithm by providing training datasets to it and our goal is to detect diabetic affected using different types of classification algorithms.

# Contribution
Our objective is to train our algorithm by providing training datasets to it and our goal is to detect diabetic retinopathy using different types of classification algorithms. Here we split our dataset using cross validation so that we can train our data perfectly. And we reduce our data feature which is unnecessary. For that our model is so faster and give perfect accuracy. We also use different classification algorithm for training and test our data.

# Data Description
Our dataset contains different types of features that is extracted from the csv file. This dataset is used to predict whether a human contains signs of diabetic retinopathy or not. The value here represents different point of retina of diabetic patients. First 9 columns in the dataset are independent variables or input column and last column is dependent variables or output column. Outputs are represented by binary numbers. “1” means the patient has diabetic retinopathy and “0” means absence of the disease.


# Data preprocessing:
- To remove unwanted data from the dataset.
- To fill-up NaN data by taking mean or median value of that attribute
- Normalize the data
- Taking important attribute (Feature Reduction) for better accuracy
- Standardize the data
- K-fold cross validation
After preprocessing, we fit different model. we observe that the accuracy of the both training and testing set is quite similar and for both training and testing dataset GBM algorithm is giving higher accuracy rate which is around 90%.So, we can say that this algorithm will give us more accurate prediction about the disease. As our main purpose of the thesis is to build a model which will classify the diabetic retinopathy as accurate as possible, we hope that this final model will give us proper and appropriate results.


# Experiments and Results
In the previous chapter we have discussed about proposed system of our thesis. Now we discussing about the results we obtained from our experiments upon the implementation of this system. We have divided our dataset into two parts- training and testing dataset. In this chapter we will show the outcome of the training and testing dataset. First, we trained our dataset with these four algorithms and then we built a model. Then, we tested our testing dataset in this model. If the test set accuracy is near to train set accuracy then we can conclude that we built a good model. We have total 913 data of different individual in our dataset. There are 913 rows and 10 columns in the dataset. After splitting the data into two parts now we have 700 rows for train data and for test data we have 213 rows.
# Training & Test Accuracy of SVM Algorithm
For SVM algorithm we got training accuracy of 100% but our testing accuracy is 70.07%. We know Support Vector Machines a classifier that is defined using a separation hyper plane between the classes. But for our training dataset the accuracy of SVM is 82.01% which is not quite satisfactory. For that again we take some important feature from the dataset and again fit our dataset. For that we find the training accuracy 87.33% and test accuracy is 81.00%.
# Training & Test Accuracy of KNN Algorithm
For KNN algorithm our training accuracy is 83.02% but testing accuracy is 75%. When KNN is used for classification, the out can be calculated as the class with the highest frequency from k-most similar instances. As we want to classify our result into two part we decided to use this algorithm. . For that again we take some important feature from the dataset and again fit our dataset. For that we find the training accuracy 84.01% and test accuracy is 79.08%.
# Training Accuracy of GBM Algorithm
For GBM our training accuracy 100% but testing accuracy is 75.08%. We see that the accuracy result of KNN and Random Forest is quite close. For that again we take some important feature from the dataset and again fit our dataset. For that we find the training accuracy 100% and test accuracy is 90.05%.

# Results Analysis
After training the model we test the model with the testing dataset. We have 20% data for testing in the testing set. In the below we shows the training accuracy and testing accuracy for all algorithms

# Conclusion
We have tried to construct an ensemble to predict if a patient has diabetic retinopathy using features from retinal photos. After training and testing the model the accuracy we get is quite similar. For both sets GBM is providing higher accuracy rate for predicting DR. Despite the shortcomings in reaching good performance results, this work provided a means to make use and test multiple machine learning algorithms and try to arrive to ensemble models that would outperform individual learners. It also allowed exploring a little feature selection, feature generation, parameter selection and experiences the constraints in computation time when looking for possible candidate models in high combinatorial spaces, even for a small dataset as the one used. The structure of our research has been built in such a way that with proper dataset and minor alternation it can work to classify the disease in any number of categories.
