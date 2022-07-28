#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 10:52:53 2022

@author: sheeva
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 17:35:07 2022

@author: sheeva
"""

##########################################################################
# PROBLEM STATEMENT : to predict the status of the interviews so that    # 
#recruiters can check the sanity of the interview & find if the interview#
# was biased.                                                            #
##########################################################################

#----------#
# FEATURES #
#----------#
#========================================================================#
#Independant values: 
    
#1  Interview Id : Id for the interview
#2  Candidate Id : Id for the candidate
#3  Interviewer Id : Id for the interviewer
#4  Profile : Profile type
#5  S.L.R.C (Speak to Listen Ratio Candidate): It is the ratio of speaking time to the listening time of the Candidate.
#6  S.L.R.I (Speak to Listen Ratio Interviewer): It is the ratio of speaking time to the listening time of the Interviewer.
#7  A.T.T Avg Turn Time (Interactivity Time): It is the average amount of time in which single interaction happens between the Interviewer and the Candidate.
#8  L.M.I (Longest Monologue Interviewer): It is the longest amount of time that the interviewer spoke continuously.
#9  L.M.C (Longest Monologue Candidate): It is the longest amount of time that the candidate spoke continuously.
#10 S.R (SILENCE RATIO):It is the percentage of time when no one had spoken
#11 L.J.T.C (Late Joining Time Candidate): It is the amount of time a candidate joined late for the interview call.
#12 L.J.T.I (Late Joining Time Interviewer): It is the amount of time the interviewer joined late for the interview call.
#13 N.I.C(Noise Index Candidate): Percentage of Background Noise present on the candidate side.
#14 N.I.I(Noise Index Interviewer): Percentage of Background Noise present on the interviewer’s side.
#15 S.P.I - Speaking Pace interviewer: Average Number of words spoken per minute.
#16 S.P.C - Speaking Pace Candidate: Average Number of words spoken per minute.
#17 L.A.I - Live Absence interviewer: It is the percentage of time the interviewer was not present in the video call.
#18 L.A.C - Live Absence candidate: It is the percentage of time the candidate was not present in the video call.
#19 Q.A - Question asked during the interview
#20 P.E.I - Perceived Emotion Interviewer: It is the perceived emotion of Interviewer which can be either Positive or Negative
#21 P.E.C - Perceived Emotion Candidate: It is the perceived emotion of candidate which can be either Positive or Negative
#22 Compliance ratio - Does the interviewer follow the structure that has been guided by the company. It is the ratio of questions assigned to the number of questions which were actually asked.
#23 Interview Duration : For how much time interview happened
#24 Interview Intro :Does interviewer give the self-introduction to the candidate or not
#25 Candidate Intro :Does candidate give the self-introduction to the interviewer or not
#26 Opp to ask : Does the interviewer give a chance to the candidate to ask questions at the end.

#Dependant values: 
#15 Status : Status of the candidate.

###########################################################################
#            IMPORT LIBRARIES                                             #
###########################################################################

import os 
import numpy as np 
import pandas as pd
import seaborn as sns 
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import figure

#-------------------------------------------------------------------------#
# Evaluation Metric: Accuracy,is a metric that summarizes the performance # 
# of a classification model as the number of correct predictions divided  #
# by the total number of predictions.                                     #
#-------------------------------------------------------------------------#

###########################################################################
#  LOADING THE DATA                                                       #
###########################################################################

# To increase the output of the print for better clarity #

pd.set_option('display.max_rows', 600)
pd.set_option('display.max_columns',600)
pd.set_option('display.width', 1200) 

# Reading the file #

os.chdir('/Users/sheeba/Desktop/data_science /python/interview/')

# Reading the data #
crudedf=pd.read_csv('train.csv') #contains both training and our 'testing' data
predicdf=pd.read_csv('test.csv') 

# Getting their shapes #

print("Shape of train :", crudedf.shape)
print("Shape of test :", predicdf.shape)

# Adding an extra column to differenciate them
crudedf['Source']='Train' # Here, the train.csv data directly has a new column
# created for itself. Also note that this data has not been split yet.

predicdf['Source']='Prediction' # Here, the prediction data is the original data
# with n-1 columns of crudedf data or train.csv data

# Here you concatinate the data so that you can do the missing value treatment
# at once #
newdf = pd.concat([crudedf,predicdf], sort = False , ignore_index= True)

# Checking the head

newdf.head()

# Dropping these three ids as it is off the bat a noisy data, # 
# that is meaningless #

newdf = newdf.drop(['Interview Id'], axis = 1) 
newdf = newdf.drop(['Candidate Id'], axis = 1) 
newdf = newdf.drop(['Interviewer Id'], axis = 1)  

# Checking the data types of the variables #
print(newdf.dtypes)

#Checking the shape of the data #
newdf.shape

# Checking for NA values #
newdf.isnull().sum() # There are ample variables with NA values
# Another important thing to note before we begin the missing value imputation
# is that initially, when I had downloaded the data, there were 4 missing values
# in the "Status" (dependant variable) so I manually imputed it, resaved the csv
# file and then loaded it into the data 

###########################################################################
#    Univariate Analysis: Missing Value Imputation                        #                                             
###########################################################################
# Infomation about the dataset
newdf.info()

# This is a loop method for missing value imputation I created #

newdf_columns=newdf.columns
print(newdf_columns)
for i in (newdf_columns):
    
    if (i in ["Status","Source" ]):
        continue
    
    if newdf[i].dtype == object:
        print("Cat: ", i)
        tempMode = newdf.loc[newdf["Source"] == "Train", i].mode()[0]
        newdf[i].fillna(tempMode, inplace = True)
    else:
        print("Cont: ", i)
        tempMedian = crudedf[i].median()
        newdf[i] = newdf[i].fillna(tempMedian)
        
newdf.isnull().sum()  # You can see that only 1200 missing values are shown
# for the "Status" section, that is coherent with the predicdf data.

# Since we didn't concatinate AFTER the train-test split, I'm also separating 
# the Y_train that is the dependant values exclusive of training data. The
# [0:5796] extracts the traindf values. Originally supposed to be 5800 with the
# data given by the website, but since I physcially removed it in the csv file,
# its 5796.

y_train=newdf["Status"][0:5796] #instead of 5800

#y_train.isnull().sum() #dependant variable yet to be split, just checking

X=newdf.drop(["Status"], axis=1) #independant variables yet to be split 

###########################################################################
#   Bivariate Analysis Independant Variables  : Boxplot                   #                                             
###########################################################################

traindf = newdf.loc[newdf['Source'] == 'Train']
continuousVars = traindf.columns[traindf.dtypes != object]
continuousVars

fileName = "/Users/sheeba/Desktop/data_science /python/interview/Analysis_Continuous_Variable.pdf"
pdf = PdfPages(fileName)
for colNumber, colName in enumerate(continuousVars): # enumerate gives key, value pair
    # print(colNumber, colName)
    figure()
    sns.boxplot(y = traindf[colName], x = traindf["Status"])
    pdf.savefig(colNumber+1) # colNumber+1 is done to ensure page numbering starts from 1 (and NOT 0)
    
pdf.close()

#  You see that the boxplot of a lot of variables are almost at par with one
# another, but I'm not removing the data due to a small possibility of this
# information having some sort of significance to the model #
# Most boxplots are almost at par so they can be dropped if possible

# Creating correlation plot
corrdf= newdf[newdf["Source"]== "Train"].corr()
corrdf.head()

#To visually see the correlation plot
rawcorr=sns.heatmap(corrdf, 
            xticklabels= corrdf.columns, yticklabels= corrdf.columns, 
            cmap='YlOrBr')
# According to this, we can see that most variables are not highly correlated at all.

# Put a limit of correlation between (0.5,1)
limitcorr=sns.heatmap(corrdf,vmin=0.5, vmax=1.0,
                      xticklabels= corrdf.columns, yticklabels= corrdf.columns, 
            cmap='YlOrBr' )

# According to this, we should technically remove all the variabbles but that is not possible
# So we can't.

###########################################################################
#   Numeric Distributuions                                                #                                             
###########################################################################

# Using the pandas function to plot a histogram of each numeric feature in the dataset.
newdf.hist(figsize=(30,30), xrot=45)
plt.show()
# Since there are a lot of unique values for each of the dataset, it is almost compressed but the distribution is 
# almost equal in all the datasets

# Summary Statistics #
newdf_Summary = newdf.describe(include = "all")
BeforeOutliers=newdf.describe() # This has been created to also check if there
# is any actual influence on the data or not.

###########################################################################
#    Bivariate Analysis Categorical Independant Variables : Histogram     #                                          #                                             
###########################################################################

categVar = traindf.columns[traindf.dtypes == object]
categVar

#Just first being done manually if the code is working or not.
sns.histplot(traindf, x="Profile", hue="Status", stat="probability", multiple="fill")

#Then you create a loop
fileName = "/Users/sheeba/Desktop/data_science /python/interview/Analysis_Categorical_Variable.pdf"
pdf = PdfPages(fileName)
for colNumber, colName in enumerate(categVar): # enumerate gives key, value pair
    
    figure()
    sns.histplot(traindf, x=colName, hue="Status", stat="probability", multiple="fill")
    # sns.displot(trainDf, x=colName, hue="Default_Payment", multiple="stack") # For count instead of proportion
    pdf.savefig(colNumber+1) # colNumber+1 is done to ensure page numbering starts from 1 (and NOT 0)
    
pdf.close()

###########################################################################
#  Label Encoder : Converting Categorical to Numerical                    #                                             
###########################################################################

# I also tried the One Hot Encoding but this worked better

from sklearn.preprocessing import LabelEncoder
X = X.apply(LabelEncoder().fit_transform) #transformed data

traindf = X[X['Source'] == 1].copy() #Just data with Train+Test = to transformed
# data that is extracting all the 1s which are train
traindf=traindf.drop(['Source'], axis = 1)
predicdf = X[X['Source'] == 0].copy() #is the production data extracted with 0=
# prediction data
predicdf=predicdf.drop(['Source'], axis = 1) #also dropping as it is unneccesary 

###########################################################################
#  Splitting the data                                                     #                                             
###########################################################################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(traindf, y_train, test_size=0.2, random_state=2)

# So the method I used is worked on most baseline models like GradientBoost, 
# KNN and Random Forest. My KNN and Random Forest baseline model didn't work
# so I'm using this first just to see base accuracy score #

import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
start_gb = time.time() # to start counting the time taken to run

gb = GradientBoostingClassifier()
gb_scores = cross_val_score(gb, X_train, y_train, cv=5, scoring='accuracy')
print('The accuracy of the Gboost classifier with Label Encoding & 10 fold cross-validation is ' + str(gb_scores.mean()))

end_gb =time.time()
gb_time = (end_gb-start_gb)/60
print('The time taken for the classifier for 10 fold cross validation is ' + str(gb_time))
# got 92.9% accuracy


###########################################################################
#   Model Building : KNN  baseline                                        #                                             
###########################################################################

# Three models are used and cross compared according to accuracy and variance

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn_scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
print('The accuracy of the knn classifier with 10 fold cross-validation is ' + str(knn_scores.mean()))
print(knn_scores)
knn_cv_mean = np.mean(knn_scores)
knn_cv_variance = np.var(knn_scores)

print('Knn Mean score : ', knn_cv_mean)
print('Knn Score variance : ', knn_cv_variance) #84.98% accuracy

###########################################################################
#   Model Building : Random Forest baseline                               #                                             
###########################################################################

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier() 
forest_scores = cross_val_score(forest, X_train, y_train, cv=10, scoring='accuracy')
print('The accuracy of the Random forest classifier with 10 fold cross-validation is ' + str(forest_scores.mean()))
print(forest_scores)
forest_cv_mean = np.mean(forest_scores)
forest_cv_variance = np.var(forest_scores)

print('forest Mean score : ', forest_cv_mean)
print('forest Score variance : ', forest_cv_variance)
#89.47% accuracy

###########################################################################
#   Model Building : Gradient Boost                                       #                                             
###########################################################################

gb = GradientBoostingClassifier()
start_gb = time.time()

gb_scores = cross_val_score(gb, X_train, y_train, cv=10, scoring='accuracy')
print('The accuracy of the Random forest classifier with 10 fold cross-validation is ' + str(gb_scores.mean()))

end_gb =time.time()
gb_time = (end_gb-start_gb)/60
print('The time taken for the classifier for 10 fold cross validation is ' + str(gb_time))
print(gb_scores)

gb_cv_mean = np.mean(gb_scores)
gb_cv_variance = np.var(gb_scores)

print('gb Mean score : ', gb_cv_mean)
print('gb Score variance : ', gb_cv_variance) #93.2% accuracy
 # Off the beat, you can see that GB method is better but will create a dataframe
 # and check
 
 ###########################################################################
#   Model Selection                                                       #                                             
###########################################################################

classifiers = { }
classifiers['Classifier'] = ['Knn','Random Forest', 'Gradient Boost']
classifiers['Mean'] = [knn_cv_mean,forest_cv_mean, gb_cv_mean]
classifiers['Variance'] = [knn_cv_variance, forest_cv_variance, gb_cv_variance]
classifiers = pd.DataFrame(classifiers)
classifiers

# Here, you see that Gradient Boost is a better model followed by Random Forest.
# Overfitting, that is train data perfoms well but the model performs very bad to 
# alien data or production data and is usually associated with higher variance

gb.fit(X_train, y_train)


# You check accuracy %
gb_accuracy = gb.score(X_test, y_test)*100  
print(gb_accuracy) #which is 92% accuracy

# Predicting the final values
Test_Pred = gb.predict(predicdf)

# Storing for submission

OutputDf=pd.read_csv('sample.csv')
#OutputDf["Interview Id"]=predicdf["Name"]
OutputDf["Status"]=Test_Pred
OutputDf.to_csv("submission_code_without_tuning.csv",index=False)






#discussion 






gb_scores = cross_val_score(gb, X_train, y_train, cv=10, scoring='accuracy')
print('The accuracy of the Random forest classifier with 10 fold cross-validation is ' + str(gb_scores.mean()))

gb_cv_mean = np.mean(gb_scores)
gb_cv_variance = np.var(gb_scores)

print('gb Mean score : ', gb_cv_mean)
print('gb Score variance : ', gb_cv_variance)

gb.fit(X, y)
#gb.fit(X, y_test)
gb_accuracy = gb.score(X_test, y_test)*100      
gb_accuracy 

gb = GradientBoostingClassifier(n_estimators = 150, learning_rate= 0.05, max_depth= 3, max_features = 5, min_samples_split=50)
gb.fit(X_train,y_train)
gb_accuracy = gb.score(X_test, y_test)*100  
print(gb_accuracy)
gb_predict=gb.predict(predicdf)



X_train.isnull().sum()
X_test.isnull().sum()
predicdf.isnull().sum()

X_train.columns
predicdf.columns

OutputDf=pd.read_csv('sample.csv')
#OutputDf["Interview Id"]=predicdf["Name"]
OutputDf["Status"]=gb_predict
OutputDf.to_csv("SUBM.csv",index=False)