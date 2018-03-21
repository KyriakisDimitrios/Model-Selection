# -*- coding: utf-8 -*-
"""
Created on May 2017
Project: Methods in Bioinformatics

@author: Dimitris Kyriakis
"""

#=============================================================================#
#================================ LIBRARIES ==================================#
#=============================================================================#

print("\n\n#################\nLoadinLg Libraries")
## MAIN ##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## PIPELINE ##
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

## FEATURE ##
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.feature_selection import VarianceThreshold

## CLASSIFIERS ##
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectPercentile,f_classif
transform = SelectPercentile(f_classif)
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
###############################################################################

#=============================================================================#
#============================ CHOOSE DATA SET ================================#
#=============================================================================#

Working_Dir = "C:/Users/Eddie/Desktop/Master/2nd_Semester/Methods_in_Bioinformatics/Tsamard/Exercise/"

Choose_Data_Set = input("""Choose_Data_Set:\n\t
         1.Lupus\n\t
         2.Psoriasis\n\t
         3.Autism\n\t
         4.Psoriasis_RNAseq\n\t
 (1/2/3/4): """)
#Choose_Data_Set = "3"
 
if Choose_Data_Set =="1":
    Train = Working_Dir + "Lupus/SRP062966Give.csv"
    Test = Working_Dir + "Lupus/SRP062966Validation.csv"
    Case= "Lupus"
    control = "control"
elif Choose_Data_Set == "2":
    Train = Working_Dir + "Psoriasis/GDS4602Give.csv"
    Test = Working_Dir + "Psoriasis/GDS4602Validation.csv"
    Case= "Psoriasis"
    control = "healthy"
elif Choose_Data_Set == "3":
    Train = Working_Dir + "Autism/GDS4431Give.csv"
    Test = Working_Dir + "Autism/GDS4431Validation.csv"
    Case= "Autism"
    control = "control"
else:
    Train = Working_Dir + "Psoriasis_RNAseq/SRP035988Give.csv"
    Test = Working_Dir + "Psoriasis_RNAseq/SRP035988Validation.csv"
    Case= "Psoriasis_RNAseq"

print("\n\n#################\n"+Case + " Choosed\n")

Data = np.genfromtxt(Train, delimiter=',', dtype=str)
Test_Data_raw = np.genfromtxt(Test, delimiter=',', dtype=str)
Test_Data_val = Test_Data_raw[1:,1:].astype(np.float)
Test_Data_samples = Test_Data_raw[1:,0] 
Samples_Name = Data[1:,0].reshape(Data.shape[0]-1,1)
Features = Data[0,1:]

Train_Data = (Data[1:,1:Data.shape[1]-1]).astype(np.float)

Labels_raw = Data[1:,Data.shape[1]-1]#.reshape(Data.shape[0]-1,1)
Labels_raw = Labels_raw != control
Labels = Labels_raw.astype(int)
count_control = sum(Labels == 0)
Min_Folds = min(count_control, Labels.shape[0] - count_control)

print("\n\n#################\n"+"Data Loaded\n")




###############################################################################

#=============================================================================#
#============================== Choose Classifier ============================#
#=============================================================================#
clas = input("Choose Classifier\n1. KNN\n2.Logistic Regression\n3. SVM\n((1/2/3) = ")

def Classifier_Ch(choice_cl,Min_F,thres,S,D):
    '''
    **Description:**\n 
    Create a pipeline with stdscale, variance feature selection and classifier (user input).\n
    **Input:**\n
    - Classifier: 1 = KNN, 2 = Logistic Regression, 3. SVM\n
    - Min_f: Number of Folds\n
    - Variance threshold\n
    **Output:**\n
    - Pipeline 
    '''
    ## Assign Scaler  ##
    scaler = StandardScaler()
#    scaler = MinMaxScaler()
    ## Assign Feature selection ##
#    selector = VarianceThreshold(threshold = 0.001)
    ## Assign Classifier ##
    if choice_cl == "1" :
        name = "KNN"
        clf = KNeighborsClassifier()
        param_grid = [{'clf__n_neighbors': list(range(1, 20,2)),'clf__metric': ['minkowski','euclidean','manhattan'] ,'clf__weights':['uniform','distance']}]
    elif choice_cl == "2" :
        name = "LG"
        clf = LogisticRegression()
        scaler = MinMaxScaler()
        param_grid = [{'clf__penalty': ['l1','l2'],'clf__C': [0.0001, 0.001, 0.01, 0.1, 10, 100]}]
    elif choice_cl == "3":
        clf1 = KNeighborsClassifier(n_neighbors=19,metric='manhattan',weights='uniform')
        clf2 = SVC(C =  0.0001, kernel = 'linear')
        clf3 = LogisticRegression(C = 0.01, penalty = 'l2')
        pipe1 = Pipeline([('std', StandardScaler()),('clf1', clf1)])
        pipe2 = Pipeline([('std', StandardScaler()),('clf2', clf2)])
        pipe3 = Pipeline([('std', StandardScaler()),('clf3', clf3)])       
        clf = VotingClassifier(estimators=[('clf1',pipe1),('clf2', pipe2), ('clf3',pipe3)],voting='soft')
    else:
        name = "SVM"
        clf = SVC()
        param_grid = [{'clf__kernel': ['linear'],'clf__C': [0.0001, 0.001, 0.01, 0.1, 10, 100]},{'clf__kernel': ['poly'],'clf__C': [0.0001, 0.001, 0.01, 0.1, 10, 100],'clf__degree' :[2,3]},
        {'clf__kernel': ['rbf'],'clf__C': [0.0001, 0.001, 0.01, 0.1, 10, 100],'clf__gamma': [0.001, 0.0001]}]


    ## Create Pipeline ##
    pipe = Pipeline([('std', scaler),('clf', clf)])
    gcv = GridSearchCV(estimator=pipe,param_grid=param_grid,scoring='accuracy',cv=Min_F)
    return gcv,name

if clas == "1":
    name = "KNN"
elif clas == "2":
    name = "LG"
else:
    name = "SVM"
    
    
###############################################################################

#VotingClassifier(estimators=[('clf1',pipe1),('clf2', pipe2), ('clf3',pipe3)],voting='soft')



#=============================================================================#
#=================== FEATURE SELECTION =======================================#
#=============================================================================#
thres = 0.01
in_out = "IN"


#=============================================================================#
#================================ MAIN CODE ==================================#
#=============================================================================#

if Min_Folds >=10:
    Min_Folds = 10
Min_Folds =5

mat = np.zeros((2,2))
mean_acc = 0
Best_Param = {}
lista_acc =[]
### REPEATED NESTED ###
output = open("Results/"+name+"_"+Case+".txt","w")
for  i in [1,2,3,4,5]:
    ## NESTED CROSS VALIDATION ##
    kfold = StratifiedKFold(y=Labels, n_folds= Min_Folds, shuffle=True)
   
 
    svm_list = []
    knn_list =[]
    lg_list = []
    
    for train_idx, test_idx in kfold:
        ## Assign ##
        inner_Train = Train_Data[train_idx]
        S = inner_Train.shape[0]
        D = inner_Train.shape[1]
        inner_Test = Train_Data[test_idx]
        inner_labels = Labels[train_idx]
        count_control_inner = sum(inner_labels == 0)
        Min_Folds_inner = min(count_control_inner, inner_labels.shape[0] - count_control_inner)
        if Min_Folds_inner >=10:
            Min_Folds_inner = 9
        ## Prepare ##
    
        gcv,name = Classifier_Ch(clas ,Min_Folds_inner,thres,S,D)     
        ## CROSS VALIDATION ##
        gcv.fit(inner_Train,inner_labels)
        best_par = gcv.best_params_
       
        if str(best_par) in Best_Param.keys():
            Best_Param[str(best_par)] +=1
        else:
            Best_Param[str(best_par)] = 1
        ## Predict ##
        y_test = Labels[test_idx]
        y_pred = gcv.predict(inner_Test)
        
        ## Accuracy ##
        acc = accuracy_score(y_true=y_test, y_pred=y_pred)
        lista_acc.append(acc)
        output.write("\n"+ str(best_par)+' | inner ACC %.2f%% | outer ACC %.2f%%' % (gcv.best_score_ * 100, acc * 100))
        mean_acc = mean_acc + acc
    print(i)


output.write(str(mean_acc/(5*Min_Folds)))
output.write(str(lista_acc))
output.write(str(Best_Param))
output.close()
###############################################################################








#=============================================================================#
#=================================== END =====================================#
#=============================================================================#
#
### 1.  Assign Classifier ##
#if Choose_Data_Set =="1":
#    clf = SVC(C = 0.0001,  kernel = 'linear')
#elif Choose_Data_Set == "2":
#    clf = SVC(C = 0.0001 , kernel = 'linear')
#elif Choose_Data_Set == "3":
#    clf = SVC(C = 0.0001 , kernel = 'linear')    
#else:
#    clf = SVC(C = 0.0001 , kernel = 'linear')
#
#
### 2.  Feature ##
#
#scaler = StandardScaler()
#selector = VarianceThreshold(threshold = 1)
#
## 3.  Prepare ##
#pipe = Pipeline([('std', scaler),('feat', selector),('clf', clf)])
pipe = Pipeline([('std', scaler),('clf', clf)])
#pipe.fit(Train_Data,Labels)
#
### 4.  Predict ##
#predictions  = pipe.predict(Test_Data_val)
#
## 5.  Write ##
Test_Data_samples = list(range(1,predictions.shape[0]+1))
predict = {'Id': Test_Data_samples, 'Predicted':list(predictions)}
predict = pd.DataFrame(predict)
predict.to_csv(Working_Dir + Case +"_predictions_Lasso_"+in_out+"_thres"+ str(thres) +".csv", index = False)

