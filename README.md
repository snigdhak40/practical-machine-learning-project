The report details the building of machine learning models which predicts the classe (quality of exercise) variable from data provided by accelerometer. To do this, 5 classification models were initially selected and the best, in terms of accuracy, was fine tuned to improve the overall prediction capability. The goal was to predict the classe variable for 5 exercise performed by 6 different participants.

The performance specifications of the final model are outlined below:
Model Name	Out of Sample Accuracy	95% Confidence Interval for Accuracy	No Information Rate (NIR)	P-Value [Accuracy > NIR]	Kappa	Mcnemar’s Test P-Value
Final Model :Random Forest Model with K-Fold Cross Validation	0.982	(0.98, 0.984)	0.287	<2e-16 (Significant)	0.978	N/A
1.0 - Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behaviour, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

2.0 - Purpose and Scope
Data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants were utilized. These participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways with machine learning models. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har.

3.0 - Data Preparation
The following section details the procedure required to prepare the data for analysis. This includes getting the data, cleaning it and then splitting it.

3.1 - Getting that Data
The data sets used here include:

Training data set:
location: https://d396quszas40orc.cloudfront.net/predmachlearn/pml-training.csv
dimensions: 19622 observations of 160 variables
data extracted on: Monday, July 21, 2014 @ 12:41 AM
Testing data set:
location: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
dimensions: 20 observations of 160 variables
data extracted on: Monday, July 21, 2014 @ 12:41 AM
Since the testing data set only consists of 0.10% of the total available data, the training data set will be split into a cross validation data set to prevent over-fitting and get an out of sample error rate (for further details on the cross-validation (CV) data set, please see section 3.3)

# Importing data into memory
trainData <- read.csv(trainFile)
testData <- read.csv(testFile)
3.2 - Cleaning the Data
The original data sets included two sets of extraneous variables namely:
1. Tracking specific variables: Variables that contain record specific information (login, test number, etc…) that would be useless for model building.
2 . Zero varience variables: Variables that have zero-variance (or near zero variance) which is meaningless to the machine learning models.
3. Aggregate specific variables: Calculations that are done on an aggregate of records. Hence, these variables contain mainly NA’s (~90%).

The following code details how the above two types of extronious variables are identified and removed from the data sets.

# Identifying tracking specific variables
toMatch <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window")
removeColumns <- grep(paste(toMatch,collapse="|"), colnames(trainData))

# Identifying Zero Varience Values
nzv <- nearZeroVar(trainData, saveMetrics = TRUE)
removeColumns <- c(removeColumns,which(nzv$nzv == TRUE))

# Identifying aggregate specific variables
AggregateVals <- names(trainData[,colSums(is.na(trainData), na.rm = FALSE) > 0.95*nrow(trainData)])
NAColumns <- grep(paste(AggregateVals,collapse="|"), colnames(trainData))
removeColumns <- c(removeColumns,NAColumns)

# Finalizing the variables
removeColumns <- unique(removeColumns)
removeColumns <- sort(removeColumns)

#Preparing Tidy Data Sets
trainDataTidy <- trainData[,-removeColumns]
testDataTidy <- testData[,-removeColumns]
3.3 - Data Splitting: Preparing Cross-Validation Data Set
Since the training data set is such a huge proportion of the available data, a data partition of p=0.3 was used on the training data set to split it into a training and a cross-validation set. This split would provide 5,889 observation for training and 13733 for cross-validation. This would leave us with ~30% of available data for model training and 70% for model testing (more specifically, 69.92% for cross-validation and 0.010% for testing).

The code below details how the cross-validation data set was created.

set.seed(112)
inTest <- createDataPartition(y=trainDataTidy$classe,
                               p=0.3, list=FALSE)
training <- trainDataTidy[inTest,] 
crossVal <- trainDataTidy[-inTest,]
4.0 - Exploratory Data Analysis
Analysis was done on the cleaned training data set, to detect outlines and certain anomalies that might effect certain models.

A hierarchical cluster analysis was conducted to analyze the relationship between variables: hclust of traing data variables

Conclusion: The great a amount of correlation among variables suggest that techniques such as PCA can be used to characterize the magnitude of the problem. PCA would also help reduce computation complexity and increase numerical stability.

5.0 - Model Building
To predict the classe variable, three classes of classification models were built with different pre-processing options using the Caret package. Given the nature of the prediction variable, regression models were ruled out of this analysis.

5.1 - Model Training
The table below details the different types of models that were trained with the training data set.

Model Type	Model Class	Pre-Processing
Random Forest	Classification Models: Classification Tree	None
Stochastic Gradient Boosting (gbm)	Classification Models: Rule Based	None
Support Vector Machines (svmRadial)	Classification Models: Nonlinear	Normalization (center, scale)
Random Forest	Classification Models: Classification Tree	PCA
Stochastic Gradient Boosting (gbm)	Classification Models:Rule Based	PCA
The subsequent code blocks contain code and specific parameters used when training models. Furthermore, the doParallel library was utilized to take full advantage of multi-core machine architecture and improve run time.

** Model 1:Random Forest Model**

# Enabeling multi-core processing
cl <- makeCluster(detectCores())
registerDoParallel(cl)

# Random Forest with PCA
set.seed(112)
modelFit_rf <- train(classe ~ ., data=training, method="rf", prox=TRUE)
** Model 2:Stochastic Gradient Boosting (gbm)**

# Enabeling multi-core processing
cl <- makeCluster(detectCores())
registerDoParallel(cl)

set.seed(112)
modelFit_gbm <- train(classe ~ ., method="gbm", data=training, verbose=FALSE)
** Model 3:Support Vector Machines (svmRadial)**

# Enabeling multi-core processing
cl <- makeCluster(detectCores())
registerDoParallel(cl)

set.seed(112)
modelFit_svm <- train(classe ~ ., data=training, method="svmRadial", preProc = c("center", "scale"), metric = "Accuracy")
** Model 4:Random Forest with PCA**

# Enabeling multi-core processing
cl <- makeCluster(detectCores())
registerDoParallel(cl)

# Random Forest with PCA
set.seed(112)
modelFit_rf_PCA <- train(classe ~ ., data=training, method="rf",preProcess = "pca", prox=TRUE)
** Model 5:Stochastic Gradient Boosting (gbm) with PCA**

# Enabeling multi-core processing
cl <- makeCluster(detectCores())
registerDoParallel(cl)

set.seed(112)
modelFit_gbm_PCA <- train(classe ~ ., method="gbm", preProcess = "pca", data=training, verbose=FALSE)
5.2 - Initial Model Evaluation
The trained models were evaluated against the cross-validation data sets. The results below indicate the out-of-sample metrics for all 5 models trained above. Further details on model evaluation on all five models can be found in Appendix A: Initial Model Evaluation Detials

Model Name	Out of Sample Accuracy	95% Confidence Interval for Accuracy	No Information Rate (NIR)	P-Value [Accuracy > NIR]	Kappa	Mcnemar’s Test P-Value
Model 1:Random Forest Model	0.982	(0.98, 0.984)	0.287	<2e-16 (Significant)	0.978	N/A
Model 2:Stochastic Gradient Boosting (gbm)	0.953	(0.949, 0.956)	0.29	<2e-16 (Significant)	0.94	<2e-16 (Significant)
Model 3:Support Vector Machines (svmRadial)	0.885	(0.88, 0.891)	0.302	<2e-16 (Significant)	0.855	<2e-16 (Significant)
Model 4:Random Forest with PCA	0.939	(0.935, 0.943)	0.289	<2e-16 (Significant)	0.923	<2e-16 (Significant)
Model 5:Stochastic Gradient Boosting (gbm) with PCA	0.8	(0.793, 0.806)	0.293	<2e-16 (Significant)	0.746	<2e-16 (Significant)
6.0 - Final Model Tuning
6.1 - Final Model Training
Given that accuracy was used as a metric to evaluate all the trained models, Model 1:Random Forest Model had the best performance. Hence, this model was further tuned to improve its accuracy.

In order to improve overall accuracy while prevent over-fitting, a k-fold cross validation (where k=10) was done while re-training the model.

The code below details the process.

# Enabeling multi-core processing
cl <- makeCluster(detectCores())
registerDoParallel(cl)

set.seed(112)
fitControl <- trainControl(## 10-fold CV
                           method = "repeatedcv",
                           number = 10,
                           ## repeated ten times
                           repeats = 10)
modelFit_rf_CV <- train(classe ~ ., data=training, method="rf", trControl = fitControl, prox=TRUE)
6.2 - Final Model Evaluation
Model Name	Out of Sample Accuracy	95% Confidence Interval for Accuracy	No Information Rate (NIR)	P-Value [Accuracy > NIR]	Kappa	Mcnemar’s Test P-Value
Final Model :Random Forest Model with K-Fold Cross Validation	0.982	(0.98, 0.984)	0.287	<2e-16 (Significant)	0.978	N/A
As seen above, after the model tuning, the performance of the model has little-to-no improvement. Hence, if processing power was a limited resource, it is recommended that cross validation is not required in this case.

The figure below shows the scaled variable importance of the final model. plot of chunk unnamed-chunk-2

Hence, we see that the roll_belt is vital for model reduction and the subsequent 19 variables are major contributors to model accuracy.

Final model specifications:

mtry: 27
In-Sample Accuracy: 0.979
Kappa: 0.973
AccuracySD: 0.00613
KappaSD: 0.00776
Performance Metric: Accuracy
7.0 - Predicting the Results
7.1 - Prediction on the Test Set
Prob(A)	Prob(B)	Prob(C)	Prob(D)	Prob(E)	Final Prediction
0.08	0.79	0.09	0.03	0.02	B
0.92	0.05	0.01	0.00	0.01	A
0.10	0.67	0.17	0.01	0.04	B
0.90	0.00	0.05	0.05	0.00	A
0.96	0.01	0.02	0.00	0.01	A
0.01	0.13	0.11	0.05	0.69	E
0.03	0.01	0.14	0.77	0.05	D
0.06	0.50	0.13	0.23	0.08	B
1.00	0.00	0.00	0.00	0.00	A
1.00	0.00	0.00	0.00	0.00	A
0.05	0.49	0.34	0.07	0.04	B
0.03	0.15	0.71	0.03	0.07	C
0.02	0.92	0.01	0.00	0.06	B
1.00	0.00	0.00	0.00	0.00	A
0.01	0.03	0.04	0.01	0.91	E
0.04	0.09	0.01	0.04	0.82	E
0.97	0.00	0.00	0.00	0.03	A
0.06	0.73	0.01	0.15	0.05	B
0.11	0.85	0.01	0.02	0.01	B
0.00	1.00	0.00	0.00	0.00	B
Appendix
Appendix A: Initial Model Evaluation Detials
confusionMatrix(crossVal$classe, predict(modelFit_rf, crossVal))
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3885   17    2    0    2
##          B   57 2561   36    3    0
##          C    0   25 2362    8    0
##          D    0    3   52 2189    7
##          E    0    7   18    6 2493
## 
## Overall Statistics
##                                        
##                Accuracy : 0.982        
##                  95% CI : (0.98, 0.984)
##     No Information Rate : 0.287        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.978        
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.986    0.980    0.956    0.992    0.996
## Specificity             0.998    0.991    0.997    0.995    0.997
## Pos Pred Value          0.995    0.964    0.986    0.972    0.988
## Neg Pred Value          0.994    0.995    0.990    0.999    0.999
## Prevalence              0.287    0.190    0.180    0.161    0.182
## Detection Rate          0.283    0.186    0.172    0.159    0.182
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.992    0.986    0.977    0.993    0.997
confusionMatrix(crossVal$classe, predict(modelFit_gbm, crossVal))
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3835   41   16    3   11
##          B  129 2411   95   10   12
##          C    1   69 2290   33    2
##          D    4   15   81 2130   21
##          E    8   26   28   45 2417
## 
## Overall Statistics
##                                         
##                Accuracy : 0.953         
##                  95% CI : (0.949, 0.956)
##     No Information Rate : 0.29          
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.94          
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.964    0.941    0.912    0.959    0.981
## Specificity             0.993    0.978    0.991    0.989    0.991
## Pos Pred Value          0.982    0.907    0.956    0.946    0.958
## Neg Pred Value          0.986    0.986    0.981    0.992    0.996
## Prevalence              0.290    0.187    0.183    0.162    0.179
## Detection Rate          0.279    0.176    0.167    0.155    0.176
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.979    0.960    0.951    0.974    0.986
confusionMatrix(crossVal$classe, predict(modelFit_svm, crossVal))
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3792   22   61   23    8
##          B  292 2134  195   11   25
##          C   25  128 2120  115    7
##          D   21   30  264 1923   13
##          E   11   72  136  114 2191
## 
## Overall Statistics
##                                        
##                Accuracy : 0.885        
##                  95% CI : (0.88, 0.891)
##     No Information Rate : 0.302        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.855        
##  Mcnemar's Test P-Value : <2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.916    0.894    0.764    0.880    0.976
## Specificity             0.988    0.954    0.975    0.972    0.971
## Pos Pred Value          0.971    0.803    0.885    0.854    0.868
## Neg Pred Value          0.964    0.977    0.942    0.977    0.995
## Prevalence              0.302    0.174    0.202    0.159    0.163
## Detection Rate          0.276    0.155    0.154    0.140    0.160
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.952    0.924    0.869    0.926    0.974
confusionMatrix(crossVal$classe, predict(modelFit_rf_PCA, crossVal))
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3812   43   33   13    5
##          B  125 2390  118    4   20
##          C   10   68 2252   41   24
##          D   14   15  141 2068   13
##          E    6   51   49   47 2371
## 
## Overall Statistics
##                                         
##                Accuracy : 0.939         
##                  95% CI : (0.935, 0.943)
##     No Information Rate : 0.289         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.923         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.961    0.931    0.868    0.952    0.975
## Specificity             0.990    0.976    0.987    0.984    0.986
## Pos Pred Value          0.976    0.900    0.940    0.919    0.939
## Neg Pred Value          0.984    0.984    0.970    0.991    0.994
## Prevalence              0.289    0.187    0.189    0.158    0.177
## Detection Rate          0.278    0.174    0.164    0.151    0.173
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.976    0.954    0.928    0.968    0.980
confusionMatrix(crossVal$classe, predict(modelFit_gbm_PCA, crossVal))
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3447  149  152  134   24
##          B  272 1947  264   66  108
##          C  132  201 1911  105   46
##          D  102   61  248 1767   73
##          E   77  247  162  130 1908
## 
## Overall Statistics
##                                         
##                Accuracy : 0.8           
##                  95% CI : (0.793, 0.806)
##     No Information Rate : 0.293         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.746         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.855    0.747    0.698    0.802    0.884
## Specificity             0.953    0.936    0.956    0.958    0.947
## Pos Pred Value          0.882    0.733    0.798    0.785    0.756
## Neg Pred Value          0.941    0.941    0.927    0.962    0.978
## Prevalence              0.293    0.190    0.199    0.160    0.157
## Detection Rate          0.251    0.142    0.139    0.129    0.139
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.904    0.842    0.827    0.880    0.915
Appendix B: Final Model Evaluation Detials
confusionMatrix(crossVal$classe, predict(modelFit_rf_CV, crossVal))
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3887   14    3    0    2
##          B   57 2561   38    1    0
##          C    0   29 2358    8    0
##          D    0    2   53 2189    7
##          E    0    8   17    5 2494
## 
## Overall Statistics
##                                        
##                Accuracy : 0.982        
##                  95% CI : (0.98, 0.984)
##     No Information Rate : 0.287        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.978        
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.986    0.980    0.955    0.994    0.996
## Specificity             0.998    0.991    0.997    0.995    0.997
## Pos Pred Value          0.995    0.964    0.985    0.972    0.988
## Neg Pred Value          0.994    0.995    0.990    0.999    0.999
## Prevalence              0.287    0.190    0.180    0.160    0.182
## Detection Rate          0.283    0.186    0.172    0.159    0.182
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.992    0.986    0.976    0.994    0.997
