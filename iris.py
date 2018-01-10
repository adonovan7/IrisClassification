# Load necessary libraries
import pandas as pd # dataframe manipulation
from pandas.plotting import scatter_matrix # pandas.tools.plotting deprecated
import matplotlib.pyplot as plt # plotting
# sklearn for machine learning:
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# help("functionName") # for help on function and its syntax

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data" 
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] # specify column names
ds = pd.read_csv(url, names=names) # use pandas to load the data set with colnmanes as specified

# data variable exploration
print(ds.shape) # 150 rows, 5 columns
#print(ds.head(5)) # head first 5 rows
print(ds.describe()) # statistical properties
print(ds.groupby('class').size()) # class distribution; 50 of each type of iris

## data visualization
ds.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False) #share x,y = true puts only one ticked axis per row
#plt.show()

ds.hist() # histogram
#plt.show()

scatter_matrix(ds) # to look at correlations
#plt.show()

# machine learning
a = ds.values # puts all of the rows into array format 
# ex: [ [5.1 3.5 1.4 0.2 'Iris-setosa'], ..., ... ]
# print(a)
X = a[:, 0:4] # first four columns with 'sepal-length', 'sepal-width', 'petal-length', and  'petal-width'
Y = a[:, 4] # last column: 'class'. This is our classification attribute
validation_size = 0.20 # 80% training data, 20% test data
seed = 7 # set seed for reproducibility of random process
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size= validation_size, random_state=seed) # randomly split daa into training and test sets

seed=7
scoring= 'accuracy' # ratio of correct out of total; used to evaluate model

## Machine Learning Algorithms:
# Linear Regression, Linear Discriminant Analysis, KNN, Classification and Regression Trees, Gaussian Naive Bayes, and Support Vector Machines
# both linear and non-linear algorithms

models=[]
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results=[]
names=[]

for name, model in models:
		kfold = model_selection.KFold(n_splits=10, random_state=seed) # 10 folds, put in seed for reproducibility
		cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring) # cross validation with 10 "folds" or iterations
		results.append(cv_results)
		names.append(name)
		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) # prints the mean and standard devation of each set of cross validations for each algorithm. The %.. are place holders in strings
		print(msg)

# SVM best & KNN and CART tied for second best accuracy rates

# plot to compare algorithms
fig = plt.figure()
fig.suptitle('Algorithm Performance')
ax = fig.add_subplot(111) # 1 plot, x,y location = 1
plt.boxplot(results) # boxplot of accuracy results
ax.set_xticklabels(names) # put in model names on x-axis
plt.show() # some sample reached 100% accuracy. Notice some outliers for KNN, CART, NB, and SVM

# now that we have built and tested different models on the training data, we want to pick the best one or two to try on the test data

# model 1: KNN
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train) # fit on training data
pred_knn = knn.predict(X_validation) # make predictions based off of fitted model
print(accuracy_score(Y_validation, pred_knn)) # accuracy (correct/ total)
print(confusion_matrix(Y_validation, pred_knn)) # confusion matrix checks for True Positives, False Negatives, False Positives, and True Negatives errors/ correct predictions
# checks actual (rows) by predicted (columns) for Iris-Setosa (IS), Iris-veriscolor (IVer), and Iris-Virginica (IV)
# all 7 IS were corrected predicted, 11 IVer were correctly predicted, 1 was wrongly classified as (IV), 9 IV were correctly predicted with 2 wrongly classified as IVer
print(classification_report(Y_validation, pred_knn)) # report
# f1 is a analysis of the binary classification (correct or not); harmonic average of precision and recall, 1 is best, 0 is worst
# recall is how complete the results are (how many did the model fail to classify correctly)
# precision is how many of the classifications were correct
# in other words, precision = true pos / (true pos + false pos)
# recall = true pos / (true pos + false neg)
# accuracy of 90 %, that's pretty good!!

# model 2: SVM
svm = SVC()
svm.fit(X_train, Y_train)
pred_svm = svm.predict(X_validation)
print(accuracy_score(Y_validation, pred_svm))
print(confusion_matrix(Y_validation, pred_svm))
print(classification_report(Y_validation, pred_svm))
# accuracy of 93.33 %, even better!! 


#### So using the Support Vector Machine algorithm, I was able to get 93.33 % accuracy in the classification of iris class







