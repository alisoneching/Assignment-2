#-------------------------------------------------------------------------
# AUTHOR: Alison Ching
# FILENAME: knn.py
# SPECIFICATION: read the file email_classification.csv and compute the LOO-CV error rate for a 1NN classifier on the spam/ham classification task
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hr
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

# initialize variables to calculate error rate
error_count = 0
total_samples = len(db)
  
#Loop your data to allow each instance to be your test set
for i, test in enumerate(db):

    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 3], [2, 1,], ...]].
    #Convert each feature value to float to avoid warning messages

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages

    X = []
    Y = []

    for j, row in enumerate(db):
       if i != j:
        X.append([float(value) for value in row[:-1]])
        if row[-1] == 'spam':
            Y.append(float(1))
        elif row[-1] == 'ham':
           Y.append(float(2))

    #Store the test sample of this iteration in the vector testSample
    testSample = [float(value) for value in test[:-1]]
    
    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    class_predicted = clf.predict([testSample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    if test[-1] == 'spam':
       true_label = float(1)
    elif test[-1] == 'ham':
       true_label = float(2)  

    if true_label != class_predicted:
       error_count += 1

#Print the error rate
error_rate = error_count / total_samples
print('Error Rate:', error_rate)