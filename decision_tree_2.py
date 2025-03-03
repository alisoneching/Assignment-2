#-------------------------------------------------------------------------
# AUTHOR: Alison Ching
# FILENAME: decision_tree_2.py
# SPECIFICATION: train, test, and output the performance of the 3 models created by using each training set on the test set contact_lens_test.csv
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hr
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    for row in dbTraining:
        array = []  # temporary array to store numerical values of row
        # change attribute 'age' values into numerical
        age = row[0]
        if age == 'Young':
           array.append(1)
        elif age == 'Presbyopic':
           array.append(2)
        elif age == 'Prepresbyopic':
           array.append(3)
        # change attribute 'spectacle prescription' values into numerical
        spectacle = row[1]
        if spectacle == 'Myope':
           array.append(1)
        elif spectacle == 'Hypermetrope':
           array.append(2)
        # change attribute 'astigmatism' values into numerical
        astigmatism = row[2]
        if astigmatism == 'Yes':
           array.append(1)
        elif astigmatism == 'No':
           array.append(2)
        # change attribute 'tear production rate' values into numerical
        tear = row[3]
        if tear == 'Normal':
           array.append(1)
        elif tear == 'Reduced':
           array.append(2)
        # add array of numerical values into X array
        X.append(array)

    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    for row in dbTraining:
        recommended = row[-1]   # take last value of every row for 'recommended lenses'
        if recommended == 'Yes':
            Y.append(1)
        elif recommended == 'No':
            Y.append(2)

    accuracy = []
    #Loop your training and test tasks 10 times here
    for i in range (10):

        #Fitting the decision tree to the data setting max_depth=5
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=5)
        clf = clf.fit(X, Y)

        #Read the test data and add this data to dbTest
        dbTest = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    dbTest.append(row)

        # initialize true positive, false positive, etc. for calculating accuracy
        tp = 0
        tn = 0
        fp = 0
        fn = 0
       
        for data in dbTest:
           #Transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            array = []  # temporary array to store numerical values of row
            # change attribute 'age' values into numerical
            age = data[0]
            if age == 'Young':
                array.append(1)
            elif age == 'Presbyopic':
                array.append(2)
            elif age == 'Prepresbyopic':
                array.append(3)
            # change attribute 'spectacle prescription' values into numerical
            spectacle = data[1]
            if spectacle == 'Myope':
                array.append(1)
            elif spectacle == 'Hypermetrope':
                array.append(2)
            # change attribute 'astigmatism' values into numerical
            astigmatism = data[2]
            if astigmatism == 'Yes':
                array.append(1)
            elif astigmatism == 'No':
                array.append(2)
            # change attribute 'tear production rate' values into numerical
            tear = data[3]
            if tear == 'Normal':
                array.append(1)
            elif tear == 'Reduced':
                array.append(2)
            # find the class predicted using numerical array
            class_predicted = clf.predict([array])[0]

            #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            if data[4] == 'Yes':
                if class_predicted == 1:
                    tp += 1
                elif class_predicted == 2:
                    fn += 1
            if data[4] == 'No':
                if class_predicted == 1:
                    fp += 1
                elif class_predicted == 2:
                    tn += 1

        accuracy.append((tp + tn) / (tp + tn + fp + fn))

    #Find the average of this model during the 10 runs (training and test set)
    average = sum(accuracy) / len(accuracy)

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    print(f"Final accuracy when training on {ds}: {average:.3f}")




