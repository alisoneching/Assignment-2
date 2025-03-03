#-------------------------------------------------------------------------
# AUTHOR: Alison Ching
# FILENAME: naive_bayes.py
# SPECIFICATION: read the file weather_training.csv and output the classification of each of the 10 instances form the file weather_test if the classification confidence is >= 0.75
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hr
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

training = []

#Reading the training data in a csv file
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            training.append (row)

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
X = []

def x_mapping(set):
    """
    This function stores each value as numerical
    It takes the original set and returns the array of the sets of numerical values
    """
    x = []
    for row in set:
        array = []  # temporary array to store numerical values of row
        # change attribute 'outlook' values into numerical
        outlook = row[1]
        if outlook == 'Sunny':
            array.append(1)
        elif outlook == 'Overcast':
            array.append(2)
        elif outlook == 'Rain':
            array.append(3)
        # change attribute 'temperature' values into numerical
        temperature = row[2]
        if temperature == 'Hot':
            array.append(1)
        elif temperature == 'Mild':
            array.append(2)
        elif temperature == 'Cool':
            array.append(3)
        # change attribute 'humidity' values into numerical
        humidity = row[3]
        if humidity == 'High':
            array.append(1)
        elif humidity == 'Normal':
            array.append(2)
        # change attribute 'wind' values into numerical
        wind = row[4]
        if wind == 'Weak':
            array.append(1)
        elif wind == 'Strong':
            array.append(2)
        x.append(array)
    return x
# add array of numerical values into X array
X = x_mapping(training)

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
Y = []
for row in training:
    playTennis = row[-1]   # take last value of row for 'PlayTennis'
    if playTennis == 'Yes':
        Y.append(1)
    elif playTennis == 'No':
        Y.append(2)

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
test_raw = []
test = []
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            test_raw.append (row)
        #Printing the header os the solution
        elif i == 0:
            print("".join([f"{v:<12}" for v in row]), end="")
            print("Confidence")
test = x_mapping(test_raw)

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
predictions = clf.predict_proba(test)
predicted_classes = clf.predict(test)
for i, (prob, pred) in enumerate(zip(predictions, predicted_classes)):
    # check if classification confidence is >= 0.75
    if max(prob) >= 0.75:
        # print line directly from weather_test.csv
        with open('weather_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for j, row in enumerate(reader):
                if j == i + 1:
                    print("".join([f"{v:<12}" for v in row[:-1]]), end="")
        #print PlayTennis prediction
        if pred == 1:
            class_label = 'Yes'
        elif pred == 2:
            class_label = 'No'
        print(f"{class_label:<12}", end="")
        # print classification confidence
        print(f"{max(prob):.2f}")