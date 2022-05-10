import pandas as pd
import numpy as np

import tabulate as tb
import matplotlib.pyplot as plt
import matplotlib.image as pltimg 
import graphviz
import pydotplus

from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier as DC
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import classification_report as CR
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score



#Since we get a warning saying we are working on a copy this removes the warning
pd.options.mode.chained_assignment = None

def dataProcessing(data):

    print(f"The number of tuples present in the data set are: {len(data.index)}")
    print(f"The number of classes present in the data set are: {len(data.columns.values)}")
    for x in data.keys():
        print(f"\tThe column \'{x}\' have \'{data[x].dtypes}\' type of data with {data[x].nunique()} unique values")

    print(f"\nThe feautres present in the data set are: {feautres}")
    print(f"The classification present in the data set are: {target}")
    
    print("\nChecking the data for any missing vaule(s)...")
    print("The null values presented in the data set are located at index(s): ")
    for x in range(len(nlist)):
        print(data.iloc[[nlist[x]]])
    choice = int(input("\nHow would you like to deal with the missing vaule(s)\n1.Drop all missig vaules\n2.Add data in its place\n"))
    if choice == 1:
        data = data.dropna().reset_index() 
    elif choice == 2:
        for x in range(len(nlist)):
            print(data.iloc[[nlist[x]]])
            dr = input("\nWhat would you like to replace the missing data in this row with: ") 
            data.iloc[nlist[x]] = data.iloc[nlist[x]].fillna(dr)
    
    print("\nProcessing the vaule distribution of the data...")
    #getting the number of males and females to calculate the Percentage 
    male = data[target].tolist().count('Male') / len(data.index) * 100
    female = data[target].tolist().count('Female') / len(data.index) * 100
    print(f"after calulating the data we found that males are %{male} and females are %{female}")
    choice = int(input("\nWould you like to visualize the data? \n1.Yes\n2.No\n"))
    if choice == 1:
        draw(data)

    return data 


def draw(data):
    plt.hist(data.Gender , density=0 , bins = 2 )
    plt.axis([-1, 2, 0, len(data.index)])
    plt.xlabel('Gender')
    plt.ylabel('Users')
    plt.show()

def Tree(training, test, features, target):

    t_data = DC()

    t_data = t_data.fit(training[features], training[target])


    dot_graph = tree.export_graphviz(t_data, out_file=None, feature_names = features, filled=True)
    graph = pydotplus.graph_from_dot_data(dot_graph)
    graph.write_png("decision tree.png")
    image = pltimg.imread("decision tree.png")
    image_plot = plt.imshow(image) 
    choice = int(input("\nthe decision tree have been saved to \"decision tree.png\" would you like to visualize it right now?\n1.Yes\n2.No\n"))
    if choice == 1:
        plt.show()

    print("Done Training.")
    print("Testing the decision tree model...")
    prediction = t_data.predict(test[features])
    print("Predicions:\n", prediction)
    print("The actual data:\n", list(test[target]))
    
  
    matrix = CR(test[target], prediction, labels = [0,1])
    print("\nDecision Tree Classification Report: \n", matrix)

def KNN(training, test, features, target, n):

    k_data = KNC(n)
    k_data = k_data.fit(training[features], training[target])

    print("Done Training.")
    print("Testing the KNN model...")
    prediction = k_data.predict(test[features])
    print("Predicions:\n", prediction)
    print("The actual data:\n", list(test[target]))
  
    matrix = CR(test[target], prediction, labels = [0,1])
    print("\nKNN Classification Report: \n", matrix)

    print("Training a model using GridSearchCV for finding the optimal n_neighbors for this data set...")

    param_gird = {"n_neighbors": np.arange(1,25)}

    knn_gscv = GridSearchCV(k_data, param_gird, cv = 5)
    knn_gscv = knn_gscv.fit(training[features], training[target])

    print("Done Training.")
    print("Testing the KNN with the aid of GridSearchCV model...")
    prediction_gscv = knn_gscv.predict(test[features])
    print("Predicions:\n", prediction_gscv)
    print("The actual data:\n", list(test[target]))

    matrix_gscv = CR(test[target], prediction_gscv, labels = [0,1])
    print(f"\nKNN Classification Report using the optimal n_neighbors {knn_gscv.best_params_}: \n", matrix_gscv)

    
    return list(knn_gscv.best_params_.values())


def KF(n, features, target):
    knn = KNC(n)
    kf = KFold(n_splits=10)
    i = 0
    acc = []
    print("The actual data in the dataset:\n",list(target.values))
    for train, test in kf.split(features):
 
       knn.fit(features.iloc[train], target.iloc[train])
       print("Done Training.")
       print("Testing the KNN model...")
       prediction = knn.predict(features.iloc[test])
       print("\nPredicions:\n", prediction)
       print(f"{i+1}.KNN Classification Report using the optimal n_neighbors {n}: \n", CR(target.iloc[test], prediction))
       i+= 1
       acc.append(int(accuracy_score(target.iloc[test] , prediction) * 100))

    plt.plot(range(i),acc)
 
# naming the x axis
    plt.xlabel('i')
# naming the y axis
    plt.ylabel('The accuracy')
# giving a title to my graph
    plt.title('K fold accuracy graph')
 
# function to show the plot
    plt.show() 




data = pd.read_csv("input.csv")

feautres = data.columns.values[1:]
target = data.columns.values[0]

n_Gender = {"Female": 0, "Male" : 1}
n_App = {"Whatsapp": 0, "Snapchat": 1, "Twitter": 2, "TikTok": 3, "Linkedin": 4, "Reddit": 5, "Instagram" : 6, "Discord" : 7}

nlist = np.where(data.isna() == True)[0]

print("Processing the data...")
data = dataProcessing(data)

#shuffling the data
encoded_data = data.sample(frac=1).reset_index(drop=True)
encoded_data["Gender"] = encoded_data["Gender"].map(n_Gender)
encoded_data["App"] = encoded_data["App"].map(n_App)

training_data = encoded_data.iloc[0 : int((len(data.index) *  0.7 ) + 1) ]

testing_data = encoded_data.iloc[int((len(data.index) *  0.7 ) + 1):]
testing_data = testing_data.reset_index()

print("\nTraining the decision tree model...")
Tree(training_data, testing_data, feautres, target)

print("\nTraining the KNN model...")
n = KNN(training_data, testing_data, feautres, target, 5)

print("Traning the previous KNN model with K fold methods...")
KF(n[0], encoded_data[feautres], encoded_data[target])

print("Saving the encoded data to \'output.csv\'")
encoded_data.to_csv("output.csv")





