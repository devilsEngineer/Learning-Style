# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 19:59:33 2019

@author: kuttattu
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from areaOfInterestRules import areaOfInterest,learningCombination
import time
from pieChart import drawPieChart
import collections
from os import system
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.cluster import adjusted_rand_score



start = time.time()

"""Reading The CSV file"""

df = pd.read_csv("my_csv.csv")

"""Creating Initial Learner Pie Chart"""

initialCount=df['Learner'].value_counts().to_dict()
drawPieChart(list(initialCount.keys()),list(initialCount.values()),'initial.png')

"""Preprocessing The Data"""

df['Learner']=df['Learner'].map({'A':0,'V':1,'K':2})
def preprocess_features(X):
    
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    #df.iteritems(): Iterator over (column name, Series) pairs.
    for col, col_data in X.iteritems():
        
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        output = output.join(col_data)
    
    return output

X_all=preprocess_features(df)

"""Clustering The Data Using K-Means"""

n_clusters=51
kmeans = KMeans(n_clusters)
kmeans.fit(X_all)
centroids =(kmeans.cluster_centers_)
cluster = kmeans.predict(X_all)
#print(cluster)
X_all.insert(len(X_all.columns)-1,"Cluster",cluster)
print(X_all.head())

"""Spliting The Dataset"""
x=X_all.iloc[:,:-1]
y=X_all.iloc[:,-1]
x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.30)

"""Training Logestic model to predict the learner"""
model = svm.SVC()
model.fit(x_train, y_train)

'''After training combining y_train with x_train'''
y_train=y_train.to_frame()
x_train['Learner']=(y_train['Learner'])
print(x_train.head())

"""Finding learning style frequency in each cluster"""
clusterFreq=[]
for i in range(0,n_clusters):
    temp=[0,0,0]
    for index,j in x_train.iterrows():
        if(i==j['Cluster']):
            temp[j['Learner']]+=1
    clusterFreq.append(temp)       
#print(clusterFreq)

"""Finding the dominant learning style in each cluster"""
clusterDominance={}
ind=0
for i in clusterFreq:
    clusterDominance[ind]=i.index(max(i))
    ind+=1
#print(clusterDominance)

    
"""Finding adjusted rand index value with respect of each cluster centroids"""
cluster_similarity=[]
for i in range(0,len(centroids)):
    temp=[]
    for c in centroids:
        temp.append(adjusted_rand_score(centroids[i],c))
    cluster_similarity.append(temp)

print("Adjusted Rand index :")
print(cluster_similarity)


"""To get the indices of similar clusters (Matching Threshold) """
threshold=1.0
def indices(lst, element):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset+1)
        except ValueError:
            return result
        result.append(offset)

"""Finding combination of learning style by combining dominance of similiar clusters""" 
mapLearn={'0':'A','1':'V','2':'K'}
finalLearning=[]
for clust in range(0,len(cluster_similarity)):
    thresholdIndex=indices(cluster_similarity[clust],threshold)
    finalLearning.append(mapLearn[str(clusterDominance[i])])
    for key in thresholdIndex:
        if(mapLearn[str(clusterDominance[key])] not in finalLearning[clust]):
            finalLearning[clust]= finalLearning[clust]+mapLearn[str(clusterDominance[key])]

print("Cluster And Respective Learning Style Combination :")      
for i in range(0,len(finalLearning)):
    print("Cluster " +str(i) +" : "+finalLearning[i])
    
"""Adding combination value to x_train"""
x_train.insert(len(x_train.columns),"Combination","NA")
for index,j in x_train.iterrows():
        x_train.loc[index, 'Combination']=''.join(sorted(finalLearning[j['Cluster']]))


"""prediction for learner"""
pred=model.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
print("Accuracy for learner:")
print(accuracy_score(y_test,pred))
print("Confusion Matrix for learner:")
print(confusion_matrix(y_test,pred))
print("Classification Matrix for learner:")
print(classification_report(y_test, pred))


"""training the model for combination using x_train"""
y_CombinationTrain=x_train["Combination"]
x_CombinationTrain=pd.DataFrame()
x_CombinationTrain['Cluster']=x_train["Cluster"]
x_CombinationTrain['Learner']=x_train["Learner"]
x_CombinationTest=pd.DataFrame()
x_CombinationTest['Cluster']=x_test["Cluster"]
x_CombinationTest['Learner']=pred

clf = DecisionTreeClassifier(criterion = "entropy")
clf = clf.fit(x_CombinationTrain, y_CombinationTrain)

feature_names = ['Cluster','Learner']
dotfile = open("dtree.dot", 'w')
dotfile = tree.export_graphviz(clf, out_file = dotfile, feature_names = feature_names)
system("dot -Tpng .dot -o dtree.png")

  
"""Predictions on the test data for combinations"""
combinationPred = clf.predict(x_CombinationTest) 
print(combinationPred)

"""Creating Pie Chart for final Learning style"""
trainCount=y_CombinationTrain.value_counts().to_dict()
testCount=collections.Counter(combinationPred)
#print(list(trainCount.values()))
#print(list(testCount.values()))

for trainkey in trainCount:
    for testkey in testCount:
        if(trainkey==testkey):
            trainCount[trainkey]=trainCount[trainkey]+testCount[trainkey]
#print(list(trainCount.values()))
drawPieChart(list(trainCount.keys()),list(trainCount.values()),'final.png')


"""Mapping Based on rule system"""
mappedData=[]
engine = areaOfInterest()
engine.reset()
for pred_comb in combinationPred:  
    engine.reset()
    engine.declare(learningCombination(list(pred_comb)))
    engine.run()
    mappedData.append(engine.value)
print(mappedData)


end = time.time()
print("Time taken for execution - > "+str(end - start)+" seconds")





