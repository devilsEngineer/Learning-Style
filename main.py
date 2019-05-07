# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 19:59:33 2019

@author: kuttattu
"""
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("my_csv.csv")
df['Learner']=df['Learner'].map({'A':0,'V':1,'K':2})
#print(df)
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

from sklearn.cluster import KMeans
n_clusters=21
kmeans = KMeans(n_clusters)
kmeans.fit(X_all)
centroids =(kmeans.cluster_centers_)
cluster = kmeans.predict(X_all)
print("Cluster Result:")
#print(cluster)
X_all.insert(len(X_all.columns)-1,"Cluster",cluster)
print(X_all.head())

x=X_all.iloc[:,:-1]
y=X_all.iloc[:,-1]
#x['Cluster'] = cluster
x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.30)

"""Training SVM model to predict the learner"""
from sklearn.svm import SVC
svmModel=SVC()
svmModel.fit(x_train, y_train)

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
#from scipy.spatial import distance
    
"""Finding adjusted rand index value with respect of each cluster centroids"""
from sklearn.metrics.cluster import adjusted_rand_score
cluster_distance=[]
for i in range(0,len(centroids)):
    temp=[]
    for c in centroids:
        temp.append(adjusted_rand_score(centroids[i],c))
        #temp.append(distance.euclidean(centroids[i],c))
    cluster_distance.append(temp)

print("Adjusted Rand index :")
print(cluster_distance)

threshold=1.0

"""Function to find indices of a given element"""
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
for clust in range(0,len(cluster_distance)):
    thresholdIndex=indices(cluster_distance[clust],threshold)
    finalLearning.append(mapLearn[str(clusterDominance[i])])
    for key in thresholdIndex:
        if(mapLearn[str(clusterDominance[key])] not in finalLearning[clust]):
            finalLearning[clust]= finalLearning[clust]+mapLearn[str(clusterDominance[key])]
print("Cluster Learning Style Combination :")      
for i in range(0,len(finalLearning)):
    print("Cluster " +str(i) +" : "+finalLearning[i])
    
"""Adding combination value to x_train"""
x_train.insert(len(x_train.columns),"Combination","NA")
for index,j in x_train.iterrows():
        x_train.loc[index, 'Combination']=''.join(sorted(finalLearning[j['Cluster']]))
#bigdata['Learner']=bigdata['Learner'].map({'A':0,'V':1,'K':2})
"""printing training data with combination"""
print(x_train.head())


"""prediction for learner"""
pred=svmModel.predict(x_test)
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
from sklearn.naive_bayes import GaussianNB 
naiveModel = GaussianNB() 
naiveModel.fit(x_CombinationTrain, y_CombinationTrain) 
  
"""Predictions on the test data for combinations"""
combinationPred = naiveModel.predict(x_CombinationTest) 
print(combinationPred)


