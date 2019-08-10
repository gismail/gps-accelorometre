# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:13:36 2019

@author: admin
"""
import pandas as pd
import numpy as np
data_t = pd.read_csv('irp.csv');
data = pd.read_csv('irp.csv');
data = data.drop(["Unnamed: 7", "Unnamed: 8", "Unnamed: 9", "Unnamed: 10",
                  "Unnamed: 11", "Unnamed: 12", "Unnamed: 13", 
                  "Unnamed: 14", "Unnamed: 15"], axis=1)
dt=data.columns = ['DATE','TIME','ACC-GPS','VAL','X','Y','Z']
data["X"]=pd.to_numeric(data["X"],errors='coerce')
data["Y"]=pd.to_numeric(data["Y"],errors='coerce')
data["Z"]=pd.to_numeric(data["Z"],errors='coerce')
#data["VAL"]=pd.to_numeric(data["VAL"],errors='coerce')
#data["N-SATTELITE"]=pd.to_numeric(data["VAL"],errors='coerce')


'''Scaler= (Xi - Xmean) / (standard Deviation of that feature)   '''  
''' Xmean=(xi+xi+1+...+xn)/n'''


data=data[(data.values  == "ACC")|(data.values  == "GPS" ) ]

data['TIMES'] = pd.to_datetime(data['TIME'],format= '%H:%M:%S').apply(pd.Timestamp)
data['TIMES'] = data['TIMES'] - data['TIMES'].iloc[0]
data['SECONDES'] = data['TIMES'].dt.total_seconds()



data_test=data[["X", "Y","Z"]]


data2=data[data["ACC-GPS"] == "ACC"] 
data2.columns = ['DATE','TIME','ACC-GPS','VAL','X','Y','Z','TIMES','SECONDES']

data2.loc[:,"X"]=pd.to_numeric(data2.loc[:,"X"],errors='coerce')
data2.loc[:,"Y"]=pd.to_numeric(data2.loc[:,"Y"],errors='coerce')
data2.loc[:,"Z"]=pd.to_numeric(data2.loc[:,"Z"],errors='coerce')





data3=data[data["ACC-GPS"]=="GPS"]
data3.columns = ['DATE','TIME','ACC-GPS','X','Y','Z','N-SATTELITE','TIMES','SECONDES']

data3.loc[:,"X"]=pd.to_numeric(data3.loc[:,"X"],errors='coerce')
data3.loc[:,"Y"]=pd.to_numeric(data3.loc[:,"Y"],errors='coerce')
data3.loc[:,"Z"]=pd.to_numeric(data3.loc[:,"Z"],errors='coerce')
data3.loc[:,"X"]=data3.loc[:,'X'].astype(np.float64)

data_t=data_t[data_t["19-03-18"]>="19-03-31"]
data_t=data_t[data_t["19-03-18"]!="19-04-01"]
data4=data2[data2["DATE"]=="19-03-31"]#ACC NON VELO
data5=data3[data3["DATE"]=="19-03-31"]#GPS NON VELO

data6=data2[data2["DATE"]=="19-04-05"]#ACC VELO
data7=data3[data3["DATE"]=="19-04-05"]#GPS VELO

data_acc_non_velo = data4[data4['TIME'] >= '16:45:06'] #ACC NON VELO
data_gps_non_velo = data5[data5['TIME'] >= '16:45:06'] #GPS NON VELO



data_acc_velo = data6[data6['TIME'] >= '19:13:06'] #ACC VELO
data_gps_velo = data7[data7['TIME'] >= '19:13:06'] #GPS VELO



data_acc_non_velo.loc[:,'MODE'] = '0'
data_gps_non_velo.loc[:,'MODE'] = '0'

data_acc_velo.loc[:,'MODE'] = '1'
data_gps_velo.loc[:,'MODE'] = '1'

data_acc = pd.concat([data_acc_non_velo,data_acc_velo])
data_acc = data_acc.drop(["DATE", "TIME", "TIMES",], axis=1)
data_gps = pd.concat([data_gps_non_velo,data_gps_velo])
data_gps = data_gps.drop(["DATE", "TIME", "TIMES",], axis=1)
    
data_test_acc=data_acc[["X","Y","Z"]]
data_test_gps=data_gps[["X","Y","Z"]]


from sklearn.preprocessing import StandardScaler

std = StandardScaler()
X1 = std.fit_transform(data_test_acc)
X2 = std.fit_transform(data_test_gps)

data_norm_acc = pd.DataFrame(X1, columns=data_test.columns)
data_norm_gps = pd.DataFrame(X2, columns=data_test.columns)


for i in data_norm_acc.index: 
    data_acc.loc[data_acc.index[i],"Xn"]=data_norm_acc.loc[i,"X"]
    data_acc.loc[data_acc.index[i],"Yn"]=data_norm_acc.loc[i,"Y"]
    data_acc.loc[data_acc.index[i],"Zn"]=data_norm_acc.loc[i,"Z"]
for i in data_norm_gps.index: 
    data_gps.loc[data_gps.index[i],"Xn"]=data_norm_gps.loc[i,"X"]
    data_gps.loc[data_gps.index[i],"Yn"]=data_norm_gps.loc[i,"Y"]
    data_gps.loc[data_gps.index[i],"Zn"]=data_norm_gps.loc[i,"Z"]

j=0
for i in data_acc.index:
    if (j<32):
        k=data_acc.loc[i,"SECONDES"]+(1/33)*j
        data_acc.loc[i,"SECONDES"]= k
        j=j+1
        if(j==32):
            j=0
    else :
        j=0

data_acc.loc[:,'ID']='0'
data_acc=data_acc.dropna(subset=["Y"]) 
data_acc=data_acc.dropna(subset=["Z"]) 
data_acc=data_acc.dropna(subset=["X"]) 
j=0
for i in range(0,len(data_acc)):
    data_acc.loc[data_acc.index[i],'ID']=j
    j+=1
    

data_gps.loc[:,'ID']='0'
data_gps=data_gps.dropna(subset=["Y"]) 
data_gps=data_gps.dropna(subset=["Z"]) 
data_gps=data_gps.dropna(subset=["X"]) 
j=0
for i in range(0,len(data_gps)):
    data_gps.loc[data_gps.index[i],'ID']=j
    j+=1



import matplotlib.pyplot as plt

def plot_axis_acc(ax, x, y, title):
    ax.plot(x, y,'r')
    ax.set_title(title)
    ax.xaxis.set_visible(True)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)
def plot_axis_gps(ax, x, y, title):
    ax.plot(x, y,'b')
    ax.set_title(title)
    ax.xaxis.set_visible(True)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)   
def plot_activity_acc(activity):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
    plot_axis_acc(ax0, activity['ID'], activity['Xn'], 'X(acc)')
    plot_axis_acc(ax1, activity['ID'], activity['Yn'], 'Y(acc)')
    plot_axis_acc(ax2, activity['ID'], activity['Zn'], 'Z(acc)')
    plt.subplots_adjust(hspace=0.1)
    plt.show()
def plot_activity_gps(activity):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
    plot_axis_gps(ax0, activity['ID'], activity['Xn'], 'X(gps)')
    plot_axis_gps(ax1, activity['ID'], activity['Yn'], 'Y(gps)')
    plot_axis_gps(ax2, activity['ID'], activity['Zn'], 'Z(gps)')
    plt.subplots_adjust(hspace=0.1)
    plt.show()
plot_activity_acc(data_acc)
plot_activity_gps(data_gps)

import math

def magnitude(activity):
    x2 = activity['Xn'] * activity['Xn']
    y2 = activity['Yn'] * activity['Yn']
    z2 = activity['Zn'] * activity['Zn']
    m2 = x2 + y2 + z2
    m = m2.apply(lambda x: math.sqrt(x))
    return m
data_acc['magnitude']=magnitude(data_acc)
data_gps['magnitude']=magnitude(data_gps)

def plot_magnitudes_acc(activity):
    fig, (ax0) = plt.subplots(nrows=1, figsize=(15, 10), sharex=True)
    plot_axis_gps(ax0, activity['ID'], activity['magnitude'], 'Magnitude(ACC)')
    plt.subplots_adjust(hspace=0.2)
    plt.show()
def plot_magnitudes_gps(activity):
    fig, (ax0) = plt.subplots(nrows=1, figsize=(15, 10), sharex=True)
    plot_axis_gps(ax0, activity['ID'], activity['magnitude'], 'Magnitude(GPS)')
    plt.subplots_adjust(hspace=0.2)
    plt.show()
plot_magnitudes_acc(data_acc)      
plot_magnitudes_gps(data_gps)


#Function for defining the window on data
def window(axis,dx=2):
    start = 0;
    end=0;
    size = axis.count();
    print("start: ",start,"size : ",size)
    while (start <= size and end <= size ):
        end = start + dx
        if(end <= size):
            yield start,end
        start = start+int (dx/2)
        
        
#Features which are extracted from Raw sensor data
def window_summary(axis, start, end):
#    print("start: ",start,"size : ",end)
    acf = stattools.acf(axis[start:end])
#    print("acf: ",acf)
    acv = stattools.acovf(axis[start:end])
    sqd_error = (axis[start:end] - axis[start:end].mean()) ** 2
    return [
        axis[start:end].mean(),
        axis[start:end].std(),
        axis[start:end].var(),
        axis[start:end].min(),
        axis[start:end].max(),
        acf.mean(), # mean auto correlation
        acf.std(), # standard deviation auto correlation
        acv.mean(), # mean auto covariance
        acv.std(), # standard deviation auto covariance
        skew(axis[start:end]),
        kurtosis(axis[start:end]),
        math.sqrt(sqd_error.mean())
    ]

def features(user_id):
    for (start, end) in window(user_id['ID']):
#        print("start: ",start,"end: ",end)
        if(start==0):
            features = [user_id.iloc[start]["MODE"],user_id.iloc[start]["ACC-GPS"]]
        else:
            features = [user_id.iloc[start+1]["MODE"],user_id.iloc[start]["ACC-GPS"]]
        for axis in ['Xn', 'Yn', 'Zn', 'magnitude']:
            features += window_summary(user_id[axis], start, end)
#        for axis in ['Xn', 'Yn', 'Zn', 'magnitude']:
#            features += window_summary(user_id[axis], start, end)
#        for axis in ['Xn', 'Yn', 'Zn', 'magnitude']:
#            features += window_summary(user_id[axis], start, end)
        yield features        

   
     

#Main code for Pre-processing of the Data
COLUMNS = ['SECONDES', 'X', 'Y', 'Z']
i=0
for i in range(0,len(data_acc)): 
    data_acc.loc[data_acc.index[i],"ACC-GPS"]=0
i=0
for i in range(0,len(data_gps)): 
    data_gps.loc[data_gps.index[i],"ACC-GPS"]=1
user_list = [data_acc,data_gps]
#Add an additional axis of magnitude of the sensor data
from scipy.stats import skew, kurtosis
from statsmodels.tsa import stattools
import csv
#Write the feature vectors to a separate excel file
with open('Features.csv', 'w') as out:
    rows = csv.writer(out)
    for i in range(0, len(user_list)):
        for f in features(user_list[i]):
                rows.writerow(f)
# =============================================================================
dataset = np.loadtxt('Features.csv', delimiter=",")
X = dataset[:, 1:]
Yreel = dataset[:, 0]
X=pd.DataFrame(X)
Yreel=pd.DataFrame(Yreel)
dataset=pd.DataFrame(dataset)
categories = {'':["MODE","ACC-GPS"],
              "X": ["MEAN","STD","VAR","MIN","MAX","ACF-MEAN","ACF-STD","ACV-MEAN","ACV-STD","SKEW","KURTOSIS","STD-ERR",],
              "Y": ["MEAN","STD","VAR","MIN","MAX","ACF-MEAN","ACF-STD","ACV-MEAN","ACV-STD","SKEW","KURTOSIS","STD-ERR",],
              "Z": ["MEAN","STD","VAR","MIN","MAX","ACF-MEAN","ACF-STD","ACV-MEAN","ACV-STD","SKEW","KURTOSIS","STD-ERR",],
              "Magnitude": ["MEAN","STD","VAR","MIN","MAX","ACF-MEAN","ACF-STD","ACV-MEAN","ACV-STD","SKEW","KURTOSIS","STD-ERR",],
              }
dataset.columns = pd.MultiIndex.from_tuples([(k, sub) for k,v in categories.items() for sub in v])

Yreel.columns=["MODE"]

from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")
X = imputer.fit_transform(X)# remplir les valeurs qui manquent avec la methode median 
# reinject in pandas.Dataframe: 
X = pd.DataFrame(X)
#dataset.info()
categories = {'':["ACC-GPS"],
              "X": ["MEAN","STD","VAR","MIN","MAX","ACF-MEAN","ACF-STD","ACV-MEAN","ACV-STD","SKEW","KURTOSIS","STD-ERR",],
              "Y": ["MEAN","STD","VAR","MIN","MAX","ACF-MEAN","ACF-STD","ACV-MEAN","ACV-STD","SKEW","KURTOSIS","STD-ERR",],
              "Z": ["MEAN","STD","VAR","MIN","MAX","ACF-MEAN","ACF-STD","ACV-MEAN","ACV-STD","SKEW","KURTOSIS","STD-ERR",],
              "Magnitude": ["MEAN","STD","VAR","MIN","MAX","ACF-MEAN","ACF-STD","ACV-MEAN","ACV-STD","SKEW","KURTOSIS","STD-ERR",],
              }
X.columns = pd.MultiIndex.from_tuples([(k, sub) for k,v in categories.items() for sub in v])

#X.plot(secondary_y=('X','MEAN'))



#=============================================================================


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, Yreel, test_size=.4)
train =X_train
train['MODE'] = y_train
train.to_csv('TRAIN.csv', index=False, encoding='utf-8')
test =X_test
test['MODE'] = y_test
test.to_csv('TEST.csv', index=False, encoding='utf-8')


#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=61)
dt = DecisionTreeClassifier(criterion="entropy", max_depth=10) # generates predictions by respecting the training set's class distribution
rocc =[]
rocc=pd.DataFrame(rocc)
rocc["fpr"]=[]
rocc["tpr"]=[]
rocc["Threshold"]=[]
dt_result = []
knn_result=[]
for i in range(0, 10):
     X_train, X_test, y_train, y_test = train_test_split(X, Yreel, test_size=.4)
     Y_train=y_train[y_train["MODE"]==0]
     dt.fit(X_train, y_train.values.ravel())
     knn.fit(X_train, y_train.values.ravel())
     decitree =dt.score(X_test, y_test)
     kn = knn.score(X_test, y_test)
     print ('Loop ', i," : " ,'DecisionTree : ', decitree,' KNN : ',kn)
     dt_result.append(decitree)
     knn_result.append(kn)
     probs = dt.predict_proba(X_test)
     preds = probs[:,1]
     fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
     roc_auc = metrics.auc(fpr, tpr)
    # method I: plt
     plt.title('Fonction d’efficacité du récepteur')
     plt.plot(fpr, tpr, 'b', label = 'AUC DCT = %0.5f' % roc_auc)
     plt.legend(loc = 'lower right')
     plt.plot([0, 1], [0, 1],'r--')
     plt.xlim([0, 1])
     plt.ylim([0, 1])
     plt.ylabel('Taux vrai positive')
     plt.xlabel('Taux faux positive')
     plt.show()
     probs = knn.predict_proba(X_test)
     preds = probs[:,1]
     fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
     roc_auc = metrics.auc(fpr, tpr)
    
    # method I: plt
     plt.title('Fonction d’efficacité du récepteur')
     plt.plot(fpr, tpr, 'r', label = 'AUC KNN = %0.5f' % roc_auc)
     plt.legend(loc = 'lower right')
     plt.plot([0, 1], [0, 1],'r--')
     plt.xlim([0, 1])
     plt.ylim([0, 1])
     plt.ylabel('Taux vrai positive')
     plt.xlabel('Taux faux positive')
     plt.show()


    
y_pred = knn.predict(X_test)
print("Confusion Matrix Knn:\n",confusion_matrix(y_test,y_pred))

y_pred2 = dt.predict(X_test)
print("Confusion Matrix decisionTree:\n",confusion_matrix(y_test,y_pred2))

print ('DecisionTree accuracy : ', np.mean(dt_result),'\nDecisionTree STD      : ', np.std(dt_result))
print ('Knn accuracy          : ', np.mean(knn_result),'     \nKnn STD               : ', np.std(knn_result))


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(dt, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, X, Yreel.values.ravel(), cv=5)
print("cross validationn : ",scores,"\n")




from joblib import dump, load
dump(dt, 'decisiontree.joblib') 

ad = load('decisiontree.joblib')
y_pred2 = dt.predict(X_test)
print("Confusion Matrix decisionTree:\n",confusion_matrix(y_test,y_pred2))
y_pred2 = ad.predict(X_test)
print("Confusion Matrix decisionTree:\n",confusion_matrix(y_test,y_pred2))

from sklearn.metrics import f1_score
f1_score(y_test, y_pred, average=None)

#
#fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
#auc = metrics.roc_auc_score(y_test, y_pred)
#plt.plot(fpr,tpr,label="ROC knn, auc="+str(auc))
#plt.legend(loc=4)
#plt.show()
#
#fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred2)
#auc = metrics.roc_auc_score(y_test, y_pred2)
#plt.plot(fpr,tpr,label="ROC decisionTree, auc="+str(auc))
#plt.legend(loc=4)
#plt.show()

# =============================================================================
