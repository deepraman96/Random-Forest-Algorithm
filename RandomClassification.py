#pip install pandas
#pip install imblearn
#pip install matplotlib
#pip install tensorflow
#pip install keras

print("Process Started...")
import warnings
warnings.filterwarnings("ignore")

import pandas
import numpy as np
import matplotlib.pyplot as plotG

import socket, struct

import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.preprocessing import LabelEncoder
from collections import Counter

pathForTheDataSet = "/Users/mac/Desktop/Project/archive/data.csv"
objReader = pandas.read_csv(pathForTheDataSet)

# Checking if any value in the dataframe is null
if objReader.isnull().values.any():
    # removing NA values
    objReader.dropna()
    print("Some of the values in dataframe is null")
else:
    print("None of the values in dataframe is null")



# Checking types of values
dataTypesInDataSet=objReader.dtypes
columns=objReader.columns.tolist()
for x in range(len(dataTypesInDataSet)):
    print('{} , {}'.format(columns[x],dataTypesInDataSet[x]))


# Checking occurance of each application
print(objReader['ProtocolName'].value_counts())
print('Total no. of protocols used : {}'.format(len(objReader['ProtocolName'])))


index_names = objReader[ (objReader['ProtocolName'] != "EBAY") & (objReader['ProtocolName'] != "NETFLIX") & (objReader['ProtocolName'] != "INSTAGRAM") & (objReader['ProtocolName'] != "SPOTIFY") & (objReader['ProtocolName'] != "OFFICE_365")].index
objReader.drop(index_names, inplace = True)

# Plot the number of records for individual applications
countOfProtocolUsed = objReader['ProtocolName'].value_counts()
plotG.figure(1)
countOfProtocolUsed.plot(kind='bar', title='Occurance Of Individual Application');
plotG.show()


features = [x for x in objReader.columns if x != 'ProtocolName' and x != 'Flow.ID' and x != 'Timestamp' and x != 'Label' and x != 'Source.IP' and x != 'Destination.IP']
X = objReader[features].astype(float)
Y = objReader['ProtocolName']

SampSize=6000
SlctApps = {"OFFICE_365":   SampSize,
            "EBAY":       SampSize,
            "NETFLIX":    SampSize,
            "INSTAGRAM":  SampSize,
            "SPOTIFY":    SampSize}

# manage unbalanced data
pipe = make_pipeline(
    SMOTE(sampling_strategy=SlctApps)
)
X_resampled, y_resampled = pipe.fit_resample(X, Y)


print("Size of Total dataset " + str(objReader.shape))
print("Size of Actual Features " + str(X.shape))
print("Size of Processed Features " + str(X_resampled.shape))

###################################
#Converting output class to numeric
labelEncoder = LabelEncoder()
labelEncoder.fit(Y)
labelEncoded_Y = labelEncoder.transform(Y)
Y=labelEncoded_Y

#Converting datatype to category
data_y = Y


#Spliting of data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, data_y, test_size = 0.4, random_state = 42)

#Scaling of data
minMaxScaler = MinMaxScaler()
X_train = minMaxScaler.fit_transform(X_train)
X_test = minMaxScaler.transform(X_test)


model = RandomForestClassifier(max_depth=24, n_estimators=20, max_features='auto')

import time
startR = time.time()
history = model.fit(X_train, y_train,
                    validation_data=(X_train, y_train),
                    epochs=10,
                    batch_size=64)
endR = time.time()
print(endR - startR, "train seconds")

plotG.figure(2)
plotG.plot(history.history['acc'])
plotG.plot(history.history['val_acc'])
plotG.title('Calculated Accuracy of Training and Validation Data')
plotG.ylabel('Accuracy')
plotG.xlabel('Epochs')
plotG.legend(['Training', 'Validation'], loc='upper left')
plotG.show()


import time
startE = time.time()
y_pred=model.predict(X_test)
endE = time.time()
print(endE - startE, "test seconds")

Conf=metrics.confusion_matrix(y_test, y_pred)
fig, ax = plotG.subplots(figsize=(7.5, 7.5))
ax.matshow(Conf, cmap=plotG.cm.Blues, alpha=0.3)
for i in range(Conf.shape[0]):
    for j in range(Conf.shape[1]):
        ax.text(x=j, y=i,s=Conf[i, j], va='center', ha='center', size='xx-large')

plotG.xlabel('Predictions', fontsize=18)
plotG.ylabel('Actuals', fontsize=18)
plotG.title('Confusion Matrix For random Forest', fontsize=18)
plotG.show()


accuracy = metrics.accuracy_score(y_test, y_pred)*100
precision = metrics.precision_score(y_test, y_pred,average='macro')*100
recall = metrics.recall_score(y_test, y_pred,average='macro')*100
fscore = metrics.f1_score(y_test, y_pred,average='macro')*100

print("Accuracy:",accuracy)
print("Precision:",precision)
print("Recall:",recall)
print("Fscore:",fscore)


plotG.figure(6)
plotG.bar(['Accuracy','Precision','Recall','Fscore'],
        [accuracy,precision,recall,fscore], color ='maroon',
        width = 0.4)
plotG.xlabel("Performance factors")
plotG.ylabel("Values(%age)")
plotG.title('Network\'s Performance Results');
plotG.show()
