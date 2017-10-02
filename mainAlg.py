import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mpld3 as mpl

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

df  = pd.read_csv('cancer.csv', header =0)
df.drop('id',axis = 1, inplace=True)    #don't need ID & Unnamed columns
df.drop('Unnamed: 32',axis =1,inplace = True)
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})        #map [M]alignant values to '1', [B]enign values to '0'
plt.hist(df['diagnosis'])
plt.title('diagnosis(M:1 , B:0)')
#plt.show()
feature_means = list(df.columns[1:11])

dfBenign = df[df['diagnosis'] == 0]
dfMalignant = df[df['diagnosis'] == 1]

#figure, axes = #todo
print("Accuracy: %s" % "{0:.3%}".format(0.45))
traindf, testdf = train_test_split(df, train_size = 0.7)

def classModel(model, data, predictors, outcome):
        model.fit(data[predictors],data[outcome])
        prediction = model.predict(data[predictors])
        acc = metrics.accuracy_score(prediction, data[outcome])
        print("Accuracy: {0:.3%}".format(acc))
        kf = KFold(data.shape[0], n_folds = 5, shuffle = False) #datashape is the # of observations
        error = []
        for train, test in kf:
                train_Xvar = data[predictors].iloc[train,:]
                train_Yvar = data[outcome].iloc[train]
                test_Xvar = data[predictors].iloc[test,:]
                test_Yvar = data[outcome].iloc[test]
                model.fit(train_Xvar,train_Yvar)
                error.apppend(model.score(test_Xvar,test_Yvar))
                sumError = {"0:.3%"}.format(np.mean(error))
                print('CV-Score: ' + sumError)
        model.fit(data[predictors], data[outcome])
predictor = feature_means
outcome = 'diagnosis'
model = RandomForestClassifier(n_estimators=100)
classModel(model, traindf, predictor, outcome)
