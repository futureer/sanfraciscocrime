# -*- coding: utf-8 -*-
"""
Created on Sat Jun 04 17:16:21 2016
#Model Random  Forest
@author: Administrator
"""
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def define_address(addr):
    return 1 if '/' in addr and 'of' not in addr else 0

def segment_hour(hour):
    return hour/6

print('Preparing training and test data...') 
dftrain = pd.read_csv('./dataset/train.csv')
#dftrain = dftrain.head(100000)
dftest = pd.read_csv('./dataset/test.csv', index_col="Id")
#dftest = dftest.head(100000)

#independent variable
train_WITH = dftrain["Category"]

#dependent variables
train_WITHOUT = dftrain.drop("Category", axis=1)

#remove insignificant variables for training
#train_WITHOUT = train_WITHOUT.drop("Descript", axis=1) 
#train_WITHOUT = train_WITHOUT.drop("Resolution", axis=1)

print('Making training a categorical structure...') 
#hours = pd.get_dummies(train_WITHOUT.Dates.map(lambda x: pd.to_datetime(x).hour), prefix="hour")
hours = pd.get_dummies(train_WITHOUT.Dates.map(lambda x: pd.to_datetime(x).hour/6), prefix="hour")
months = pd.get_dummies(train_WITHOUT.Dates.map(lambda x: pd.to_datetime(x).month), prefix="month")
years = pd.get_dummies(train_WITHOUT.Dates.map(lambda x: pd.to_datetime(x).year), prefix="year")
district = pd.get_dummies(train_WITHOUT["PdDistrict"])
day_of_week = pd.get_dummies(train_WITHOUT["DayOfWeek"])
addr = pd.get_dummies(train_WITHOUT["Address"].map(define_address),prefix="Address")

print('Train cleaning...')
train_WITHOUT = pd.concat([train_WITHOUT, hours, months, years, district, day_of_week, addr], axis=1)
train_WITHOUT = train_WITHOUT.drop(['PdDistrict', 'Dates', 'DayOfWeek', 'Address'], axis = 1)

print('Making test a categorical structure...') 
#hours = pd.get_dummies(dftest.Dates.map(lambda x: pd.to_datetime(x).hour), prefix="hour")
hours = pd.get_dummies(dftest.Dates.map(lambda x: pd.to_datetime(x).hour/6), prefix="hour")
months = pd.get_dummies(dftest.Dates.map(lambda x: pd.to_datetime(x).month), prefix="month")
years = pd.get_dummies(dftest.Dates.map(lambda x: pd.to_datetime(x).year), prefix="year")
district = pd.get_dummies(dftest["PdDistrict"])
day_of_week = pd.get_dummies(dftest["DayOfWeek"])
addr = pd.get_dummies(dftest["Address"].map(define_address),prefix="Address")

print('Train cleaning...')
dftest = pd.concat([dftest, hours, months, years, district, day_of_week, addr], axis=1)
dftest = dftest.drop(['PdDistrict', 'Dates', 'Address', 'DayOfWeek'], axis = 1)
#s_idx = np.random.permutation(train_WITHOUT.index)
#trian_shuffle = train_WITHOUT.reindex(s_idx)
#trian_label = train_WITH.reindex(s_idx)
#Start predicting

print("Classifier...")
st = time.time()
rfcl = RandomForestClassifier(n_estimators=50,n_jobs=4)

#rfcl.fit(trian_shuffle.head(300000),trian_label.head(300000))
rfcl.fit(train_WITHOUT, train_WITH)
print('Predictor...')
result = pd.DataFrame(rfcl.predict_proba(dftest), index=dftest.index, columns=rfcl.classes_)
ed=time.time()
print "takes %f seconds." %(st-ed)
print("Result...")
#result.to_csv('out_hour.csv.gz',compression='gzip')
#print(result)