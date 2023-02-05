import os
os.chdir("/Users/dustinpulver/CISC_451_Project/")  ## REPLACE ## # Replace directory path with your local path 

import pandas as pd
import numpy as np
from itertools import chain, groupby

application = pd.read_csv('/Users/dustinpulver/CISC_451_Project/application_record.csv') ## REPLACE ## # Replace file path with your local path for application_record.csv
credit = pd.read_csv('/Users/dustinpulver/CISC_451_Project/credit_record.csv') ## REPLACE ## # Replace file path with your local path for credit_record.csv

#consumers spend 50% of their income
expenditure     = .50
#transcational interchange %
interchange     = .02
#annual credit card APR
APR             = .1999



##################################APPLICATION##################################################
#map binary categorical features to numeric
application['CODE_GENDER'].replace('M',1,inplace=True)
application['CODE_GENDER'].replace('F',0,inplace=True)

#map binary categorical features to numeric
application['FLAG_OWN_CAR'].replace('Y',1,inplace=True)
application['FLAG_OWN_CAR'].replace('N',0,inplace=True)

#map binary categorical features to numeric
application['FLAG_OWN_REALTY'].replace('Y',1,inplace=True)
application['FLAG_OWN_REALTY'].replace('N',0,inplace=True)

#round income into 10,000 increments
application['AMT_INCOME_TOTAL'] = application['AMT_INCOME_TOTAL'].round(decimals=-4)

#drop highly correlated feature
application = application.drop(['CNT_CHILDREN'], axis=1)

#map categorical features to ordinal
application['NAME_EDUCATION_TYPE'].replace('Lower secondary',               0,inplace=True)
application['NAME_EDUCATION_TYPE'].replace('Secondary / secondary special', 1,inplace=True)
application['NAME_EDUCATION_TYPE'].replace('Incomplete higher',             2,inplace=True)
application['NAME_EDUCATION_TYPE'].replace('Higher education',              3,inplace=True)
application['NAME_EDUCATION_TYPE'].replace('Academic degree',               4,inplace=True)

#convert days since birth to age, rounded to nearest int
application['DAYS_BIRTH'] = round(-(application['DAYS_BIRTH'])//365)


#occupation type null values handling
#assign all student and pensioners as students and pensioners
#for rest, assign occupation type based on highest count occupation by socio-demographic factors (income)

#create a dictionary of the most common occupation by socio-demographic factors
#https://stackoverflow.com/questions/52192177/convert-pandas-dataframe-to-dictionary-with-multiple-keys
df = application[['AMT_INCOME_TOTAL', 'OCCUPATION_TYPE']]
df = df.groupby(['AMT_INCOME_TOTAL', 'OCCUPATION_TYPE'])['OCCUPATION_TYPE'].count().reset_index(name='COUNT')
df = df.drop(['COUNT'], axis=1)
dctn = df.set_index(['AMT_INCOME_TOTAL']).stack().to_dict()

#iterate through rows in dataset replacing values
application['OCCUPATION_TYPE'] = application['OCCUPATION_TYPE'].fillna(0)
for i, row in application.iterrows():
    #assign students
    if row['NAME_INCOME_TYPE'] == 'Student':
        application.loc[i, 'OCCUPATION_TYPE'] = 'Student'
    #assign pensioners
    elif row['NAME_INCOME_TYPE'] == 'Pensioner':
        application.loc[i, 'OCCUPATION_TYPE'] = 'Pensioner'
    #assign null occupation types
    elif row['OCCUPATION_TYPE'] == 0:
        row['OCCUPATION_TYPE'] = dctn[(row['AMT_INCOME_TOTAL'], 'OCCUPATION_TYPE')] 
        application.loc[i, 'OCCUPATION_TYPE'] = row['OCCUPATION_TYPE']



#convert positive values (currently unemployed) to zero years of employement
application['DAYS_EMPLOYED'] = np.where((application['DAYS_EMPLOYED'] == 365243), 0, application['DAYS_EMPLOYED'])
application['DAYS_EMPLOYED'] = round(-(application['DAYS_EMPLOYED'])//365)

#if pensioneer, set working years to age-18, else set to 0 for unemployed
application['DAYS_EMPLOYED'] = np.where(((application['NAME_INCOME_TYPE'] == 'Pensioner') & (application['DAYS_EMPLOYED'] == 0)), application['DAYS_BIRTH'] - 18, application['DAYS_EMPLOYED'])

#caclulate monthly expenditure
application['month_spend'] = application['AMT_INCOME_TOTAL'] * expenditure / 12

#rename columns to avoid confusion
application = application.rename(columns={'DAYS_BIRTH':'AGE', 'DAYS_EMPLOYED':'YEARS_EMPLOYED'})



##################################CREDIT######################################################
#transpose credit columns
credit = pd.get_dummies(data=credit,columns=['STATUS'], prefix='',prefix_sep='').groupby('ID')[sorted(credit['STATUS'].unique().tolist())].sum()

credit = credit.rename(columns=
                      {'0':'pastdue_1_29',
                       '1':'pastdue_30_59',
                       '2':'pastdue_60_89',
                       '3':'write_off',
                       '4':'write_off_120',
                       '5':'write_off_150',
                       'C':'paid_off',
                       'X':'no_loan',
                      })

#merge with applicant record
vintage = pd.merge(left=application, right=credit, how='inner', on='ID')

#assign write offs as binary flag
#if they've ever gone beyond 90 days late, its a write off
vintage.loc[(vintage['write_off'] >= 1) | (vintage['write_off_120'] >= 1) | (vintage['write_off_150'] >= 1), 
            'write_off'] = 1
vintage = vintage.drop(['write_off_120', 'write_off_150'], axis=1)

#caclulate monthly expenditure
vintage['month_spend'] = vintage['AMT_INCOME_TOTAL'] * expenditure / 12


#interchange revenue
vintage['interchange_rev'] = vintage['month_spend'] * interchange * (
                        vintage['paid_off'] +
                        vintage['pastdue_1_29'] +
                        vintage['pastdue_30_59'] +
                        vintage['pastdue_60_89'])

vintage['interchange_rev'] = vintage['interchange_rev'].fillna(0)



#cumulative monthly interest rate
interest = APR / 12

#Interest = P(1+i)^n - P 
vintage['interest_rev'] =  ((((vintage['month_spend'] * vintage['pastdue_1_29'])*((1+interest) ** 1)) - (vintage['month_spend'] * vintage['pastdue_1_29'])) +
                            (((vintage['month_spend'] * vintage['pastdue_30_59'])*((1+interest) ** 2)) - (vintage['month_spend'] * vintage['pastdue_30_59'])) +
                            (((vintage['month_spend'] * vintage['pastdue_60_89'])*((1+interest) ** 3)) - (vintage['month_spend'] * vintage['pastdue_60_89'])))

vintage['interest_rev'] = vintage['interest_rev'].fillna(0)


vintage.loc[vintage['write_off'] == 0, 'revenue'] = (vintage['interchange_rev'] + vintage['interest_rev'])
#write offs use up 4 months worth of credit before we consider a net loss
vintage.loc[vintage['write_off'] == 1, 'revenue'] = (vintage['interchange_rev'] - (vintage['month_spend'] * 4))

#assign target values
#binary classification used to increase recall
vintage['target'] = vintage['revenue'].map(lambda x: 1 if x >= 0 else 0)



#get dummies but retain row values in column names
def convert_dummy(df, feature,rank=0):
    pos = pd.get_dummies(df[feature], prefix=feature)
    mode = df[feature].value_counts().index[rank]
    biggest = feature + '_' + str(mode)
    pos.drop([biggest],axis=1,inplace=True)
    df.drop([feature],axis=1,inplace=True)
    df=df.join(pos)
    return df

vintage = convert_dummy(vintage, 'NAME_FAMILY_STATUS')
vintage = convert_dummy(vintage, 'NAME_HOUSING_TYPE')
vintage = convert_dummy(vintage, 'NAME_INCOME_TYPE')
vintage = convert_dummy(vintage, 'OCCUPATION_TYPE')




###################################MODELING########################################
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

print("pre sample values : ", vintage['target'].value_counts())

resampled_df_1 = vintage[vintage['target'] == 1].sample(n=302, random_state=0)
resampled_df_0 = vintage[vintage['target'] == 0]

resampled_df = pd.concat([resampled_df_1, resampled_df_0])

print("post sample values : ",  resampled_df['target'].value_counts())

X = resampled_df.drop(['month_spend', 'pastdue_1_29', 'pastdue_30_59',
                    'pastdue_60_89', 'write_off', 'no_loan', 'interchange_rev', 
                    'interest_rev', 'revenue', 'target'], axis=1)

#X = resampled_df.drop(['FLAG_MOBIL', 'FLAG_EMAIL', 'NAME_FAMILY_STATUS_Civil marriage', 'NAME_FAMILY_STATUS_Separated', 'NAME_FAMILY_STATUS_Widow', 'NAME_HOUSING_TYPE_Co-op apartment', 'NAME_HOUSING_TYPE_Municipal apartment', 'NAME_HOUSING_TYPE_Office apartment', 'NAME_HOUSING_TYPE_Rented apartment', 'NAME_HOUSING_TYPE_With parents', 'NAME_INCOME_TYPE_State servant', 'NAME_INCOME_TYPE_Student', 'OCCUPATION_TYPE_Accountants', 'OCCUPATION_TYPE_Cleaning staff', 'OCCUPATION_TYPE_Cooking staff', 'OCCUPATION_TYPE_Drivers', 'OCCUPATION_TYPE_HR staff', 'OCCUPATION_TYPE_High skill tech staff', 'OCCUPATION_TYPE_IT staff', 'OCCUPATION_TYPE_Low-skill Laborers', 'OCCUPATION_TYPE_Managers', 'OCCUPATION_TYPE_Medicine staff', 'OCCUPATION_TYPE_Pensioner', 'OCCUPATION_TYPE_Private service staff', 'OCCUPATION_TYPE_Realty agents', 'OCCUPATION_TYPE_Sales staff', 'OCCUPATION_TYPE_Secretaries', 'OCCUPATION_TYPE_Security staff', 'OCCUPATION_TYPE_Student'], axis=1)

Y = resampled_df['target']

# X = vintage.drop(['month_spend', 'pastdue_1_29', 'pastdue_30_59',
#                     'pastdue_60_89', 'write_off', 'no_loan', 'interchange_rev', 
#                     'interest_rev', 'revenue', 'target'], axis=1)

# Y = vintage['target']

# rus = RandomUnderSampler(random_state=42)
# X_res, y_res = rus.fit_resample(X, y)

#split data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.3)

#use smote to generate equal class distribution
#X_train, Y_train = SMOTE().fit_resample(X_train, Y_train)

from sklearn.metrics import confusion_matrix, classification_report
import xgboost 
#import lightgbm 
from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler 
from imblearn.pipeline import Pipeline as imbpipeline
from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
import seaborn as sns

#https://towardsdatascience.com/the-5-classification-evaluation-metrics-you-must-know-aa97784ff226
#we want recall of 1 for target class 0  
#   how well we can predict write offs among all true write offs
###recall doesnt measure if we mislabel (predict) good clients as bad
###but it measures how many bad clients are classified correctly (1 = perfectly classified)
###the material loss of a single write off outweighs the gains of approving many good clients
###Recall should be the primary scoring method


#https://www.kaggle.com/code/palmbook/growing-rforest-97-recall-and-100-precision
##I found this link that outlines how to fine tune tree models for high recall

# model = LogisticRegression(C=0.8, random_state=0, solver='lbfgs')
# model.fit(X_train, Y_train)
# Y_predict = model.predict(X_test)

# print('Logistic Regression')
# print(classification_report(Y_test, Y_predict, digits=3))



# model = DecisionTreeClassifier(max_depth=12, min_samples_split=8,random_state=0)

# model.fit(X_train, Y_train)
# Y_predict = model.predict(X_test)

# print('Decision Tree')
# print(classification_report(Y_test, Y_predict, digits=3))



# model = RandomForestClassifier(n_estimators=250, max_depth= 12, min_samples_leaf=16)

# model.fit(X_train, Y_train)
# Y_predict = model.predict(X_test)

# print('Random Forest')
# print(classification_report(Y_test, Y_predict, digits=3))



# model = LGBMClassifier(num_leaves=31, max_depth=8, learning_rate=0.02, n_estimators=250,
#                       subsample = 0.8, colsample_bytree =0.8)

# model.fit(X_train, Y_train)
# Y_predict = model.predict(X_test)

# print('LGBM Classifier')
# print(classification_report(Y_test, Y_predict, digits=3))

# model = XGBClassifier(max_depth=12, n_estimators=250, min_child_weight=8, subsample=0.8, 
#                      learning_rate =0.02,seed=0)

# model.fit(X_train, Y_train)
# Y_predict = model.predict(X_test)

# print('XGBoost')
# print(classification_report(Y_test, Y_predict, digits=3))

def my_custom_recall_func(Y_test, y_pred):
    recall = metrics.recall_score(Y_test, y_pred, pos_label = 0)
    return recall

recall_0 = make_scorer(my_custom_recall_func, greater_is_better=True)


#plot most important features
def plot_importance(classifer, X_train, point_size = 25):
    values = sorted(zip(X_train.columns, classifer.feature_importances_), key = lambda x: x[1] * -1)
    imp = pd.DataFrame(values,columns = ["Name", "Score"])
    imp = imp[imp.Score > 0 ]
    imp.sort_values(by = 'Score',inplace = True)
    sns.set(font_scale=0.5)
    sns.scatterplot(x = 'Score',y='Name', linewidth = 0,
                data = imp,s = point_size, color='red').set(
    xlabel='importance', 
    ylabel='features')
    
    
plot_importance(model, X_train,20)   
plt.show()


#stratisfied_kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=11)

# full_pipline = imbpipeline(
#     steps=[
#         ('under_sample', RandomUnderSampler(sampling_strategy = 'majority')), #{1: x}
#         ('my_classifier', XGBClassifier(
#             objective='binary:logistic', seed=1))
#     ]
# )

# param_grid = {
#     'under_sample__sampling_strategy': Categorical(['majority']),
#     'my_classifier__n_estimators': Integer(200,300),
#     'my_classifier__max_depth': Integer(10,20) #,prior='log-uniform')
# }

param_grid = {
    'n_estimators': Integer(200,300),
    'max_depth': Integer(10,20) #,prior='log-uniform')
}

bayes_search = BayesSearchCV(
    model,param_grid, cv=5, verbose=3, n_jobs=1,n_iter=100, #stratisfied_kfold
    scoring=recall_0,return_train_score=True)

# bayes_search = BayesSearchCV(
#     full_pipline,param_grid, cv=5, verbose=3, n_jobs=1,n_iter=20, #stratisfied_kfold
#     scoring=recall_0,return_train_score=True)

bayes_search.fit(X_train, Y_train)

print('best score {}'.format(bayes_search.best_score_))
print('best score {}'.format(bayes_search.best_params_))

y_pred = bayes_search.predict(X_test)

acc = metrics.accuracy_score(Y_test, y_pred)

recall_0 = metrics.recall_score(Y_test, y_pred, pos_label = 0) # recall = TN / TN + FP
recall_1 = metrics.recall_score(Y_test, y_pred, pos_label = 1) # recall = TP / TP + FN


cm = confusion_matrix(Y_test, y_pred)


print("values : ", Y_test.value_counts())
print("accuracy: ", acc )
print("recall_0 : ", recall_0 )
print("recall_1 : ", recall_1 )
print("confusion matrix: ", cm )




# ''' commented out until other models provide higher recall reates, takes too long to process'
# model = CatBoostClassifier(iterations=100, learning_rate=0.2, od_type='Iter', verbose=25,
#                            depth=16, random_seed=0)

# model.fit(X_train, Y_train)
# Y_predict = model.predict(X_test)
# print('CatBoost')
# print(classification_report(Y_test, Y_predict, digits=3))
# '''