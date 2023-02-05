import os
os.chdir(".\\Project")

import pandas as pd
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
import seaborn as sns

application = pd.read_csv('application_record.csv')
credit = pd.read_csv('credit_record.csv')


#Number of unique applicants
print(application['ID'].count())

#Number of unique applicants
print(application['ID'].nunique())

#Number of accounts with cc record
print(credit['ID'].nunique())


#Number of null applicant values
ax = sns.set(rc={'figure.figsize':(10,10)})
ax = sns.heatmap(application.isnull(), cbar_kws={'label':'Null Values'})
ax.figure.axes[-1].yaxis.label.set_size(10)
plt.show()

#Number of null credit values
ax = sns.set(rc={'figure.figsize':(10,10)})
ax = sns.heatmap(credit.isnull(), cbar_kws={'label':'Null Values'})
ax.figure.axes[-1].yaxis.label.set_size(10)
plt.show()



#map binary categorical features to numeric
application['CODE_GENDER'].replace('M',0,inplace=True)
application['CODE_GENDER'].replace('F',1,inplace=True)
application['FLAG_OWN_CAR'].replace('Y',0,inplace=True)
application['FLAG_OWN_CAR'].replace('N',1,inplace=True)
application['FLAG_OWN_REALTY'].replace('Y',0,inplace=True)
application['FLAG_OWN_REALTY'].replace('N',1,inplace=True)

print(application.dtypes)


application['NAME_INCOME_TYPE'] = application['NAME_INCOME_TYPE'].astype('category')
application['NAME_INCOME_TYPE'].hist()
plt.xlabel('Income Source')
plt.ylabel('Number of Applicants')
plt.grid(False)
plt.show()

application['NAME_EDUCATION_TYPE'] = application['NAME_EDUCATION_TYPE'].astype('category')
application['NAME_EDUCATION_TYPE'].hist()
plt.xlabel('Education')
plt.ylabel('Number of Applicants')
plt.grid(False)
plt.show()

application['NAME_FAMILY_STATUS'] = application['NAME_FAMILY_STATUS'].astype('category')
application['NAME_FAMILY_STATUS'].hist()
plt.xlabel('Family Status')
plt.ylabel('Number of Applicants')
plt.grid(False)
plt.show()

application['NAME_HOUSING_TYPE'] = application['NAME_HOUSING_TYPE'].astype('category')
application['NAME_HOUSING_TYPE'].hist()
plt.xlabel('Housing')
plt.ylabel('Number of Applicants')
plt.grid(False)
plt.show()


#customer MOB duration
MOB = credit
MOB['MONTHS_BALANCE'] = credit['MONTHS_BALANCE'].apply(lambda x: x*-1)
MOB = MOB.groupby(['MONTHS_BALANCE']).count()
MOB = MOB.drop(['STATUS'], axis=1)
max = MOB['ID'][0]
MOB['ID'] = MOB['ID'].apply(lambda x: x/max)
MOB = MOB.rename(columns={"MONTHS_BALANCE": "Month on Book", "ID": "% Applicants"})
MOB.plot(legend=False, grid=True)
plt.xlabel('Months on Book')
plt.ylabel('% Total Applicants')
plt.show()


#how many wait the longest
credit = credit.drop('ID', axis=1)
avg_status=credit.groupby('STATUS').count()
avg_status.plot(kind='bar')
plt.show()


#Age histogram
sns.set(rc={'figure.figsize':(10,3)})
application['Age']=-(application['DAYS_BIRTH'])//365	
application['Age'].plot(kind='hist',bins=20,density=True)
plt.show()



#Income histogram
sns.set(rc={'figure.figsize':(10,3)})
sns.displot(application['AMT_INCOME_TOTAL'])
application['AMT_INCOME_TOTAL'] = application['AMT_INCOME_TOTAL'].astype(object)
application['AMT_INCOME_TOTAL'].plot(kind='hist', bins=50, density=True)
application['AMT_INCOME_TOTAL'].plot(kind='kde')
plt.show()


#income IQR by gender
sns.boxplot(x='CODE_GENDER', y='AMT_INCOME_TOTAL', data=application)
plt.show()

#income IQR by age
application['Age'] = application['Age'].apply(lambda x: round(x/10)*10)
sns.boxplot(x='Age', y='AMT_INCOME_TOTAL', data=application)
plt.show()


#income IQR by age
sns.boxplot(x='NAME_INCOME_TYPE', y='AMT_INCOME_TOTAL', data=application)
plt.show()

#income IQR by age
sns.boxplot(x='NAME_EDUCATION_TYPE', y='AMT_INCOME_TOTAL', data=application)
plt.show()



#age income relationship
application['Age'] = application['Age'].apply(lambda x: round(x/5)*5)
application.groupby(["Age"]).AMT_INCOME_TOTAL.mean().plot.barh(y='Age')
plt.show()





#numeric coorelation map
application['CODE_GENDER'].replace('M',0,inplace=True)
application['CODE_GENDER'].replace('F',1,inplace=True)
application['FLAG_OWN_CAR'].replace('Y',0,inplace=True)
application['FLAG_OWN_CAR'].replace('N',1,inplace=True)
application['FLAG_OWN_REALTY'].replace('Y',0,inplace=True)
application['FLAG_OWN_REALTY'].replace('N',1,inplace=True)


application.loc[application['DAYS_EMPLOYED'] > 0, 'DAYS_EMPLOYED'] = 0

application2 = application.drop(['Age'], axis=1)
corrMatrix = application2.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()



#why Students make so much money
application['DAYS_EMPLOYED'] = application['DAYS_EMPLOYED'].apply(lambda x: round(x/3650)*-10)
application['Age'] = application['Age'].apply(lambda x: round(x/10)*10)

application['FLAG_STUDENT'] = [1 if ele == 'Student' else 0 for ele in application['NAME_INCOME_TYPE']]

pvt_tbl = pd.pivot_table(data = application, values= 'FLAG_STUDENT', index = ['Age'], columns = ['NAME_FAMILY_STATUS'], aggfunc = sum,  fill_value = 0)
hm = sns.heatmap(data = pvt_tbl, annot = True)
plt.show()

