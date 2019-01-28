#Coder: Abraham Do
#DSC 240 Final Project

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix

#Change to csv file location to avoid errors!
survey = pd.read_csv("/Users/pc/pycharmProjects/surveyProject/responses.csv")


#Grouping of different parts of the dataset
music = survey.iloc[:,2:19]
math = survey.iloc[:,34:36]
science = survey.iloc[:,38:41]
spending = survey.iloc[:,135:140]
active = survey.iloc[:,52:56]
religion = survey.iloc[:,102:104]

# #Quick analyzation of the descriptions of the dataset.
# print(survey.describe())
#
# #Graph of Musical Preferences
# fig, (axis1) = plt.subplots(1, figsize = (10,5), sharey= True)
# #
# sns.countplot(survey['Music'],ax=axis1, palette = 'hls')
# axis1.set_xlim(-1,5.5)
# axis1.set_ylabel('')
# plt.show()
#
#Music Correlations
# sns.heatmap(music.corr(), annot = True)
# plt.show()
#


# #Logistic Regression
#
# ind_var = music
#
# count = 1
#
# #Change the "math" variables to one of the grouped variables above to see the alternative tests!
# for i in math.columns:
#     y = math.fillna(math.mean(), inplace=True)[i]
#     X_train, X_test, y_train, y_test = train_test_split(ind_var, y, test_size=0.4, random_state=50)
#
#     log = LogisticRegression()
#     log.fit(X_train, y_train)
#     prediction = log.predict(X_test)
#
#     print('{}. {}'.format(count, i.upper()))
#     print('  Matrix:\n', confusion_matrix(y_test, prediction), '\n')
#     print('  Classification :\n', classification_report(y_test, prediction), '\n')
#     count += 1
#
# independentVar = pd.DataFrame({'features': ind_var.columns, 'impacts': log.coef_[0]})
# independentVar = independentVar.sort_values('impacts', ascending=False)
#
# fig, ax1 = plt.subplots(1,1, figsize=(40,7))
# _ = sns.barplot(x=independentVar.features, y=independentVar.impacts, ax=ax1)
# _ = ax1.set_title('Music Correlations', size=25)
# _ = ax1.set_xticklabels(labels=independentVar.features, size=20, rotation=90)
# _ = ax1.set_ylabel('Relation', size=25)
# plt.show()

# #PCA
# ind_var = music
# scale = StandardScaler()
# scale.fit(ind_var)
# fitted_data = scale.transform(ind_var)
# pca = PCA(n_components=2)
# pca.fit(fitted_data)
# x_pca = pca.transform(fitted_data)
#
# fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
#
# #Change the "math" variables to one of the grouped variables above to see the alternative tests!
#
# num = 0
# for i in range(1):
#     for j in range(2):
#         y = math.fillna(math.mean(), inplace=True)[math.columns[num]]
#         axes[i, j].scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap='magma')
#         axes[i, j].set_title('{}'.format(math.columns[num]), fontsize=10)
#
#         num += 1
#         if num == 7: break
#
# plt.show()


# #Random Forest
# ind_var = music
# count = 1
#
# #Change the "math" variables to one of the grouped variables above to see the alternative tests!
#
# for i in math.columns:
#
#     y = math.fillna(spending.mean(),inplace=True)[i]
#     X_train, X_test, y_train, y_test = train_test_split(ind_var, y, test_size=0.4, random_state=50)
#
#     randomForest = RandomForestClassifier(n_estimators=500)
#     randomForest.fit(X_train, y_train)
#     prediction = randomForest.predict(X_test)
#
#     print('{}. {}'.format(count,i.upper()))
#     print('  Matrix:\n', confusion_matrix(y_test, prediction), '\n')
#     print('  Classification:\n', classification_report(y_test, prediction), '\n')
#     count += 1







