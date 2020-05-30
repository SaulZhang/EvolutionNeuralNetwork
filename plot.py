import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import cm
from scipy import stats
from pandas import set_option

bcdf = pd.read_csv('../data/data.csv')
diagnosis_coder = {'M':1, 'B':0}
bcdf.diagnosis = bcdf.diagnosis.map(diagnosis_coder)
bcdf.drop(['id','Unnamed: 32'], axis = 1, inplace = True)
diagnosis = bcdf.diagnosis
bcdf.drop('diagnosis', axis = 1, inplace = True)
bcdf['Diagnosis'] = diagnosis
bcdf.head()
bcdf.groupby('Diagnosis').mean()
bcdf_n = bcdf[bcdf['Diagnosis'] == 0]
bcdf_y = bcdf[bcdf['Diagnosis'] == 1]
features_means =list(bcdf.columns[0:10])
outcome_count = bcdf.Diagnosis.value_counts()
outcome_count = pd.Series(outcome_count)
outcome_count = pd.DataFrame(outcome_count)
outcome_count.index = ['Benign', 'Malignant']
outcome_count['Percent'] = 100*outcome_count['Diagnosis']/sum(outcome_count['Diagnosis'])
outcome_count['Percent'] = outcome_count['Percent'].round().astype('int')
print('The Perecentage of tumors classified as \'malignant\' in this data \
    set is: {}'.format(100*float(bcdf.Diagnosis.value_counts()[1])/float((len(bcdf)))))
print('\nA good classifier should therefore outperform blind guessing \
    knowing the proportions i.e. > 62% accuracy')

sns.barplot(x = ['Benign', 'Malignant'], y = 'Diagnosis', data = outcome_count, 
                                                                alpha = .8)
plt.title('Frequency of Diagnostic Outcomes in Dataset')
plt.ylabel('Frequency')
plt.show()
fig = plt.figure()
for i,b in enumerate(list(bcdf.columns[0:10])):
    i +=1
    ax = fig.add_subplot(3,4,i)
    sns.distplot(bcdf_n[b], kde=True, label='Benign')
    sns.distplot(bcdf_y[b], kde=True, label='Malignant')
    ax.set_title(b)
sns.set_style("whitegrid")
plt.tight_layout()
plt.legend()
plt.show()
fig = plt.figure()
for i,b in enumerate(list(bcdf.columns[0:10])):
    i +=1
    ax = fig.add_subplot(3,4,i)
    ax.boxplot([bcdf_n[b], bcdf_y[b]])
    ax.set_title(b)
sns.set_style("whitegrid")
plt.tight_layout()
plt.legend()
plt.show()
sns.heatmap(bcdf.corr())
sns.set_style("whitegrid")
plt.show()
