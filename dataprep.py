<<<<<<< HEAD
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('TAMU_FINAL_SUBSET.csv')
#dataset = dataset.iloc[ :, 2:8]
#g = sns.PairGrid(dataset)
#g.map(plt.scatter)

dataset.describe()
uniques = pd.DataFrame(dataset.apply(lambda x: x.nunique()))
uniques['Index']= uniques.index
uniques = uniques.rename(columns={0:'distinct'})
uniques =uniques.loc[uniques['distinct'] == 1]

dataset.drop(['FLAG_DIAB_2015','FLAG_PREDIABETES_2015','FLAG_CANCER_ACTIVE_2014','AMM_2014',
                    'FLAG_CANCER_ACTIVE_2015', 'FLAG_ESRD_2015', 'FLAG_DIAB_2014','FLAG_ESRD_2014',
                    'AMM_ACUTE_GAP_2014','CDC_LDL100_GAP_2015','Decile_struggle_Med_lang'], axis=1, inplace=True)

dataset =dataset.dropna(subset = ['ESRD_IND'])
dataset =dataset.dropna(subset = ['PCP_ASSIGNMENT'])
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
dataset["SEX_CD"] = labelencoder.fit_transform(dataset["SEX_CD"])#(Male =1)
dataset["ESRD_IND"] = labelencoder.fit_transform(dataset["ESRD_IND"])#N=0
dataset["HOSPICE_IND"] = labelencoder.fit_transform(dataset["HOSPICE_IND"])#N=0
dataset["DUAL"] = labelencoder.fit_transform(dataset["DUAL"])#N=0
dataset["INSTITUTIONAL"] = labelencoder.fit_transform(dataset["INSTITUTIONAL"])#N=0
dataset["LIS"] = labelencoder.fit_transform(dataset["LIS"])#N=0
dataset["MCO_HLVL_PLAN_CD"] = labelencoder.fit_transform(dataset["MCO_HLVL_PLAN_CD"])#MAPD=1

dataset = pd.get_dummies(dataset, columns=["PCP_ASSIGNMENT"], drop_first = True)#dropped Attributed
dataset = pd.get_dummies(dataset, columns=["MCO_PROD_TYPE_CD"], drop_first = True)#dropped HMO
dataset = dataset.reindex_axis(list([a for a in dataset.columns if a != 'MAJOR_GEOGRAPHY'] +['MAJOR_GEOGRAPHY']), axis=1)
dataset = dataset.reindex_axis(list([a for a in dataset.columns if a != 'MINOR_GEOGRAPHY'] +['MINOR_GEOGRAPHY']), axis=1)
dataset = dataset.reindex_axis(list([a for a in dataset.columns if a != 'Dwelling_Type'] +['Dwelling_Type']), axis=1)
dataset =dataset.drop(['MAJOR_GEOGRAPHY','MINOR_GEOGRAPHY','Dwelling_Type'], axis=1)
dataset1 = dataset
dataset2 =corr_df(dataset1, 0.75)
#dataset.to_csv('dataset.csv', sep=',')
X = dataset.iloc[:, 2:870].values                
y = dataset.iloc[:, 0].values

                
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(X,y)

from sklearn import linear_model
clf = linear_model.LinearRegression()
clf.fit([[getattr(dataset, 'x%d' % i) for i in range(2, 870)] for dataset in texts],
        [dataset.ADMISSIONS for dataset in texts])
# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred = (y_pred>0.5, 1,0)
y_pred.describe()
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Building the optimal model using backward Elimination
import statsmodels.formula.api as sm
X=np.append(arr= np.ones((39499, 1)).astype(int), values = X, axis=1)
X_opt = X[:, 0:]
regressor_OLS = sm.OLS(endog=y, exog= X_opt).fit()
regressor_OLS.summary()

#Finding Correlation between the variables
corr_matrix = dataset.corr().abs()
high_corr_var=np.where(corr_matrix>0.8)
high_corr_var=[(corr_matrix.columns[2:100],corr_matrix.columns[0]) for x,y in zip(*corr_matrix) if x!=y and x<y]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#cols = dataset.columns.tolist()
#cols = cols[0:2]+cols[-1:-7] + cols[2:-7]
#dataset = dataset[cols]

 


#dataset1[:,13]
#dataset.dtypes
#PCP_ASSIGNMENT                    object-10
#DUAL                              object-11
#INSTITUTIONAL                     object-12
#LIS                               object-13
#MAJOR_GEOGRAPHY                   object-14- many
#MINOR_GEOGRAPHY                   object-15- many
#MCO_HLVL_PLAN_CD                  object-16
#MCO_PROD_TYPE_CD                  object-17
#ESRD_IND                          object-5
#HOSPICE_IND                       object-6
#SEX_CD                            object-3
#Dwelling_Type                     object-928- many


#To find correlations
#dataset.corr(method='pearson',  min_periods=1)
c = dataset.corr().abs()
s = c.unstack()
so = s.order(kind="quicksort")
print (so[-10:-1])

print("Correlation Matrix")
print(df.corr())
print()

df= dataset
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=10):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
=======
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('TAMU_FINAL_SUBSET.csv')
#dataset = dataset.iloc[ :, 2:8]
#g = sns.PairGrid(dataset)
#g.map(plt.scatter)

dataset.describe()
uniques = pd.DataFrame(dataset.apply(lambda x: x.nunique()))
uniques['Index']= uniques.index
uniques = uniques.rename(columns={0:'distinct'})
uniques =uniques.loc[uniques['distinct'] == 1]

dataset.drop(['FLAG_DIAB_2015','FLAG_PREDIABETES_2015','FLAG_CANCER_ACTIVE_2014','AMM_2014',
                    'FLAG_CANCER_ACTIVE_2015', 'FLAG_ESRD_2015', 'FLAG_DIAB_2014','FLAG_ESRD_2014',
                    'AMM_ACUTE_GAP_2014','CDC_LDL100_GAP_2015','Decile_struggle_Med_lang'], axis=1, inplace=True)

dataset =dataset.dropna(subset = ['ESRD_IND'])
dataset =dataset.dropna(subset = ['PCP_ASSIGNMENT'])
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
dataset["SEX_CD"] = labelencoder.fit_transform(dataset["SEX_CD"])#(Male =1)
dataset["ESRD_IND"] = labelencoder.fit_transform(dataset["ESRD_IND"])#N=0
dataset["HOSPICE_IND"] = labelencoder.fit_transform(dataset["HOSPICE_IND"])#N=0
dataset["DUAL"] = labelencoder.fit_transform(dataset["DUAL"])#N=0
dataset["INSTITUTIONAL"] = labelencoder.fit_transform(dataset["INSTITUTIONAL"])#N=0
dataset["LIS"] = labelencoder.fit_transform(dataset["LIS"])#N=0
dataset["MCO_HLVL_PLAN_CD"] = labelencoder.fit_transform(dataset["MCO_HLVL_PLAN_CD"])#MAPD=1

dataset = pd.get_dummies(dataset, columns=["PCP_ASSIGNMENT"], drop_first = True)#dropped Attributed
dataset = pd.get_dummies(dataset, columns=["MCO_PROD_TYPE_CD"], drop_first = True)#dropped HMO
dataset = dataset.reindex_axis(list([a for a in dataset.columns if a != 'MAJOR_GEOGRAPHY'] +['MAJOR_GEOGRAPHY']), axis=1)
dataset = dataset.reindex_axis(list([a for a in dataset.columns if a != 'MINOR_GEOGRAPHY'] +['MINOR_GEOGRAPHY']), axis=1)
dataset = dataset.reindex_axis(list([a for a in dataset.columns if a != 'Dwelling_Type'] +['Dwelling_Type']), axis=1)
dataset =dataset.drop(['MAJOR_GEOGRAPHY','MINOR_GEOGRAPHY','Dwelling_Type'], axis=1)
dataset1 = dataset
dataset2 =corr_df(dataset1, 0.75)
#dataset.to_csv('dataset.csv', sep=',')
X = dataset.iloc[:, 2:870].values                
y = dataset.iloc[:, 0].values

                
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(X,y)

from sklearn import linear_model
clf = linear_model.LinearRegression()
clf.fit([[getattr(dataset, 'x%d' % i) for i in range(2, 870)] for dataset in texts],
        [dataset.ADMISSIONS for dataset in texts])
# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred = (y_pred>0.5, 1,0)
y_pred.describe()
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Building the optimal model using backward Elimination
import statsmodels.formula.api as sm
X=np.append(arr= np.ones((39499, 1)).astype(int), values = X, axis=1)
X_opt = X[:, 0:]
regressor_OLS = sm.OLS(endog=y, exog= X_opt).fit()
regressor_OLS.summary()

#Finding Correlation between the variables
corr_matrix = dataset.corr().abs()
high_corr_var=np.where(corr_matrix>0.8)
high_corr_var=[(corr_matrix.columns[2:100],corr_matrix.columns[0]) for x,y in zip(*corr_matrix) if x!=y and x<y]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#cols = dataset.columns.tolist()
#cols = cols[0:2]+cols[-1:-7] + cols[2:-7]
#dataset = dataset[cols]

 


#dataset1[:,13]
#dataset.dtypes
#PCP_ASSIGNMENT                    object-10
#DUAL                              object-11
#INSTITUTIONAL                     object-12
#LIS                               object-13
#MAJOR_GEOGRAPHY                   object-14- many
#MINOR_GEOGRAPHY                   object-15- many
#MCO_HLVL_PLAN_CD                  object-16
#MCO_PROD_TYPE_CD                  object-17
#ESRD_IND                          object-5
#HOSPICE_IND                       object-6
#SEX_CD                            object-3
#Dwelling_Type                     object-928- many


#To find correlations
#dataset.corr(method='pearson',  min_periods=1)
c = dataset.corr().abs()
s = c.unstack()
so = s.order(kind="quicksort")
print (so[-10:-1])

print("Correlation Matrix")
print(df.corr())
print()

df= dataset
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=10):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
>>>>>>> 5ced3a02ca7b34cc82908ada6f572f4f4a16c863
print(get_top_abs_correlations(df, 10))