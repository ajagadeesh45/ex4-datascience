
# EXNO:4-DS


# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```py
from google.colab import drive
drive.mount('/content/drive')

ls drive/MyDrive/'Colab Notebooks'/
```


# **FEATURE SCALING**
```py
import pandas as pd
import numpy as np
import scipy as stats

df=pd.read_csv('drive/MyDrive/Data Science/bmi.csv')
df.head()
```

![Screenshot 2024-11-28 134830](https://github.com/user-attachments/assets/38b737cc-f0e2-45b5-9c34-cce539f94849)
```py
df.dropna()
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
```py
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']]) 
 df
```
 
![Screenshot 2024-11-28 134847](https://github.com/user-attachments/assets/7f498f9c-af74-46e4-a695-bba30454d3d8)

```py
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```

![Screenshot 2024-11-28 134856](https://github.com/user-attachments/assets/1aaf4a7e-6d28-4588-8ce2-ced8ba28f23d)
```py
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```

![Screenshot 2024-11-28 134904](https://github.com/user-attachments/assets/1ed2c679-e73d-450a-832f-ffb5dccda288)
```py
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![Screenshot 2024-11-28 134911](https://github.com/user-attachments/assets/b52caacb-f2f1-4738-81a7-3ef5dd3996ac)
```py
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```

![Screenshot 2024-11-28 134917](https://github.com/user-attachments/assets/06517dcf-898b-4447-b21a-a28187c92f88)
```py
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv('drive/MyDrive/Data Science/income.csv',na_values=[" ?"])
data
```

![Screenshot 2024-11-28 134932](https://github.com/user-attachments/assets/7b71e752-4f40-464d-802a-305143df3522)

```py
data.isnull().sum()
```

![Screenshot 2024-11-28 134945](https://github.com/user-attachments/assets/5bf87e09-8cea-4b3a-af63-b543ff04373a)
```py
missing=data[data.isnull().any(axis=1)]
missing
```

![Screenshot 2024-11-28 134953](https://github.com/user-attachments/assets/bfc4143f-7426-4e33-a40a-0438b41993cc)
```py
data2 = data.dropna(axis=0)
data2
```

![Screenshot 2024-11-28 135001](https://github.com/user-attachments/assets/55230dbc-c0d7-42d6-afe5-1562677f8233)
```py
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```

![Screenshot 2024-11-28 135009](https://github.com/user-attachments/assets/e1991ca0-a818-41ad-9adb-0a7012991a27)
```py
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```

![Screenshot 2024-11-28 135016](https://github.com/user-attachments/assets/1f17d7cb-0b2e-4041-9906-ea48b14a9fab)
data2

![Screenshot 2024-11-28 135023](https://github.com/user-attachments/assets/c8a3544d-5dfe-48a1-9749-7b1d3a0ac48b)
```py
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```

![Screenshot 2024-11-28 135034](https://github.com/user-attachments/assets/efb8a360-a409-4dee-a963-63e714c0d5ef)
```py
columns_list=list(new_data.columns)

features=list(set(columns_list)-set(['SalStat']))
print(features)

y=new_data['SalStat'].values
print(y)
```

![Screenshot 2024-11-28 135141](https://github.com/user-attachments/assets/1158ce13-6267-420d-ba29-ea4f4df7df24)
```py
x = new_data[features].values
print(x)
```

![Screenshot 2024-11-28 135144](https://github.com/user-attachments/assets/d274100b-3090-4674-8441-2e9fe0d63a92)
```py
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)

prediction = KNN_classifier.predict(test_x)
confusionMmatrix = confusion_matrix(test_y, prediction)
print(confusionMmatrix)
```

![Screenshot 2024-11-28 135551](https://github.com/user-attachments/assets/d078bb3c-8a2e-4652-a9bc-f1e2d4e08d2e)
```py
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)
```

![Screenshot 2024-11-28 135557](https://github.com/user-attachments/assets/acec7d7a-0198-4366-8271-c9f90614882e)
```py
print('Misclassified samples: %d' % (test_y != prediction).sum())
```
![Screenshot 2024-11-28 135602](https://github.com/user-attachments/assets/3ea67e20-ea92-420e-815c-8739c6ca019f)
```py
data.shape
```

![Screenshot 2024-11-28 135606](https://github.com/user-attachments/assets/efaa55aa-33a5-4312-9685-423b58c3da1c)

# **FEATURE SELECTION TECHNIQUES**
```py
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

![Screenshot 2024-11-28 135611](https://github.com/user-attachments/assets/de5ed262-c7f9-4ab9-b0f9-d84128e651c7)
```py
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

![Screenshot 2024-11-28 135614](https://github.com/user-attachments/assets/3f362d76-3d57-48a9-920d-c0dafe31ead8)
```py
chi2, p, _, _ = chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
```

![Screenshot 2024-11-28 135618](https://github.com/user-attachments/assets/4bcb7b45-8ba1-4a79-9698-cabdb12738e4)
```py
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target' :[0,1,1,0,1]
}
```
```py
df=pd.DataFrame(data)
X=df[['Feature1','Feature3']]
y=df['Target']
```
```py
selector=SelectKBest(score_func=mutual_info_classif, k=1)
X_new = selector.fit_transform (X,y)
selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

![Screenshot 2024-11-28 135623](https://github.com/user-attachments/assets/6ae1c1fa-eb96-4860-a82c-bd0ed9e5e2ba)

# RESULT:
       Thus, the data was successfully processed and validated using Feature Scaling and Selection techniques.
