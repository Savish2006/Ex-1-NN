<H3>ENTER YOUR NAME:</H3> <H3>SAVISH R<H3>
<H3>ENTER YOUR REGISTER NO:</H3> <H3> 212224230257<H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

### Import libraries
```PYTHON
import pandas as pd
import numpy as np
import seaborn as sns   # for outlier detection
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
```

### Read the dataset directly
```PYTHON
df = pd.read_csv('Churn_Modelling.csv')
print("First 5 rows of the dataset:")
df.head()
```

### Find missing values
```PYTHON
print(df.isnull().sum())
```

### Identify categorical columns
```PYTHON
categorical_cols = df.select_dtypes(include=['object']).columns
print("\nCategorical columns:", categorical_cols.tolist())
```

### Apply Label Encoding to categorical columns
```PYTHON
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

print("\nData after encoding:")
print(df.head(5))
```
### Handling missing values only for numeric columns
```PYTHON
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col].fillna(df[col].mean().round(1), inplace=True)

df.isnull().sum()
```

### Detect Outliers (example using seaborn)
```PYTHON
print("\nDetecting outliers (example: CreditScore column):")
sns.boxplot(x=df['CreditScore'])
```

### Example statistics for 'CreditScore'
```PYTHON
print("\nStatistics for 'CreditScore':")
df['CreditScore'].describe()
```

### Splitting features (X) and labels (y)
```PYTHON
X = df.drop('Exited', axis=1).values  # Features (drop target column)
y = df['Exited'].values   

print("\nFeature Matrix (X):")
print(X)
print("\nLabel Vector (y):")
print(y)
```
### Normalizing the features
```PYTHON
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
```

### First 5 rows after normalization
```PYTHON
pd.DataFrame(X_normalized, columns=df.columns[:-1]).head()
```

### Splitting into Training and Testing Sets
```PYTHON
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.2, random_state=42
)

print("\nShapes of Training and Testing sets:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
```


## OUTPUT:

<img width="1270" height="250" alt="image" src="https://github.com/user-attachments/assets/2c158cd4-0b15-47f5-b396-e4c65cc69b7d" />



<img width="332" height="342" alt="image" src="https://github.com/user-attachments/assets/63f6927e-7a06-4bb8-9d68-09cb2b1ddc4e" />




<img width="592" height="80" alt="image" src="https://github.com/user-attachments/assets/14c1d06f-f56c-40e1-9fb5-31688b39e22e" />




<img width="820" height="491" alt="image" src="https://github.com/user-attachments/assets/3eb2dbd4-4e2a-462b-9272-8db452bf544e" />




<img width="376" height="332" alt="image" src="https://github.com/user-attachments/assets/d488b1f9-09b7-4778-8581-1e696c3e64e1" />




<img width="842" height="642" alt="image" src="https://github.com/user-attachments/assets/abcf38ce-c49c-4c0e-8452-e3a99526b50b" />




<img width="450" height="245" alt="image" src="https://github.com/user-attachments/assets/d736026a-7ae9-49d7-885e-39c3540410c9" />



<img width="670" height="391" alt="image" src="https://github.com/user-attachments/assets/a0b66785-f7ba-4b68-bf67-6d9ed0cd7c27" />

<img width="1277" height="241" alt="image" src="https://github.com/user-attachments/assets/4737cf11-60ae-48c4-b0b9-61f13683e4b3" />

<img width="483" height="135" alt="image" src="https://github.com/user-attachments/assets/9010f1c4-0d05-448c-ad1d-842a6275ec13" />




## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


