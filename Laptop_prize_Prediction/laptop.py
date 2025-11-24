import numpy as np
import pandas as pd
from warnings import filterwarnings
filterwarnings('ignore')
df=pd.read_csv(r"C:\Users\lenovo\Downloads\laptop_data.csv")
df.head(1)
df.shape
df.info()
df.duplicated().sum()
df.isnull().sum()
df.describe()
df.columns
df.drop(columns=['Unnamed: 0'], inplace=True)
df.head()
df['Ram']=df['Ram'].str.replace('GB','')
df['Weight']=df['Weight'].str.replace('kg','')
df.head(1)
df['Ram']=df['Ram'].astype('int32')
df['Weight']=df['Weight'].astype('float32')
df.info()
import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(df['Price'])
df['Company'].value_counts().plot(kind='bar')
sns.barplot(x=df['Company'], y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()
df['TypeName'].value_counts().plot(kind='bar')
sns.barplot(x=df['TypeName'], y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()
sns.distplot(df['Inches'])
sns.scatterplot(x=df['Inches'], y=df['Price'])
df['ScreenResolution'].value_counts()
df['Touchscreen']=df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)
df.sample(5)
df['Touchscreen'].value_counts().plot(kind='bar')
sns.barplot(x=df['Touchscreen'],y=df['Price'])
df['Ips']=df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)
df.head()
df['Ips'].value_counts().plot(kind='bar')
sns.barplot(x=df['Ips'], y=df['Price'])
new=df['ScreenResolution'].str.split('x',n=1,expand=True)
df['X_res']=new[0]
df['Y_res']=new[1]
df.head()
df['X_res'] = (
    df['X_res']
    .astype(str)  # Convert all values to string
    .str.replace(',', '', regex=False)  # Remove commas
    .str.findall(r'(\d+\.?\d*)')  # Extract numeric part
    .apply(lambda x: x[0] if x else None)  # Handle empty matches safely
)
df.head()
df['X_res']=df['X_res'].astype('int')
df['Y_res']=df['Y_res'].astype('int')
df.info()
df.corr(numeric_only=True)['Price']
df['ppi'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5/df['Inches']).astype('float')
df.corr(numeric_only=True)['Price']
df.drop(columns=['ScreenResolution'], inplace=True)
df.head()
df.drop(columns=['Inches', 'X_res', 'Y_res'], inplace=True)
df.head()
df['Cpu'].value_counts()
df['Cpu Name'] = df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))
df.head()
def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'
df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)
df.head()
df['Cpu brand'].value_counts().plot(kind='bar')
sns.barplot(x=df['Cpu brand'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()
df.drop(columns=['Cpu','Cpu Name'],inplace=True)
df.head()
df['Ram'].value_counts().plot(kind='bar')
sns.barplot(x=df['Ram'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()
df['Memory'].value_counts()
import pandas as pd  # Import pandas for data manipulation

# --- STEP 1: CLEAN THE 'Memory' COLUMN ---

# Convert all values in the 'Memory' column to strings and remove any ".0" (e.g., "512.0" → "512")
df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)

# Remove "GB" from the memory column (e.g., "512GB" → "512")
df['Memory'] = df['Memory'].str.replace('GB', '', regex=False)

# Replace "TB" with "000" to convert terabytes to gigabytes (e.g., "1TB" → "1000")
df['Memory'] = df['Memory'].str.replace('TB', '000', regex=False)

# --- STEP 2: SPLIT THE MEMORY INTO TWO PARTS (e.g., "256SSD + 1TBHDD") ---

# Split on '+' — creates two columns: 'first' (before '+') and 'second' (after '+')
new = df['Memory'].str.split('+', n=1, expand=True)

# Store the first part in 'first' and remove extra spaces
df['first'] = new[0].str.strip()

# Store the second part in 'second', replace missing values (NaN) with "0", and remove extra spaces
df['second'] = new[1].fillna('0').str.strip()

# --- STEP 3: DETECT STORAGE TYPES IN THE FIRST PART ---

# Create binary columns that indicate which type of storage is present in the first layer
df['Layer1HDD'] = df['first'].apply(lambda x: 1 if 'HDD' in x else 0)
df['Layer1SSD'] = df['first'].apply(lambda x: 1 if 'SSD' in x else 0)
df['Layer1Hybrid'] = df['first'].apply(lambda x: 1 if 'Hybrid' in x else 0)
df['Layer1Flash_Storage'] = df['first'].apply(lambda x: 1 if 'Flash Storage' in x else 0)

# --- STEP 4: DETECT STORAGE TYPES IN THE SECOND PART ---

# Create binary columns for the second layer
df['Layer2HDD'] = df['second'].apply(lambda x: 1 if 'HDD' in x else 0)
df['Layer2SSD'] = df['second'].apply(lambda x: 1 if 'SSD' in x else 0)
df['Layer2Hybrid'] = df['second'].apply(lambda x: 1 if 'Hybrid' in x else 0)
df['Layer2Flash_Storage'] = df['second'].apply(lambda x: 1 if 'Flash Storage' in x else 0)

# --- STEP 5: CLEAN UP NUMERIC PARTS ---

# Remove all non-digit characters (e.g., "512SSD" → "512")
df['first'] = df['first'].str.replace(r'\D', '', regex=True)
df['second'] = df['second'].str.replace(r'\D', '', regex=True)

# Replace empty strings with "0" (to avoid conversion errors)
df['first'] = df['first'].replace('', '0')
df['second'] = df['second'].replace('', '0')

# Convert both columns to integers
df['first'] = df['first'].astype(int)
df['second'] = df['second'].astype(int)

# --- STEP 6: CALCULATE STORAGE SIZE BY TYPE ---

# Multiply the numeric size by the binary indicator for each storage type
# Example: if 'first' = 256 and 'Layer1SSD' = 1 → SSD = 256
df['HDD'] = df['first']*df['Layer1HDD'] + df['second']*df['Layer2HDD']
df['SSD'] = df['first']*df['Layer1SSD'] + df['second']*df['Layer2SSD']
df['Hybrid'] = df['first']*df['Layer1Hybrid'] + df['second']*df['Layer2Hybrid']
df['Flash_Storage'] = df['first']*df['Layer1Flash_Storage'] + df['second']*df['Layer2Flash_Storage']

# --- STEP 7: CLEAN UP TEMPORARY COLUMNS ---

# Remove intermediate columns to keep the DataFrame clean
df.drop(columns=[
    'first', 'second',
    'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid', 'Layer1Flash_Storage',
    'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid', 'Layer2Flash_Storage'
], inplace=True)
df.sample(5)
df.head()
df.drop(columns=['Memory'], inplace=True)
df.head()
df.corr(numeric_only=True)['Price']
df.drop(columns=['Hybrid', 'Flash_Storage'],inplace= True)
df.head()
df['Gpu'].value_counts()
df['Gpu brand']=df['Gpu'].apply(lambda x:x.split()[0])
df.head()
df['Gpu brand'].value_counts()
df=df[df['Gpu brand']!='ARM']
df['Gpu brand'].value_counts()
sns.barplot(x=df['Gpu brand'],y=df['Price'],estimator=np.median)
plt.xticks(rotation='vertical')
plt.show()
df.drop(columns=['Gpu'],inplace=True)
df.head()
df['OpSys'].value_counts()
sns.barplot(x=df['OpSys'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()
def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'
df['os'] = df['OpSys'].apply(cat_os)
df.head()
df.drop(columns=['OpSys'],inplace=True)
sns.barplot(x=df['os'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()
sns.distplot(df['Weight'])
sns.scatterplot(x=df['Weight'],y=df['Price'])
df.corr(numeric_only=True)['Price']
sns.heatmap(df.corr(numeric_only=True))
sns.distplot(np.log(df['Price']))
X = df.drop(columns=['Price'])
y = np.log(df['Price'])
X
y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=2)
X_train
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
#Linear regression
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

step2 = LinearRegression()

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print('R2 score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))
#Ridge Regression
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = Ridge(alpha=10)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))
#Lasso Regression
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = Lasso(alpha=0.001)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))
#KNN
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = KNeighborsRegressor(n_neighbors=3)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))
#Decision Tree

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = DecisionTreeRegressor(max_depth=8)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))
#SVM

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = SVR(kernel='rbf',C=10000,epsilon=0.1)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))
#Random Forest

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))
#ExtraTrees

step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

# Step 2: ExtraTreesRegressor (fixed: removed max_samples)
step2 = ExtraTreesRegressor(
    n_estimators=100,
    random_state=3,
    max_features=0.75,
    max_depth=15
)

# Create Pipeline
pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# Fit model
pipe.fit(X_train, y_train)

# Predict
y_pred = pipe.predict(X_test)

# Evaluation
print('R2 score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))
#AdaBoost

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = AdaBoostRegressor(n_estimators=15,learning_rate=1.0)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))
#Gradient Boost

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = GradientBoostingRegressor(n_estimators=500)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))
#XgBoost

#Exporting the Model

import pickle

pickle.dump(df,open('df.pkl','wb'))
pickle.dump(pipe,open('pipe.pkl','wb'))
print(df)
 