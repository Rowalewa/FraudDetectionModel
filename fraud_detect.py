import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Loading both files and combining
df1 = pd.read_csv('Fraudulent_E-Commerce_Transaction_Data.csv')
df2 = pd.read_csv('Fraudulent_E-Commerce_Transaction_Data_2.csv')
df = pd.DataFrame(pd.concat([df1, df2], ignore_index=True))

print(df.shape)  # ~1.5 million rows, 16 columns
print(df.head())
print(df.isnull().sum()) # no nulls - clean dataset

# Target Variable - 95% legitimate, 5% fraud
print(df['Is Fraudulent'].value_counts())
print(df['Is Fraudulent'].value_counts(normalize=True).round(3))

# Data Exploration
# Finding rows with negative ages and under 18. This is data quality problem.
print(df['Customer Age'].describe())
print('Negative Ages: ', len(df[df['Customer Age'] < 0]))
print('Under 18: ', len(df[df['Customer Age'] < 18]))

'''Decision - Removing under 18 and under 0. Removing the data quality problems.'''
# Always document decisions like using docstrings triple quotes. 
# In portfolio writeup or interview, being able to say i removed customers under 18 as a business rule - in most finacial and e-commerce settings, minors are not valid account holders.
df = df[df['Customer Age'] >= 18]
print(df.shape)

'''Explore what drives fraud'''
# Fraud rate by payment method
fraud_by_payment = df.groupby('Payment Method')['Is Fraudulent'].mean().sort_values(ascending=False)
fraud_by_payment.plot(kind='bar', color = 'coral', figsize=(8,4), grid=True)
plt.title('Fraud Rate by Payment Method')
plt.ylabel('Fraud Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Fraud Rate by product category
fraud_by_category = df.groupby('Product Category')['Is Fraudulent'].mean().sort_values(ascending=False)
fraud_by_category.plot(kind='bar', color = 'steelblue', figsize=(8,4), grid=True)
plt.title('Fraud Rate by Product Category')
plt.ylabel('Fraud Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Fraud Rate by Hour of Day
fraud_by_hour = df.groupby('Transaction Hour')['Is Fraudulent'].mean()
fraud_by_hour.plot(kind='line', color = 'steelblue', figsize=(10,4), marker = 'o')
plt.title('Fraud Rate by Hour of Day')
plt.ylabel('Fraud Rate')
plt.xlabel('Hour')
plt.tight_layout()
plt.show()

# Transaction Amount - Fraud vs legitimate
df.boxplot(column='Transaction Amount', by = 'Is Fraudulent', figsize=(8,5))
plt.title('Transaction Amount by Fraud Status')
plt.suptitle('')
plt.show()

'''Looking for patterns before modelling. Are fraudulent transactions higher value? Are certain payment methods riskier? are there peak fraud hours? 
These are insights a Fraud Team would actually use.'''

'''Prepare the data. ML models need numbers. Drop identifier columns and encode categorical ones.'''
# Drop columns not useful for modelling - IDs and addresses
drop_cols = ['Transaction ID', 'Customer ID', 'IP Address', 'Shipping Address', 'Billing Address', 'Customer Location', 'Transaction Date']
df = df.drop(columns=drop_cols, errors='ignore')

# One hot encode categorical columns
df = pd.get_dummies(df, columns=['Payment Method', 'Product Category', 'Device Used'], drop_first=True)

print(df.shape)
print(df.head())

''' Training the model. Using Random Forest with class_weight='balanced' to handle the class imbalance. 1.5 millions may take a while depending on machine. '''
# Optional: sample for speed during development
# df = df.sample(n=200_000, random_state=42)

X = df.drop('Is Fraudulent', axis=1)
y = df['Is Fraudulent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1 # Use all CPU cores to speed things up
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

''' Evaluate Properly '''
# Classification Report
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legitimate', 'Fraud'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

''' Read the classification report carefully.
Recall for fraud - what percentage of actual fraud did you catch?
Precision for fraud- when you flagged something as fraud, how often were you right?
A false negative (missed fraud) costs mone. A false positive (blocking a legitimate transaction) costs a customer.  The business decides which is worse.
Usually financial instiutions as well as security (i.e Airport security) prefer to have false positives rather than false negatives. Better to be safe than sorry.
'''

'''Find the most important features - One of the best things about Random Forest is feature importance - it tells you which columns matter most for predicting fraud.'''
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)

top_features.plot(kind='barh', color = 'steelblue', figsize=(8,5))
plt.title('Top 10 Features for Fraud Detection.')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

'''🔝This is what you show in a presentation.
Not the model internals - but "In this run, transaction amount, account age and transaction hour appeared among the stringest predictors"
'''