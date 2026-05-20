Fraud Detection using machine learning. Can we predict which e-commerce transactions are fraudulent?
This is one of the most valuable ML applications in the real world. Every online retailer, payment processor and bank runs models like this. And it is a great portfolio project because it teaches you: what to do when your data has real problems baked in.

Why fraud detection is a different kind of ML problem?
Most ML tutorials tell you to maximise model accuracy. Fraud detection is different.
Fraud is rare. In most real datasets, 1-3% of transactions are fraudulent. If your model predicts “not fraud” for every single row, it will be 95-99% accurate — and completely useless.

So use of different metrics:

Recall — of all actual fraud cases, how many did your model catch?
Precision — of all cases your model flagged as fraud, how many were actually fraud?
F1 score — the balance between the two

A model with 60% recall means 40% of fraud slipped through undetected. That is money leaving the business.
The dataset:
Fraudulent E-Commerce Transactions dataset from Kaggle.
Download link: https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions
1.4 million transactions. 16 columns. Real e-commerce context — payment methods, product categories, device types, customer demographics.

Key columns used:
- [ ] Transaction Amount — value of the transaction
- [ ] Transaction Date — when it happened
- [ ] Payment Method — credit card, debit card, bank transfer, PayPal
- [ ] Product Category — electronics, clothing, home & garden, toys & games, health & beauty
- [ ] Customer Age — age of the customer (watch this one closely.
- [ ] Device Used — mobile, desktop, tablet
- [ ] Account Age Days — how old the customer account is
- [ ] Transaction Hour — hour of day the transaction occurred (already in the dataset)

Is Fraudulent — our target variable (1 = fraud, 0 = legitimate)

Overview - 1.5 million e-commerce transactions across two files, goal is to predict  fraudulent ones before payment is processed.
Approach - combined two files, cleaned invalid ages, explored fraud patterns by payment method, category and hour, built a Random Forest classifier with balanced class weights, evaluated on recall and f1 rather than currency.
Findings - fraud rate of 5%, strongest predictors are transaction amount, account age and transaction hour. 
Fill in your actual recall score. Certain payment methods and payment categories show higher fraud rates.
Limitations - no real-time element, model would need retraining as fraud patterns evolve, no cost-benefit analysis of false positives vs false negatives included.
