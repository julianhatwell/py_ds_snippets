import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from sklearn.svm import SVC

def df_resample(df1, df2):
    return(resample(df1, 
             replace=True,     # sample with replacement
             n_samples=len(df2), # to match majority class
             random_state=123)) # reproducible results
    
def xy_split(df):
    # Separate input features (X) and target variable (y)
    y = df.balance
    X = df.drop('balance', axis=1)
    return(X, y)

def class_imbalance_demo():
    # In this demo, there is not the usual train-test split or CV
    # The idea is just to demonstrate fixes for class imbalance
    
    # Read dataset
    df = pd.read_csv('balance-scale.data',
                     names=['balance', 'var1', 'var2', 'var3', 'var4'])
    
    # Display example observations
    df.head()
    
    print('original value counts')
    print(df['balance'].value_counts())
        
    df['balance'] = [1 if b=='B' else 0 for b in df.balance]    
    print('transformed to binary value counts')
    print(df['balance'].value_counts())
    
    # Separate input features (X) and target variable (y)
    X, y = xy_split(df)
     
    # Train model
    clf_0 = LogisticRegression(solver='lbfgs').fit(X, y)
     
    # Predict on training set
    pred_y_0 = clf_0.predict(X)
    
    print('Accuracy Score of naive model:')
    print('logistic regression without class balance')
    print( accuracy_score(pred_y_0, y) )
    print('Unique predicted values only include one class!')
    print( np.unique( pred_y_0 ) )
	
    # Separate majority and minority classes
    df_majority = df[df.balance==0]
    df_minority = df[df.balance==1]
     
    # Upsample minority class
    df_minority_upsampled = resample_df(df1 = df_minority, df2 = df_majority)
     
    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
     
    # Display new class counts
    print('Upsampled data frame')
    print(df_upsampled.balance.value_counts())
    
    # Separate input features (X) and target variable (y)
    X, y = xy_split(df_upsampled)
     
    # Train model
    clf_1 = LogisticRegression(solver='lbfgs').fit(X, y)
     
    # Predict on training set
    pred_y_1 = clf_1.predict(X)
     
    print('Is our model still predicting just one class?')
    print( np.unique( pred_y_1 ) )
    # [0 1]
     
    print('Accuracy score for model trained on upsampled data')
    print( accuracy_score(y, pred_y_1) )
    
    # Downsample majority class
    df_majority_downsampled = df_resample(df1 = df_majority, df2 = df_minority)
     
    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
     
    # Display new class counts
    print(df_downsampled.balance.value_counts())

    # Separate input features (X) and target variable (y)
    X, y = xy_split(df_downsampled)
     
    # Train model
    clf_2 = LogisticRegression(solver='lbfgs').fit(X, y)
     
    # Predict on training set
    pred_y_2 = clf_2.predict(X)
     
    print('Is our model still predicting just one class?')
    print( np.unique( pred_y_2 ) )
    # [0 1]
     
    print('Accuracy score for model trained on downsampled data')
    print( accuracy_score(y, pred_y_2) )
    
    # Try using the AUROC measure
    # Predict class probabilities
    prob_y_2 = clf_2.predict_proba(X)
 
    # Keep only the positive class
    prob_y_2 = [p[1] for p in prob_y_2]
    
    print('AUROC for positive class scores on downsampled data set')
    print( roc_auc_score(y, prob_y_2) )
    
    # take a look at same for the naive, unbalanced model
    prob_y_0 = clf_0.predict_proba(X)
    prob_y_0 = [p[1] for p in prob_y_0]
    
    print('AUROC for positive class scores on unbalanced data set')
    print( roc_auc_score(y, prob_y_0) )
    
    # using unbalanced data set, and a linear kernal penalized SVM
    X, y = xy_split(df)
    
    # Train model
    clf_3 = SVC(kernel='linear', 
                class_weight='balanced', # penalize
                probability=True)
     
    clf_3.fit(X, y)
     
    # Predict on training set
    pred_y_3 = clf_3.predict(X)
     
    print('Is our model still predicting just one class?')
    print( np.unique( pred_y_3 ) )
    # [0 1]
     
    print('Accuracy score for linear SVM model trained on unbalanced data')
    print( accuracy_score(y, pred_y_3) )
         
    # What about AUROC?
    prob_y_3 = clf_3.predict_proba(X)
    prob_y_3 = [p[1] for p in prob_y_3]
    print('AUROC for positive class scores on unbalanced data set')
    print( roc_auc_score(y, prob_y_3) )
    
    # using unbalanced data set, and a random forest
    # Train model
    clf_4 = RandomForestClassifier(n_estimators = 100)
    clf_4.fit(X, y)
     
    # Predict on training set
    pred_y_4 = clf_4.predict(X)
     
    print('Is our model still predicting just one class?')
    print( np.unique( pred_y_4 ) )
    # [0 1]
     
    print('Accuracy score for random forest model trained on unbalanced data')
    print( accuracy_score(y, pred_y_4) )
    # 0.9744
     
    # What about AUROC?
    prob_y_4 = clf_4.predict_proba(X)
    prob_y_4 = [p[1] for p in prob_y_4]
    print('AUROC for positive class scores on unbalanced data set')
    print( roc_auc_score(y, prob_y_4) )
    
    print('what voodoo is this???)