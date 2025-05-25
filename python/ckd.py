
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,recall_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




df_mice = pd.read_csv('./dataIN/ckd_fill/mice.csv')
df_knn = pd.read_csv('./dataIN/ckd_fill/knn.csv',index_col=0)

results = {
    "model":[],
    "fill_method":[],
    "accuracy":[],
    "precision":[],
    "recall":[],
    "f1":[]
}

def evaluate(df,fill_method):
    models_name = [
        "RANDOM_FOREST",
        "XGBOOST",
        "LOGISTIC",
    ]
    df[df['classification']==1]=0
    df[df['classification']==2]=1
    
    X_train,X_test,y_train,y_test = train_test_split(df.drop(columns=['classification']),df['classification'],random_state=42,train_size=0.7)

    models = [
        RandomForestClassifier(max_depth=5),
        XGBClassifier(),
        LogisticRegression()
    ]

    for model,label in zip(models,models_name):
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)

        print(f"Model: {label}")
        print(f"Acurracy: {accuracy_score(y_test,y_pred)}")
        sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)
        plt.title(model)
        plt.show()
        
        results["model"].append(label)
        results["fill_method"].append(fill_method)
        results['accuracy'].append(accuracy_score(y_test,y_pred))
        results['precision'].append(precision_score(y_test,y_pred))
        results['recall'].append(recall_score(y_test,y_pred))
        results['f1'].append(f1_score(y_test,y_pred))

evaluate(df_knn,"knn")
evaluate(df_mice,"mice")

print(results)

df_result = pd.DataFrame(results)
df_result.to_csv('result.csv')
