import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('./dataIN/kidney_disease.csv',index_col=0)


quality = ['rbc','pc','pcc','ba','pcv','wc','rc','htn'
           ,'dm','cad','appet','pe','ane','classification']

df = df.dropna(subset=['htn','dm','cad','appet','pe','ane'])
print(df.info())

for it in quality:
    encoder = LabelEncoder()
    df[it] = encoder.fit_transform(df[it])
    
df_corr = pd.DataFrame(df.corr())
df_corr.to_csv('conf_ckd.csv')