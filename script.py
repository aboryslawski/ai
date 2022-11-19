import pandas as pd
import tensorflow as tf
df = pd.read_csv('employee_satisfaction_index.csv')
cols_to_norm = ['age', 'Dept', 'location', 'education', 'recruitment_type', 'job_level', 'rating', 'onsite', 'awards', 'certifications', 'salary']
lista_kolumn_ze_stringami = ['Dept', 'location', 'education', 'recruitment_type']
#print (df.apply(lambda x: pd.factorize(x)[0] if x[0] in lista_kolumn_ze_stringami else x[0]))
for x in lista_kolumn_ze_stringami:
    df[x] = pd.factorize(df[x])[0]
df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
print(df)
