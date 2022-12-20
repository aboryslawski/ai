# for dependecies run pip3 install -r requirements.txt
import pandas as pd
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
df = pd.read_csv('employee_satisfaction_index.csv')
cols_to_norm = ['age', 'Dept', 'location', 'education', 'recruitment_type', 'job_level', 'rating', 'onsite', 'awards', 'certifications', 'salary']
lista_kolumn_ze_stringami = ['Dept', 'location', 'education', 'recruitment_type']
#print(df)

for x in lista_kolumn_ze_stringami:
    df[x] = pd.factorize(df[x])[0]
#df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
#print(df)

feat_columns = []
for x in cols_to_norm:
    feat_columns.append(tf.feature_column.categorical_column_with_hash_bucket(x, hash_bucket_size=random.randrange(1, 60), dtype=tf.dtypes.int64))
print(df)
X_data = df.drop('satisfied', axis=1)
labels = df['satisfied']
X_train, X_test, y_train, y_test = train_test_split(X_data, labels, test_size=0.2, random_state=101)
#input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)
input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)
model=tf.estimator.LinearClassifier(feature_columns=feat_columns, n_classes=2)
model.train(input_fn=input_func, steps=1000)

