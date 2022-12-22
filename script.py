# for dependecies run pip3 install -r requirements.txt
import pandas as pd
import tensorflow as tf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('employee_satisfaction_index.csv')
dataset = dataset.drop(['record_number', 'emp_id'], axis=1)
#cols_to_norm = ['age', 'Dept', 'location', 'education', 'recruitment_type', 'job_level', 'rating', 'onsite', 'awards', 'certifications', 'salary']
lista_kolumn_ze_stringami = ['Dept', 'location', 'education', 'recruitment_type']

for x in lista_kolumn_ze_stringami:
    dataset[x] = pd.factorize(dataset[x])[0]

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features['satisfied']
#test_labels = test_features.pop('satisfied')
#print(train_dataset.describe().transpose()[['mean', 'std']])
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.asarray(train_features).astype('float32'))
#print(normalizer.mean.numpy())
first = np.array(train_features[:1])
satisfied_normalizer = tf.keras.layers.Normalization(input_shape=[1,], axis=None)

#with np.printoptions(precision=2, suppress=True):
#    print('First example:', first)
#    print()
#    print('Normalized:', normalizer(first).numpy())

def build_and_compile_model(norm):
    model = tf.keras.Sequential([
        norm,
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss='mean_absolute_error',
            optimizer=tf.keras.optimizers.Adam(0.001))
    return model

dnn_satisfation_model = build_and_compile_model(satisfied_normalizer)

history = dnn_satisfation_model.fit(
        train_features['satisfied'],
        train_labels,
        validation_split=0.2,
        verbose=0, epochs=100)

x = tf.linspace(0.0, 250, 251)
y = dnn_satisfation_model.predict(x)
print(y)
