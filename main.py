import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('employee_satisfaction_index.csv')
dataset = dataset.drop(['record_number', 'emp_id'], axis=1)
lista_kolumn_ze_stringami = ['Dept', 'location', 'education', 'recruitment_type']

for x in lista_kolumn_ze_stringami:
    dataset[x] = pd.factorize(dataset[x])[0]



train_dataset = dataset.sample(frac=0.8, random_state=0)

X = dataset.drop('satisfied', axis=1)
y = dataset['satisfied']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, random_state=42
)

test_dataset = dataset.drop(train_dataset.index)

# train_features = train_dataset.copy().values
# train_features = np.array(train_features)
# train_features = tf.stack(train_features)
#
# test_features = test_dataset.copy().values
# test_features = np.array(test_features)
# test_features = tf.stack(test_features)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(11,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.03),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)
history = model.fit(X_train, y_train, epochs=200)
