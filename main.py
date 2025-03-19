# coding: utf-8
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# TODO: Try removing everything but property sqft & price
property_train = pd.read_csv(
    "./NY-House-Dataset.csv",
    # THIS SETS THE NAME OF THE COLUMN NOT FILTERS
    # names=["PRICE", "BEDS", "BATH", "PROPERTYSQFT", "LATITUDE", "LONGITUDE"]
)

property_train = property_train.filter(items=[
    "PRICE",
    # "BEDS",
    # "BATH",
    "PROPERTYSQFT",
    # "LATITUDE",
    # "LONGITUDE",
])

# TODO: Need to normalize the data
# And then multiply the output by the max price ( I THINK )

property_train.head()
property_features = property_train.copy()
property_labels = property_features.pop("PRICE")
property_features = np.array(property_features)
property_features
property_model = tf.keras.Sequential([
  layers.Dense(64, activation='relu'),
  layers.Dense(1)
])

property_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.keras.optimizers.Adam())
property_model.fit(property_features, property_labels, epochs=10)
