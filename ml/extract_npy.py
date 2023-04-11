import pandas as pd
import numpy as np

# Load the CSV data into a pandas DataFrame
df = pd.read_csv('combined_data_train.csv')

# Split the dataset into input features and output label
X = df.drop('activity', axis=1).values
y = df['activity'].values

# Save the numpy arrays as files
np.save('X_train.npy', X)
np.save('y_train.npy', y)
