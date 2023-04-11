import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pickle

# Load the CSV data into a pandas DataFrame
df = pd.read_csv('combined_data_train.csv')

# Convert activity labels to numeric values
label_encoder = LabelEncoder()
df['activity'] = label_encoder.fit_transform(df['activity'])

# Split the dataset into input features and output label
X = df.drop('activity', axis=1)
y = df['activity']

# Replace missing values with mean of each column
imputer = SimpleImputer()
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a multiclass Logistic Regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

# Fit the model on the training data
model.fit(X_train, y_train)

# Evaluate the model on the testing data
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# Save the model as a pickle file
with open('model3.pkl', 'wb') as f:
    pickle.dump(model, f)
