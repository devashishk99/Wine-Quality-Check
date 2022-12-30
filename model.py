import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

wine_dataset = pd.read_csv('winequality-red.csv')

X = wine_dataset.drop('quality',axis=1)
Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

model = RandomForestClassifier()
model.fit(X_train, Y_train)

X_test_prediction = model.predict(X_test)

joblib.dump(model, "rf_model.sav")