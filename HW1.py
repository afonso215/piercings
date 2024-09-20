import numpy as np 
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.feature_selection import f_classif

path = "diabetes.arff"

data = loadarff(path)
df = pd.DataFrame(data[0])

df['Outcome'] = df['Outcome'].str.decode('utf-8')

X = df.drop('Outcome', axis=1)
y = df['Outcome']

fimportance = f_classif(X, y)

