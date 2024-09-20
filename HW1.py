import numpy as np 
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.feature_selection import f_classif

path = "/home/afonso215/Aprendizagem/diabetes.arff"

data = loadarff(path)
df = pd.DataFrame(data[0])



