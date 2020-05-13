import pandas as pd
import numpy as np

#data_frame = pd.read_csv("1684to2214.csv")

def get_proper_featurevector(dataframe_feature):
    x1 = dataframe_feature
    values_x1 = x1.split(",")
    zero = values_x1[0].split("[")
    floatzero = float(zero[1])
    last = values[-1].split("]")
    floatlast = float(last[0])
    fvector = [float(i) for i in values[1:-1]]
    fvector.insert(0,floatzero)
    fvector.append(floatlast)
    return fvector

#for x in range(len(data_frame)):
#    feature_vector = get_proper_featurevector(data_frame['Feature'][x])
#    print(len(feature_vector))
#    break