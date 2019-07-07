import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd


# finding distance in n dimension here n is 784
def dist(a,b):
    return sqrt(  np.sum(  (a-b)**2 ) )


# K-nearest neighbors algorithm
def kNN(X,Y,tp,k=5):
    tp = np.array(tp)
    m = len(X[0])
    vals = []
    for i in range(m):
        d = dist( tp , X[i] )
        vals.append( (d,Y[i][0]  ) )
        
    vals.sort()
    vals = np.array( vals[:k] )
    b = np.unique( vals[:,1] , return_counts =True )
    id = np.argmax( b[1] )
    pred = b[0][id]
    return int(pred)
# Training dataset
df  = pd.read_csv( "train.csv" )
data = df.values
X = data[:,1:]
Y = data[:,:1]


# Test dataset
df2 = pd.read_csv("test.csv")
d = df2.values

# There are total 28000 test data available to select any of them
# Enter any number between 0 t0 27999
i = int(input())
ts = d[i][:]

pre = kNN( X,Y,ts )
print("prediction is : "  ,pre    )


plt.imshow( ts.reshape(28,28) , cmap = 'gray'  )

