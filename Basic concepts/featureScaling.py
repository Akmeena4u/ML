'''Feature scaling is a technique used in machine learning to standardize the range of features or independent variables of a dataset. 
This is important because many machine learning algorithms assume that all features are on the same scale,
and features with larger scales can dominate over features with smaller scales, leading to biased results.

Here's how you can perform feature scaling in Python using some common libraries:

'''
#---------------------------------------------------------------------------------------------
#1--StandardScaler from sklearn.preprocessing

from sklearn.preprocessing import StandardScaler

# create a StandardScaler object
scaler = StandardScaler()

# fit the scaler to the data
scaler.fit(X)

# transform the data using the scaler
X_scaled = scaler.transform(X)

#-------------------------------------------------------------------------------------------------
#2--MinMaxScaler from sklearn.preprocessing

from sklearn.preprocessing import MinMaxScaler

# create a MinMaxScaler object
scaler = MinMaxScaler()

# fit the scaler to the data
scaler.fit(X)

# transform the data using the scaler
X_scaled = scaler.transform(X)

#-----------------------------------------------------------------------------------------------------
#3---RobustScaler from sklearn.preprocessing
from sklearn.preprocessing import RobustScaler

# create a RobustScaler object
scaler = RobustScaler()

# fit the scaler to the data
scaler.fit(X)

# transform the data using the scaler
X_scaled = scaler.transform(X)


