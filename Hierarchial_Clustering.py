import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 

#Loading the dataset 
customer_data= pd.read_csv("hierarchical-clustering-with-python-and-scikit-learn-shopping-data.csv")
print(customer_data.shape)

print(customer_data.head(5))

data= customer_data.iloc[:,3:5].values
# print(data)

import scipy.cluster.hierarchy as shc 
plt.figure(figsize=(10,7))
plt.title("Customer Dendograms")
dend= shc.dendrogram(shc.linkage(data,method='ward'))
# plt.show()


from sklearn.cluster import AgglomerativeClustering 
cluster= AgglomerativeClustering(n_clusters=5,metric='euclidean',linkage='ward')
labels_=cluster.fit_predict(data)
print(labels_)

plt.figure(figsize=(10,7))
plt.scatter(data[:,0],data[:,-1] ,c=cluster.labels_, cmap='rainbow')
plt.show()