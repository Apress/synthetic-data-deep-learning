import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from itertools import combinations
from math import ceil
# Many libraries are required for the code
data_class = make_classification(n_samples=150, n_features=4)# 150 samples with 4 features were generated with the make_classification function.
d_fr = pnd.DataFrame(data_class[0], columns=["x1","x2","x3","x4"])# make data frame
d_fr['y'] = data_class[1]
print (d_fr.head())

comb_var=list(combinations(d_fr.columns[:-1],3)) #Creating 3-combination sets from 4-features data set.
print (comb_var)
lenght_comb = len(comb_var)

fig = plt.figure(figsize=(11,7))
a=221
for i in range(lenght_comb):
    ax = fig.add_subplot(a+i, projection='3d')
    x1 = comb_var[i][0]
    x2 = comb_var[i][1]
    x3 = comb_var[i][2]
    ax.scatter3D(d_fr["x1"],d_fr["x2"],d_fr["x3"],c=d_fr['y'],edgecolor='b', s=100)
    plt.title('Variables'+str(comb_var[i]))
    plt.grid(True)

plt.show()

import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from itertools import combinations
from math import ceil
# Many libraries are required for the code
data_class = make_classification(n_samples=200, n_features=4,class_sep=5.0)# 200 samples with 4 features were generated with the make_classification function.
d_fr = pnd.DataFrame(data_class[0], columns=["x1","x2","x3","x4"])# make data frame
d_fr['y'] = data_class[1]
print (d_fr.head())

comb_var=list(combinations(d_fr.columns[:-1],3)) #Creating 3-combination sets from 4-features data set.
print (comb_var)
lenght_comb = len(comb_var)

fig = plt.figure(figsize=(11,7))
a=221
for i in range(lenght_comb):
    ax = fig.add_subplot(a+i, projection='3d')
    x1 = comb_var[i][0]
    x2 = comb_var[i][1]
    x3 = comb_var[i][2]
    ax.scatter3D(d_fr["x1"],d_fr["x2"],d_fr["x3"],c=d_fr['y'],edgecolor='b', s=100)
    plt.title('Variables'+str(comb_var[i]))
    plt.grid(True)

plt.show()


data_class = make_classification(n_samples=200, n_features=4,class_sep=0.01)# 200 samples with 4 features were generated with the make_classification function.
d_fr = pnd.DataFrame(data_class[0], columns=["x1","x2","x3","x4"])# make data frame
d_fr['y'] = data_class[1]
print (d_fr.head())

comb_var=list(combinations(d_fr.columns[:-1],3)) #Creating 3-combination sets from 4-features data set.
print (comb_var)
lenght_comb = len(comb_var)

fig = plt.figure(figsize=(11,7))
a=221
for i in range(lenght_comb):
    ax = fig.add_subplot(a+i, projection='3d')
    x1 = comb_var[i][0]
    x2 = comb_var[i][1]
    x3 = comb_var[i][2]
    ax.scatter3D(d_fr["x1"],d_fr["x2"],d_fr["x3"],c=d_fr['y'],edgecolor='b', s=100)
    plt.title('Variables'+str(comb_var[i]))
    plt.grid(True)

plt.show()



from faker import Faker
import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from itertools import combinations
from math import ceil
c_map=plt.cm.get_cmap("YlGnBu") #colur of the figure
fig, ax = plt.subplots(3,2,figsize=(11,6))
a=221
sep_par=[0.5,1,5,10] #different separation parameters
for i in range(4):
    data_class = make_classification(n_samples=100,class_sep=sep_par[i],n_features=3,n_informative=1,n_clusters_per_class=1,n_redundant=0, random_state=99) #Creates a data set for each seperation parameter.
    d_fr = pnd.DataFrame(data_class[0], columns=["x1","x2","x3"])
    d_fr['y'] = data_class[1]
    print (d_fr.head())
    plt.subplot(a)
    plt.scatter(d_fr["x1"],d_fr['x2'],c=d_fr['y'], s=100) # scatter plot for each separation parameter
    plt.title('Class Seperation='+ str(sep_par[i]), size=10)
    plt.grid(True)
    a+=1
plt.show()






from faker import Faker
import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from itertools import combinations
from math import ceil

c_map=plt.cm.get_cmap("YlGnBu")# color of the figure
fig, ax = plt.subplots(3,2,figsize=(11,6))# Figure definition with 3 rows and 2 columns for 6 different noise parameters

a=231
noise_data=[0.01,0.1,0.3,0.5,0.75,1]
for i in range(6):
    data_class = make_classification(n_samples=100,flip_y=noise_data[i],n_features=3,n_informative=1,n_clusters_per_class=1,n_redundant=0, random_state=99) # 100 samples with 3 features were generated with the make_classification function.
    d_fr = pnd.DataFrame(data_class[0], columns=["x1","x2","x3"])
    d_fr['y'] = data_class[1]
    print (d_fr.head())
    plt.subplot(a)
    plt.scatter(d_fr["x1"],d_fr['x2'],c=d_fr['y'], s=100)
    plt.title('Noise='+ str(noise_data[i]), size=10)
    plt.grid(True)
    a+=1
plt.show()






from faker import Faker
import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_multilabel_classification

c_map=plt.cm.get_cmap("YlGnBu")
fig, ax = plt.subplots(2,2,figsize=(10,6))
a=221
number_labels=[2,5, 7, 10]
for i in range(4):
    plt.subplot(a)
    x_class, y_class= make_multilabel_classification(n_samples=500, n_features=2,random_state=99, n_classes=3,n_labels=number_labels[i])
    new_y=np.sum(y_class*[4,2,1], axis=1)
    plt.scatter(x_class[:,0],x_class[:,1],c=new_y, s=50, cmap=c_map)
    plt.title('Number of Labels='+str(number_labels[i]))
    a+=1
plt.show()



import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skl_dt


c_map=plt.cm.get_cmap("YlGnBu")
fig, ax = plt.subplots(2,2,figsize=(10,6))
a=221
for ii in range(3,7):
    plt.subplot(a)
    x_data, y = skl_dt.make_blobs(n_samples=500,centers=ii,random_state=99)
    plt.scatter(x_data[:,0],x_data[:,1],c=y, s=50, cmap=c_map)
    
    plt.title('Number of Centers='+str(ii))
    a+=1
plt.show()
    



import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from itertools import combinations
from math import ceil
c_map=plt.cm.get_cmap("YlGnBu")
fig, ax = plt.subplots(3,2,figsize=(12,12))
data_clust = make_blobs(n_samples=500, n_features=4, centers=3)
d_fr = pnd.DataFrame(data_clust[0],columns=['x'+str(i) for i in range(1,5)])
d_fr['y'] = data_clust[1]
print (d_fr.head())
comb_var=list(combinations(d_fr.columns[:-1],2))
print (comb_var)
lenght_comb = len(comb_var)

a=321
for i in range(lenght_comb):
    print (i)
    plt.subplot(a)
    x1 = comb_var[i][0]
    x2 = comb_var[i][1]
    plt.scatter(d_fr[x1],d_fr[x2],c=d_fr['y'],edgecolor='b', s=150)
    plt.xlabel(comb_var[i][0])
    plt.ylabel(comb_var[i][1])
    plt.grid(True)
    a+=1
plt.show()



import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from itertools import combinations
from math import ceil
c_map=plt.cm.get_cmap("YlGnBu")
fig, ax = plt.subplots(2,2,figsize=(12,12))
cluster_st=[0.3,1,5,10]

a=221
for i in range(4):
    data_clust = make_blobs(n_samples=500, n_features=4, centers=3,cluster_std=cluster_st[i])
    d_fr = pnd.DataFrame(data_clust[0],columns=['x'+str(i) for i in range(1,5)])
    d_fr['y'] = data_clust[1]
    plt.subplot(a)
    plt.scatter(d_fr["x1"],d_fr["x2"],c=d_fr['y'],edgecolor='b', s=150)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title('Cluster sdt='+str(cluster_st[i]))
    plt.grid(True)
    a+=1
plt.show()









import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skl_dt

c_map=plt.cm.get_cmap("YlGnBu")

data_circle=skl_dt.make_circles(n_samples=200)
df_circle = pnd.DataFrame(data_circle[0],columns=["x1", "x2"])
df_circle['y'] =data_circle[1]
plt.figure()
plt.scatter(df_circle['x1'],df_circle['x2'],c=df_circle['y'],s=100,edgecolors='k')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.show()



import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skl_dt

c_map=plt.cm.get_cmap("YlGnBu")
fig, ax = plt.subplots(2,2)
a=221
for noise_ in [0,0.05,0.1,0.5]:
    plt.subplot(a)
    data_circle=skl_dt.make_circles(n_samples=200, noise=noise_)
    df_circle = pnd.DataFrame(data_circle[0],columns=["x1", "x2"])
    df_circle['y'] =data_circle[1]
    plt.scatter(df_circle['x1'],df_circle['x2'],c=df_circle['y'],s=100,edgecolors='k')   
    plt.title('Noise value= '+str(noise_))
    a+=1
plt.show()





import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skl_dt

c_map=plt.cm.get_cmap("YlGnBu")

data_moon=skl_dt.make_moons(n_samples=200)
df_moon = pnd.DataFrame(data_moon[0],columns=["x1", "x2"])
df_moon['y'] =data_moon[1]
plt.figure()
plt.scatter(df_moon['x1'],df_moon['x2'],c=df_moon['y'],s=100,edgecolors='k')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.show()




import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skl_dt
fig, ax = plt.subplots(2,2)
a=221
for noise_ in [0,0.05,0.1,0.5]:
    plt.subplot(a)
    data_moon=skl_dt.make_moons(n_samples=200, noise=noise_)
    df_moon = pnd.DataFrame(data_moon[0],columns=["x1", "x2"])
    df_moon ['y'] =data_moon [1]
    plt.scatter(df_moon['x1'],df_moon ['x2'],c=df_moon ['y'],s=100,edgecolors='k')   
    plt.title('Noise value= '+str(noise_))
    a+=1
plt.show()






