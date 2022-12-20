from faker import Faker
import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt # required for drawing graphics
from sklearn.datasets import make_regression # required for generating a random regression data set

c_mapp=plt.cm.get_cmap("YlGnBu") # color of the graph
data_reg=make_regression(n_samples=1000, n_features=2, 
                         noise=0.0) #1000 samples are created with two features
print (data_reg[0])
data_frame1= pnd.DataFrame(data_reg[0],columns=['x'+str(i) for i in range(1,3)])
data_frame1['y'] = data_reg[1]
print (data_frame1)

fig, axs = plt.subplots(figsize=(9,5)) # define figure size
axs.scatter(data_frame1.x1,data_frame1.x2, cmap=c_mapp,c=data_frame1.y,vmin=min(data_frame1.y), vmax=max(data_frame1.y))
axs.set_title('noise=0') # set title
plt.show()

data_reg=make_regression(n_samples=30, n_features=2,noise=0.0) #30 samples are created with two features
data_frame1= pnd.DataFrame(data_reg[0],columns=['x'+str(i) for i in range(1,3)])
data_frame1['y'] = data_reg[1]
fig, ax = plt.subplots(2,figsize=(9,5)) # two feature will be displayed on two figures

reg_fit_x1=np.polyfit(data_frame1.x1,data_frame1.y,1) # First feature
fit_function1=np.poly1d(reg_fit_x1)# regression fit line 
reg_fit_x2=np.polyfit(data_frame1.x2,data_frame1.y,1)# Second feature
fit_function2=np.poly1d(reg_fit_x2)# regression fit line #Adjusting the figure's properties, such that color, size 
ax[0].scatter(data_frame1.x1,data_frame1.y, s=100,c="red", edgecolor="black")
ax[0].plot(data_frame1.x1,fit_function1(data_frame1.x1),':b', lw=2)
ax[1].scatter(data_frame1.x2,data_frame1.y, s=100,c="red", edgecolor="black")
ax[1].plot(data_frame1.x2,fit_function2(data_frame1.x2), ':b', lw=2)
plt.show() 




from faker import Faker
import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
c_mapp=plt.cm.get_cmap("YlGnBu")
fig, ax = plt.subplots(3,2,figsize=(9,5))#generating figures for 6 different noise values consisting of 3 rows and 2 columns for each noise value
a=231
for noise_data in [1,10,50,100,500,1000,]:
    data_reg2=make_regression(n_samples=1000, n_features=2, noise= noise_data) # A regression data set is created according to different noise values.
    data_frame1= pnd.DataFrame(data_reg2[0],columns=['x'+str(i) for i in range(1,3)])
    data_frame1['y'] = data_reg2[1]
    plt.subplot(a)
    plt.scatter(data_frame1.x1,data_frame1.x2, cmap=c_mapp,c=data_frame1.y,vmin=min(data_frame1.y), vmax=max(data_frame1.y))
    plt.title('Noise='+ str(noise_data), size=10)
    a+=1
plt.show()


from faker import Faker
import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
c_mapp=plt.cm.get_cmap("YlGnBu")
fig, ax = plt.subplots(3,2,figsize=(9,5)) #generating figures for 6 features values consisting of 3 rows and 2 columns
a=231
noise_data=500
data_reg3=make_regression(n_samples=30, n_features=6, 
                         noise=noise_data) #30 samples are created with 6 features
data_frame1= pnd.DataFrame(data_reg3[0],columns=['x'+str(k) for k in range(1,7)])
data_frame1['y'] = data_reg3[1]
for i in range(6):
    reg_fit=np.polyfit(data_frame1[data_frame1.columns[i]],data_frame1.y,1)
    fit_function1=np.poly1d(reg_fit)   # regression fit line 
    plt.subplot(a)
    plt.scatter(data_frame1[data_frame1.columns[i]],data_frame1.y, s=100,c="red", edgecolor="black")#scatter plot
    plt.plot(data_frame1[data_frame1.columns[i]],fit_function1(data_frame1[data_frame1.columns[i]]),':b', lw=2)#line plot
    plt.title('X'+str(i)+" with Noise=500", size=10)
    plt.grid(True)
    a+=1
plt.show()



from faker import Faker
import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
c_mapp=plt.cm.get_cmap("YlGnBu")
fig, ax = plt.subplots(3,2,figsize=(9,5))
data_frame=pnd.DataFrame(data=np.zeros((30,1)))

a=231
noise_data=[1,10,50,100,500,1000,] #noise values
for i in range(6):
    data_reg4=make_regression(n_samples=30, n_features=1, 
                         noise=noise_data[i]) #30 samples are created with 1 features and certain noise values	
    data_frame["x"+str(i+1)]=data_reg4[0]
    data_frame["y"+str(i+1)]=data_reg4[1]
for i in range(6):
    reg_fit=np.polyfit(data_frame["x"+str(i+1)],data_frame["y"+str(i+1)],1)
    fit_function1=np.poly1d(reg_fit) # regression fit line   
    plt.subplot(a)
    plt.scatter(data_frame["x"+str(i+1)],data_frame["y"+str(i+1)], s=100,c="red", edgecolor="black") # scatter plot 
    plt.plot(data_frame["x"+str(i+1)],fit_function1(data_frame["x"+str(i+1)]),':b', lw=2) #regression line 
    plt.title('Noise='+ str(noise_data[i]), size=10)
    plt.grid(True)
    a+=1
plt.show()


import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skl_dataset
c_map=plt.cm.get_cmap("YlGnBu") #color of the figure
x_variables,y = skl_dataset.make_friedman1(n_samples=1500,n_features=6, noise=0.0)#1500 samples are created with 6 features and without any noise

data_frame1=pnd.DataFrame(x_variables,columns=['x'+str(i) for i in range(1,7)])
data_frame1['y'] = y
print (data_frame1 )


import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skl_dataset
c_map=plt.cm.get_cmap("YlGnBu")
x_variables,y = skl_dataset.make_friedman1(n_samples=1500,n_features=6,random_state=0, noise=0.0) #With the make_friedman1 function, 1500 samples are created without any noise 
data_frame=pnd.DataFrame(x_variables,columns=['x'+str(i) for i in range(1,7)])
data_frame['y'] = y
fig = plt.figure(figsize=(7,7))# figure size
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_frame.iloc[:,0], data_frame.iloc[:,1],data_frame.iloc[:,2],c=data_frame.y, cmap=c_map)# A 3D graph was drawn using the first three features of the data set created with the help of the function
plt.title('Function: Friedman1') #title of the graph
plt.show()



from faker import Faker
import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skl_dataset

c_map=plt.cm.get_cmap("YlGnBu")
x_variables,y = skl_dataset.make_friedman2(n_samples=1500,random_state=0, noise=0.0) #With the make_friedman2 function, 1500 samples are created without any noise
data_frame2=pnd.DataFrame(x_variables,columns=['x'+str(i) for i in range(1,5)])
data_frame2['y'] = y
print (data_frame2)
fig  = plt.figure(figsize=(7,7)) #figure size
ax = fig.add_subplot(111, projection='3d') # 3D figure definition
ax.scatter(data_frame2.iloc[:,0], data_frame2.iloc[:,1],data_frame2.iloc[:,2],c=data_frame2.y, cmap=c_map) # A 3D graph was drawn using the first three features of the data set created with the help of the function
plt.title('Function: Friedman2') #title of the graph
plt.show()




from faker import Faker
import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skl_dataset
c_map=plt.cm.get_cmap("YlGnBu")
x_variables,y = skl_dataset.make_friedman3(n_samples=1500,random_state=0, noise=0.0) #With the make_friedman3 function, 1500 samples are created without any noise
data_frame3=pnd.DataFrame(x_variables,columns=['x'+str(i) for i in range(1,5)])
data_frame3['y'] = y
print (data_frame3)
fig  = plt.figure(figsize=(7,7)) #figure size
ax = fig.add_subplot(111, projection='3d') # 3D figure definition
ax.scatter(data_frame3.iloc[:,0], data_frame3.iloc[:,1],data_frame3.iloc[:,2],c=data_frame3.y, cmap=c_map) # A 3D graph was drawn using the first three features of the data set created with the help of the function
plt.title('Function: Friedman3') #title of the graph
plt.show()






def symbolize(s):
    """
    Converts a a string (equation) to a SymPy symbol object
    """
    from sympy import sympify
    s1=s.replace('.','*')
    s2=s1.replace('^','**')
    s3=sympify(s2)
    
    return(s3)
def eval_multinomial(s,vals=None,symbolic_eval=False):
    """
    Evaluates polynomial at vals.
    vals can be simple list, dictionary, or tuple of values.
    vals can also contain symbols instead of real values provided those symbols have been declared before using SymPy
    """
    from sympy import Symbol
    sym_s=symbolize(s)
    sym_set=sym_s.atoms(Symbol)
    sym_lst=[]
    for s in sym_set:
        sym_lst.append(str(s))
    sym_lst.sort()
    if symbolic_eval==False and len(sym_set)!=len(vals):
        print("Length of the input values did not match number of variables and symbolic evaluation is not selected")
        return None
    else:
        if type(vals)==list:
            sub=list(zip(sym_lst,vals))
        elif type(vals)==dict:
            l=list(vals.keys())
            l.sort()
            lst=[]
            for i in l:
                lst.append(vals[i])
            sub=list(zip(sym_lst,lst))
        elif type(vals)==tuple:
            sub=list(zip(sym_lst,list(vals)))
        result=sym_s.subs(sub)
    
    return result
def flip(y,p):
    import numpy as np
    lst=[]
    for i in range(len(y)):
        f=np.random.choice([1,0],p=[p,1-p])
        lst.append(f)
    lst=np.array(lst)
    return np.array(np.logical_xor(y,lst),dtype=int)


def gen_regression_symbolic(m=None,n_samples=100,n_features=2,noise=0.0,noise_dist='normal'):
    """
    Generates regression sample based on a symbolic expression. Calculates the output of the symbolic expression 
    at randomly generated (drawn from a Gaussian distribution) points
    m: The symbolic expression. Needs x1, x2, etc as variables and regular python arithmatic symbols to be used.
    n_samples: Number of samples to be generated
    n_features: Number of variables. This is automatically inferred from the symbolic expression. So this is ignored 
                in case a symbolic expression is supplied. However if no symbolic expression is supplied then a 
                default simple polynomial can be invoked to generate regression samples with n_features.
    noise: Magnitude of Gaussian noise to be introduced (added to the output).
    noise_dist: Type of the probability distribution of the noise signal. 
    Currently supports: Normal, Uniform, t, Beta, Gamma, Poission, Laplace

    Returns a numpy ndarray with dimension (n_samples,n_features+1). Last column is the response vector.
    """
    
    import numpy as np
    from sympy import Symbol,sympify
    
    if m==None:
        m=''
        for i in range(1,n_features+1):
            c='x'+str(i)
            c+=np.random.choice(['+','-'],p=[0.5,0.5])
            m+=c
        m=m[:-1]
    
    sym_m=sympify(m)
    n_features=len(sym_m.atoms(Symbol))
    evals=[]
    lst_features=[]
    
    for i in range(n_features):
        lst_features.append(np.random.normal(scale=5,size=n_samples))
    lst_features=np.array(lst_features)
    lst_features=lst_features.T
    lst_features=lst_features.reshape(n_samples,n_features)
    
    for i in range(n_samples):
        evals.append(eval_multinomial(m,vals=list(lst_features[i])))
    
    evals=np.array(evals)
    evals=evals.reshape(n_samples,1)
    
    if noise_dist=='normal':
        noise_sample=noise*np.random.normal(loc=0,scale=1.0,size=n_samples)
    elif noise_dist=='uniform':
        noise_sample=noise*np.random.uniform(low=0,high=1.0,size=n_samples)
    elif noise_dist=='beta':
        noise_sample=noise*np.random.beta(a=0.5,b=1.0,size=n_samples)
    elif noise_dist=='Gamma':
        noise_sample=noise*np.random.gamma(shape=1.0,scale=1.0,size=n_samples)
    elif noise_dist=='laplace':
        noise_sample=noise*np.random.laplace(loc=0.0,scale=1.0,size=n_samples)
        
    noise_sample=noise_sample.reshape(n_samples,1)
    evals=evals+noise_sample
        
    x=np.hstack((lst_features,evals))
    
    return (x)

from faker import Faker
import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skl_dataset
sym_reg = gen_regression_symbolic(m='(2*x1-(x2^2)/5+15*cos(x3))',n_samples=100,noise=0.001) # generates 100 samples according to any given function with small noise value
data_frame=pnd.DataFrame(sym_reg, columns=['x'+str(i) for i in range(1,4)]+['y'])
print (data_frame)

fig, ax = plt.subplots(1,3,figsize=(10,6))
a=131
for i in range(3):  
    plt.subplot(a)
    plt.scatter(data_frame[data_frame.columns[i]],data_frame.y, s=100,c="red", edgecolor="black")
    plt.title("Symbolic Regression value of x"+str(i+1), size=10)
    plt.grid(True)
    a+=1

plt.show()



from faker import Faker
import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skl_dataset
sym_reg = gen_regression_symbolic(m='(2*x1-(x2^2)/5+15*cos(x3))',n_samples=100,noise=100) # generates 100 samples according to any given function with noise value=100
data_frame=pnd.DataFrame(sym_reg, columns=['x'+str(i) for i in range(1,4)]+['y'])
print (data_frame)

fig, ax = plt.subplots(1,3,figsize=(10,6))
a=131
for i in range(3):  
    plt.subplot(a)
    plt.scatter(data_frame[data_frame.columns[i]],data_frame.y, s=100,c="red", edgecolor="black")
    plt.title("Symbolic Regression value of x"+str(i+1), size=10)
    plt.grid(True)
    a+=1

plt.show()






















