import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
d_list = ['normal','uniform','triangular','binomial','chisquare','laplace']
p_list = ['0,1','-1,1','-3,0,8','10,0.5','2','5,4']
c_list = ['red','limegreen','gold','purple','blue','magenta']
fig, axs = plt.subplots(nrows=2, ncols=3, dpi=800,figsize=(7,6))

k=0
for i in range(2):
    for j in range(3):
        datas=eval("np.random."+d_list[k]+"("+p_list[k]+",5000)")
        axs[i][j].hist(datas, bins=50,color=c_list[k])
        axs[i][j].set_title(d_list[k]+" dist( "+p_list[k]+")",size=8)
        k+=1

plt.suptitle('Samples from Different Distributions',fontsize=15)
fig.savefig("Dist Histogram.png")
plt.close(fig)        

axs[0][1].hist(d_uniform, bins=50,color=c_list[1])
axs[0][1].set_title("Uniform Distibutions (-1,1)",size=10)
axs[0][2].hist(d_triangular, bins=50,color=c_list[2])
axs[0][2].set_title("Triangular Distibutions (-3,0,8)",size=10)
axs[1][0].hist(d_binomial, bins=50,color=c_list[3])
axs[1][0].set_title("Binomial Distibutions (10,0.5)",size=10)
axs[1][1].hist(d_chisquare, bins=50,color=c_list[4])
axs[1][1].set_title("Chi-Square Distibutions (2)",size=10)
axs[1][2].hist(d_laplance, bins=50,color=c_list[5])
axs[1][2].set_title("Laplance Distibutions (5,4)",size=10)

fig.savefig("output.png")
plt.close(fig)


