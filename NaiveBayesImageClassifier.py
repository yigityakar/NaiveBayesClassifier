#!/usr/bin/env python
# coding: utf-8

# In[20]:


import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[21]:


# read data into memory
data_set = np.genfromtxt("hw02_images.csv", delimiter = ",")



# In[22]:


input_features= len(data_set[0])


# In[23]:


data_labels=np.genfromtxt("hw02_labels.csv", delimiter = ",")
class1=[]
class2=[]
class3=[]
class4=[]
class5=[]


# In[24]:


for i in range(0,30000):
    if data_labels[i] == 1:
        class1.append(data_set[i])
    elif data_labels[i] == 2:
        class2.append(data_set[i])
    elif data_labels[i] == 3:
        class3.append(data_set[i])
    elif data_labels[i] == 4:
        class4.append(data_set[i])
    else:
        class5.append(data_set[i])


# In[25]:



class1=np.array(class1)
class2=np.array(class2)
class3=np.array(class3)
class4=np.array(class4)
class5=np.array(class5)


sample_mean1=[np.mean(class1[:,i])for i in range(0,input_features)]
sample_mean2=[np.mean(class2[:,i])for i in range(0,input_features)]
sample_mean3=[np.mean(class3[:,i])for i in range(0,input_features)]
sample_mean4=[np.mean(class4[:,i])for i in range(0,input_features)]
sample_mean5=[np.mean(class5[:,i])for i in range(0,input_features)]    


    
sample_means= np.array([sample_mean1 , sample_mean2 , sample_mean3 , sample_mean4 ,sample_mean5])

  


# In[26]:


list_stdev1=[]
list_stdev2=[]
list_stdev3=[]
list_stdev4=[]
list_stdev5=[]
total=0
for n in range(0,input_features):

    for i in range(0,6000):
        a=((sample_mean1[n]- class1[i][n])**2)
        total+=a
    
    b=np.sum(total)
    c= math.sqrt((b/6000))
    list_stdev1.append(c)
    total=0

total=0
for n in range(0,input_features):

    for i in range(0,6000):
        a=((sample_mean2[n]- class2[i][n])**2)
        total+=a
    
    b=np.sum(total)
    c= math.sqrt((b/6000))
    list_stdev2.append(c)
    total=0
    
    
total=0
for n in range(0,input_features):

    for i in range(0,6000):
        a=((sample_mean3[n]- class3[i][n])**2)
        total+=a
    
    b=np.sum(total)
    c= math.sqrt((b/6000))
    list_stdev3.append(c)
    total=0
    
total=0
for n in range(0,input_features):

    for i in range(0,6000):
        a=((sample_mean4[n]- class4[i][n])**2)
        total+=a
    
    b=np.sum(total)
    c= math.sqrt((b/6000))
    list_stdev4.append(c)
    total=0
            
total=0
for n in range(0,input_features):

    for i in range(0,6000):
        a=((sample_mean5[n]- class5[i][n])**2)
        total+=a
    
    b=np.sum(total)
    c= math.sqrt((b/6000))
    list_stdev5.append(c)
    total=0
                


# In[16]:


a= len(class1)+ len(class2)+ len(class3)+ len(class4) + len(class5)

class_priors=[]

class_priors.append(len(class1)/a)
class_priors.append(len(class2)/a)
class_priors.append(len(class3)/a)
class_priors.append(len(class4)/a)
class_priors.append(len(class5)/a)

sample_deviations= np.array([list_stdev1, list_stdev2 ,list_stdev3 ,list_stdev4 ,list_stdev5])
print(sample_means)
print(sample_deviations)
print(class_priors)






# In[27]:


result_list=[0,0,0,0,0]
s1=[]
s2=[]
s3=[]
s4=[]
s5=[]

for n in range(0,6000):

    for i in range(0,input_features):


        score = (- 0.5) * np.log(2 * math.pi * list_stdev1[i]**2) - 0.5 * (class1[n][i] - sample_mean1[i])**2 / list_stdev1[i]**2 + np.log(class_priors[0])                

        s1.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev2[i]**2) - 0.5 * (class1[n][i] - sample_mean2[i])**2 / list_stdev2[i]**2 + np.log(class_priors[1])                 

        s2.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev3[i]**2) - 0.5 * (class1[n][i] - sample_mean3[i])**2 / list_stdev3[i]**2 + np.log(class_priors[2])               

        s3.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev4[i]**2) - 0.5 * (class1[n][i] - sample_mean4[i])**2 / list_stdev4[i]**2 + np.log(class_priors[3])                   

        s4.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev5[i]**2) - 0.5 * (class1[n][i] - sample_mean5[i])**2 / list_stdev5[i]**2 + np.log(class_priors[4])                   

        s5.append(score)



    s1=np.array(s1)
    s2=np.array(s2)
    s3=np.array(s3)
    s4=np.array(s4)
    s5=np.array(s5)
    
    
    sum_list=[np.sum(s1),np.sum(s2),np.sum(s3),np.sum(s4),np.sum(s5)]    
    

    if max(sum_list)==sum_list[0]:
        result_list[0] +=1
    elif max(sum_list)==sum_list[1]:
        result_list[1] +=1
    elif max(sum_list)==sum_list[2]:
        result_list[2] +=1
    elif max(sum_list)==sum_list[3]:
        result_list[3] +=1
    else:
        result_list[4] +=1
    s1=[]
    s2=[]
    s3=[]
    s4=[]
    s5=[]    
    
   
    
    
res1 = np.array(result_list)



result_list=[0,0,0,0,0]
s1=[]
s2=[]
s3=[]
s4=[]
s5=[]

for n in range(0,6000):

    for i in range(0,input_features):


        score = (- 0.5) * np.log(2 * math.pi * list_stdev1[i]**2) - 0.5 * (class2[n][i] - sample_mean1[i])**2 / list_stdev1[i]**2 + np.log(class_priors[0])                

        s1.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev2[i]**2) - 0.5 * (class2[n][i] - sample_mean2[i])**2 / list_stdev2[i]**2 + np.log(class_priors[1])                 

        s2.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev3[i]**2) - 0.5 * (class2[n][i] - sample_mean3[i])**2 / list_stdev3[i]**2 + np.log(class_priors[2])               

        s3.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev4[i]**2) - 0.5 * (class2[n][i] - sample_mean4[i])**2 / list_stdev4[i]**2 + np.log(class_priors[3])                   

        s4.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev5[i]**2) - 0.5 * (class2[n][i] - sample_mean5[i])**2 / list_stdev5[i]**2 + np.log(class_priors[4])                   

        s5.append(score)



    s1=np.array(s1)
    s2=np.array(s2)
    s3=np.array(s3)
    s4=np.array(s4)
    s5=np.array(s5)
    
    
    sum_list=[np.sum(s1),np.sum(s2),np.sum(s3),np.sum(s4),np.sum(s5)]    
    

    if max(sum_list)==sum_list[0]:
        result_list[0] +=1
    elif max(sum_list)==sum_list[1]:
        result_list[1] +=1
    elif max(sum_list)==sum_list[2]:
        result_list[2] +=1
    elif max(sum_list)==sum_list[3]:
        result_list[3] +=1
    else:
        result_list[4] +=1
    s1=[]
    s2=[]
    s3=[]
    s4=[]
    s5=[]    
    


res2=(np.array(result_list))


result_list=[0,0,0,0,0]
s1=[]
s2=[]
s3=[]
s4=[]
s5=[]

for n in range(0,6000):

    for i in range(0,input_features):


        score = (- 0.5) * np.log(2 * math.pi * list_stdev1[i]**2) - 0.5 * (class3[n][i] - sample_mean1[i])**2 / list_stdev1[i]**2 + np.log(class_priors[0])                

        s1.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev2[i]**2) - 0.5 * (class3[n][i] - sample_mean2[i])**2 / list_stdev2[i]**2 + np.log(class_priors[1])                 

        s2.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev3[i]**2) - 0.5 * (class3[n][i] - sample_mean3[i])**2 / list_stdev3[i]**2 + np.log(class_priors[2])               

        s3.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev4[i]**2) - 0.5 * (class3[n][i] - sample_mean4[i])**2 / list_stdev4[i]**2 + np.log(class_priors[3])                   

        s4.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev5[i]**2) - 0.5 * (class3[n][i] - sample_mean5[i])**2 / list_stdev5[i]**2 + np.log(class_priors[4])                   

        s5.append(score)



    s1=np.array(s1)
    s2=np.array(s2)
    s3=np.array(s3)
    s4=np.array(s4)
    s5=np.array(s5)
    
    
    sum_list=[np.sum(s1),np.sum(s2),np.sum(s3),np.sum(s4),np.sum(s5)]    
    

    if max(sum_list)==sum_list[0]:
        result_list[0] +=1
    elif max(sum_list)==sum_list[1]:
        result_list[1] +=1
    elif max(sum_list)==sum_list[2]:
        result_list[2] +=1
    elif max(sum_list)==sum_list[3]:
        result_list[3] +=1
    else:
        result_list[4] +=1
    s1=[]
    s2=[]
    s3=[]
    s4=[]
    s5=[]    
    


res3=np.array(result_list)






result_list=[0,0,0,0,0]
s1=[]
s2=[]
s3=[]
s4=[]
s5=[]

for n in range(0,6000):

    for i in range(0,input_features):


        score = (- 0.5) * np.log(2 * math.pi * list_stdev1[i]**2) - 0.5 * (class4[n][i] - sample_mean1[i])**2 / list_stdev1[i]**2 + np.log(class_priors[0])                

        s1.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev2[i]**2) - 0.5 * (class4[n][i] - sample_mean2[i])**2 / list_stdev2[i]**2 + np.log(class_priors[1])                 

        s2.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev3[i]**2) - 0.5 * (class4[n][i] - sample_mean3[i])**2 / list_stdev3[i]**2 + np.log(class_priors[2])               

        s3.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev4[i]**2) - 0.5 * (class4[n][i] - sample_mean4[i])**2 / list_stdev4[i]**2 + np.log(class_priors[3])                   

        s4.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev5[i]**2) - 0.5 * (class4[n][i] - sample_mean5[i])**2 / list_stdev5[i]**2 + np.log(class_priors[4])                   

        s5.append(score)



    s1=np.array(s1)
    s2=np.array(s2)
    s3=np.array(s3)
    s4=np.array(s4)
    s5=np.array(s5)
    
    
    sum_list=[np.sum(s1),np.sum(s2),np.sum(s3),np.sum(s4),np.sum(s5)]    
    

    if max(sum_list)==sum_list[0]:
        result_list[0] +=1
    elif max(sum_list)==sum_list[1]:
        result_list[1] +=1
    elif max(sum_list)==sum_list[2]:
        result_list[2] +=1
    elif max(sum_list)==sum_list[3]:
        result_list[3] +=1
    else:
        result_list[4] +=1
    s1=[]
    s2=[]
    s3=[]
    s4=[]
    s5=[]    
    


res4=(np.array(result_list))






result_list=[0,0,0,0,0]
s1=[]
s2=[]
s3=[]
s4=[]
s5=[]

for n in range(0,6000):

    for i in range(0,input_features):


        score = (- 0.5) * np.log(2 * math.pi * list_stdev1[i]**2) - 0.5 * (class5[n][i] - sample_mean1[i])**2 / list_stdev1[i]**2 + np.log(class_priors[0])                

        s1.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev2[i]**2) - 0.5 * (class5[n][i] - sample_mean2[i])**2 / list_stdev2[i]**2 + np.log(class_priors[1])                 

        s2.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev3[i]**2) - 0.5 * (class5[n][i] - sample_mean3[i])**2 / list_stdev3[i]**2 + np.log(class_priors[2])               

        s3.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev4[i]**2) - 0.5 * (class5[n][i] - sample_mean4[i])**2 / list_stdev4[i]**2 + np.log(class_priors[3])                   

        s4.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev5[i]**2) - 0.5 * (class5[n][i] - sample_mean5[i])**2 / list_stdev5[i]**2 + np.log(class_priors[4])                   

        s5.append(score)



    s1=np.array(s1)
    s2=np.array(s2)
    s3=np.array(s3)
    s4=np.array(s4)
    s5=np.array(s5)
    
    
    sum_list=[np.sum(s1),np.sum(s2),np.sum(s3),np.sum(s4),np.sum(s5)]    
    

    if max(sum_list)==sum_list[0]:
        result_list[0] +=1
    elif max(sum_list)==sum_list[1]:
        result_list[1] +=1
    elif max(sum_list)==sum_list[2]:
        result_list[2] +=1
    elif max(sum_list)==sum_list[3]:
        result_list[3] +=1
    else:
        result_list[4] +=1
    s1=[]
    s2=[]
    s3=[]
    s4=[]
    s5=[]    
    

res5=(np.array(result_list))




res=np.array([res1,res2,res3,res4,res5])
res=res.transpose()
print(res)


















# In[28]:


test1=[]
test2=[]
test3=[]
test4=[]
test5=[]



for i in range(30000,35000):
    if data_labels[i] == 1:
        test1.append(data_set[i])
    elif data_labels[i] == 2:
        test2.append(data_set[i])
    elif data_labels[i] == 3:
        test3.append(data_set[i])
    elif data_labels[i] == 4:
        test4.append(data_set[i])
    else:
        test5.append(data_set[i])

        


# In[29]:


result_list=[0,0,0,0,0]
s1=[]
s2=[]
s3=[]
s4=[]
s5=[]

for n in range(0,1000):

    for i in range(0,input_features):


        score = (- 0.5) * np.log(2 * math.pi * list_stdev1[i]**2) - 0.5 * (test1[n][i] - sample_mean1[i])**2 / list_stdev1[i]**2 + np.log(class_priors[0])                

        s1.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev2[i]**2) - 0.5 * (test1[n][i] - sample_mean2[i])**2 / list_stdev2[i]**2 + np.log(class_priors[1])                 

        s2.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev3[i]**2) - 0.5 * (test1[n][i] - sample_mean3[i])**2 / list_stdev3[i]**2 + np.log(class_priors[2])               

        s3.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev4[i]**2) - 0.5 * (test1[n][i] - sample_mean4[i])**2 / list_stdev4[i]**2 + np.log(class_priors[3])                   

        s4.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev5[i]**2) - 0.5 * (test1[n][i] - sample_mean5[i])**2 / list_stdev5[i]**2 + np.log(class_priors[4])                   

        s5.append(score)



    s1=np.array(s1)
    s2=np.array(s2)
    s3=np.array(s3)
    s4=np.array(s4)
    s5=np.array(s5)
    
    
    sum_list=[np.sum(s1),np.sum(s2),np.sum(s3),np.sum(s4),np.sum(s5)]    
    

    if max(sum_list)==sum_list[0]:
        result_list[0] +=1
    elif max(sum_list)==sum_list[1]:
        result_list[1] +=1
    elif max(sum_list)==sum_list[2]:
        result_list[2] +=1
    elif max(sum_list)==sum_list[3]:
        result_list[3] +=1
    else:
        result_list[4] +=1
    s1=[]
    s2=[]
    s3=[]
    s4=[]
    s5=[]    
    
   
    
    
res1 = np.array(result_list)



result_list=[0,0,0,0,0]
s1=[]
s2=[]
s3=[]
s4=[]
s5=[]

for n in range(0,1000):

    for i in range(0,input_features):


        score = (- 0.5) * np.log(2 * math.pi * list_stdev1[i]**2) - 0.5 * (test2[n][i] - sample_mean1[i])**2 / list_stdev1[i]**2 + np.log(class_priors[0])                

        s1.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev2[i]**2) - 0.5 * (test2[n][i] - sample_mean2[i])**2 / list_stdev2[i]**2 + np.log(class_priors[1])                 

        s2.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev3[i]**2) - 0.5 * (test2[n][i] - sample_mean3[i])**2 / list_stdev3[i]**2 + np.log(class_priors[2])               

        s3.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev4[i]**2) - 0.5 * (test2[n][i] - sample_mean4[i])**2 / list_stdev4[i]**2 + np.log(class_priors[3])                   

        s4.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev5[i]**2) - 0.5 * (test2[n][i] - sample_mean5[i])**2 / list_stdev5[i]**2 + np.log(class_priors[4])                   

        s5.append(score)



    s1=np.array(s1)
    s2=np.array(s2)
    s3=np.array(s3)
    s4=np.array(s4)
    s5=np.array(s5)
    
    
    sum_list=[np.sum(s1),np.sum(s2),np.sum(s3),np.sum(s4),np.sum(s5)]    
    

    if max(sum_list)==sum_list[0]:
        result_list[0] +=1
    elif max(sum_list)==sum_list[1]:
        result_list[1] +=1
    elif max(sum_list)==sum_list[2]:
        result_list[2] +=1
    elif max(sum_list)==sum_list[3]:
        result_list[3] +=1
    else:
        result_list[4] +=1
    s1=[]
    s2=[]
    s3=[]
    s4=[]
    s5=[]    
    


res2=(np.array(result_list))


result_list=[0,0,0,0,0]
s1=[]
s2=[]
s3=[]
s4=[]
s5=[]

for n in range(0,1000):

    for i in range(0,input_features):


        score = (- 0.5) * np.log(2 * math.pi * list_stdev1[i]**2) - 0.5 * (test3[n][i] - sample_mean1[i])**2 / list_stdev1[i]**2 + np.log(class_priors[0])                

        s1.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev2[i]**2) - 0.5 * (test3[n][i] - sample_mean2[i])**2 / list_stdev2[i]**2 + np.log(class_priors[1])                 

        s2.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev3[i]**2) - 0.5 * (test3[n][i] - sample_mean3[i])**2 / list_stdev3[i]**2 + np.log(class_priors[2])               

        s3.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev4[i]**2) - 0.5 * (test3[n][i] - sample_mean4[i])**2 / list_stdev4[i]**2 + np.log(class_priors[3])                   

        s4.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev5[i]**2) - 0.5 * (test3[n][i] - sample_mean5[i])**2 / list_stdev5[i]**2 + np.log(class_priors[4])                   

        s5.append(score)



    s1=np.array(s1)
    s2=np.array(s2)
    s3=np.array(s3)
    s4=np.array(s4)
    s5=np.array(s5)
    
    
    sum_list=[np.sum(s1),np.sum(s2),np.sum(s3),np.sum(s4),np.sum(s5)]    
    

    if max(sum_list)==sum_list[0]:
        result_list[0] +=1
    elif max(sum_list)==sum_list[1]:
        result_list[1] +=1
    elif max(sum_list)==sum_list[2]:
        result_list[2] +=1
    elif max(sum_list)==sum_list[3]:
        result_list[3] +=1
    else:
        result_list[4] +=1
    s1=[]
    s2=[]
    s3=[]
    s4=[]
    s5=[]    
    


res3=np.array(result_list)






result_list=[0,0,0,0,0]
s1=[]
s2=[]
s3=[]
s4=[]
s5=[]

for n in range(0,1000):

    for i in range(0,input_features):


        score = (- 0.5) * np.log(2 * math.pi * list_stdev1[i]**2) - 0.5 * (test4[n][i] - sample_mean1[i])**2 / list_stdev1[i]**2 + np.log(class_priors[0])                

        s1.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev2[i]**2) - 0.5 * (test4[n][i] - sample_mean2[i])**2 / list_stdev2[i]**2 + np.log(class_priors[1])                 

        s2.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev3[i]**2) - 0.5 * (test4[n][i] - sample_mean3[i])**2 / list_stdev3[i]**2 + np.log(class_priors[2])               

        s3.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev4[i]**2) - 0.5 * (test4[n][i] - sample_mean4[i])**2 / list_stdev4[i]**2 + np.log(class_priors[3])                   

        s4.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev5[i]**2) - 0.5 * (test4[n][i] - sample_mean5[i])**2 / list_stdev5[i]**2 + np.log(class_priors[4])                   

        s5.append(score)



    s1=np.array(s1)
    s2=np.array(s2)
    s3=np.array(s3)
    s4=np.array(s4)
    s5=np.array(s5)
    
    
    sum_list=[np.sum(s1),np.sum(s2),np.sum(s3),np.sum(s4),np.sum(s5)]    
    

    if max(sum_list)==sum_list[0]:
        result_list[0] +=1
    elif max(sum_list)==sum_list[1]:
        result_list[1] +=1
    elif max(sum_list)==sum_list[2]:
        result_list[2] +=1
    elif max(sum_list)==sum_list[3]:
        result_list[3] +=1
    else:
        result_list[4] +=1
    s1=[]
    s2=[]
    s3=[]
    s4=[]
    s5=[]    
    


res4=(np.array(result_list))






result_list=[0,0,0,0,0]
s1=[]
s2=[]
s3=[]
s4=[]
s5=[]

for n in range(0,1000):

    for i in range(0,input_features):


        score = (- 0.5) * np.log(2 * math.pi * list_stdev1[i]**2) - 0.5 * (test5[n][i] - sample_mean1[i])**2 / list_stdev1[i]**2 + np.log(class_priors[0])                

        s1.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev2[i]**2) - 0.5 * (test5[n][i] - sample_mean2[i])**2 / list_stdev2[i]**2 + np.log(class_priors[1])                 

        s2.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev3[i]**2) - 0.5 * (test5[n][i] - sample_mean3[i])**2 / list_stdev3[i]**2 + np.log(class_priors[2])               

        s3.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev4[i]**2) - 0.5 * (test5[n][i] - sample_mean4[i])**2 / list_stdev4[i]**2 + np.log(class_priors[3])                   

        s4.append(score)

        score = (- 0.5) * np.log(2 * math.pi * list_stdev5[i]**2) - 0.5 * (test5[n][i] - sample_mean5[i])**2 / list_stdev5[i]**2 + np.log(class_priors[4])                   

        s5.append(score)



    s1=np.array(s1)
    s2=np.array(s2)
    s3=np.array(s3)
    s4=np.array(s4)
    s5=np.array(s5)
    
    
    sum_list=[np.sum(s1),np.sum(s2),np.sum(s3),np.sum(s4),np.sum(s5)]    
    

    if max(sum_list)==sum_list[0]:
        result_list[0] +=1
    elif max(sum_list)==sum_list[1]:
        result_list[1] +=1
    elif max(sum_list)==sum_list[2]:
        result_list[2] +=1
    elif max(sum_list)==sum_list[3]:
        result_list[3] +=1
    else:
        result_list[4] +=1
    s1=[]
    s2=[]
    s3=[]
    s4=[]
    s5=[]    
    

res5=(np.array(result_list))




res=np.array([res1,res2,res3,res4,res5])
res=res.transpose()
print(res)


















# In[ ]:





# In[ ]:




