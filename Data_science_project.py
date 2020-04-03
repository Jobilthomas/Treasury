#Measurement of single pulse properties
import numpy as np
import math
import pandas as pd
import scipy.optimize as op
import matplotlib.pyplot as plt

df=pd.read_table("Laser_data1000.txt",usecols=['$#ADCS 4'])
a=int(df['$#ADCS 4'][780])+int(df['$#ADCS 4'][781])
j=0
d=[]
for i in range(len(df['$#ADCS 4'])):
    b=df['$#ADCS 4'][i][0]
    if b=="$":
        continue
    d.append(int(df['$#ADCS 4'][i]))
f=int(len(d)/400)
n=np.zeros((f,400))
for i in range(f):
    for j in range (400):
        n[i,j]=d[400*i+j]
print("Number of data point=",len(d))
print("Numper of pulse=",f)


    
# data and ocilloscope specification   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fullscale=65536
volt_scale=0.5
v_offset=0.2
volt_per_bit = volt_scale/fullscale
print("volt per bit=",volt_per_bit)


#convert data point into corresponding Voltage %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i in range(f):
    for j in range(400):
        n[i][j]=-((n[i][j]*volt_per_bit - v_offset))
x=np.arange(0,50,0.125)
y=n[0]
maxindice=np.where(y==np.amax(y))
g=maxindice[0][0]
mean=sum(x*y)/sum(y)
sigma2=sum(y*(x-mean)**2)/sum(y)
m=y[g]

def gaus(x,a,x0,sigma2):                     #we can very well define any other func here
    return a*np.exp(-(x-x0)**2/(2*sigma2))


#Estimation of pulse amplitude by multi Gauss fitting algorithm

x1=x[g-75:g-15]
x2=x[g-15:g+15]
x3=x[g+21:g+100]
y1=y[g-75:g-15]
y2=y[g-15:g+15]
y3=y[g+21:g+100]

x4=x[g+15:g+130]


popt1,pcov1 = op.curve_fit(gaus,x1,y1,p0=[m,mean,sigma2])
popt2,pcov2 = op.curve_fit(gaus,x2,y2,p0=[m,mean,sigma2])
popt3,pcov3 = op.curve_fit(gaus,x3,y3,p0=[m,mean,sigma2])
plt.plot(x,y,'b+:',label='data')
plt.plot(x1,gaus(x1,*popt1),'ro:',label='fit1')
plt.plot(x2,gaus(x2,*popt2),'mo:',label='fit2')
plt.plot(x4,gaus(x4,*popt3),'ko:',label='fit3')
plt.title("Single pulse profile")
plt.xlabel("Time in nsec")
plt.ylabel("Amplitude in Volt")
plt.legend()
plt.show()


#Estimation of pulse rise time by multi Gauss fitting algorithm


x1=x[g-50:g+2]
x2=x[g+30:g+80]
y1=y[g-50:g+2]
y2=y[g+30:g+80]


x3=x[g+5:g+150]


popt1,pcov1 = op.curve_fit(gaus,x1,y1,p0=[m,mean,sigma2])
popt2,pcov2 = op.curve_fit(gaus,x2,y2,p0=[m,mean,sigma2])
plt.plot(x,y,'b+:',label='data')
plt.plot(x1,gaus(x1,*popt1),'ro:',label='fit1')
plt.plot(x3,gaus(x3,*popt2),'mo:',label='fit2')
plt.title("Single pulse profile")
plt.xlabel("Time in nsec")
plt.ylabel("Amplitude in Volt")
plt.legend()
plt.show()


#Pulse amplitude distribution


a=np.zeros(f)
for i in range(f):
    y=n[i]
    mean=sum(x*y)/sum(y)
    sigma2=sum(y*(x-mean)**2)/sum(y)
    maxindice=np.where(y==np.amax(y))
    g=maxindice[0][0]
    y2=y[g-15:g+15]
    x2=x[g-15:g+15]
    m=y[g]
    popt2,pcov2 = op.curve_fit(gaus,x2,y2,p0=[m,mean,sigma2])
    a[i]=popt2[0]


ydata,bin_edges,patches=plt.hist(a,100)
xdata=np.zeros(len(ydata))
for i in range(len(ydata)):
    xdata[i]=(bin_edges[i+1]+bin_edges[i])/2
mean=sum(xdata*ydata)/sum(ydata)
sigma2=sum(ydata*(xdata-mean)**2)/sum(ydata)
maxindice=np.where(ydata==np.amax(ydata))
g=maxindice[0][0]
m=ydata[g]
popt,pcov = op.curve_fit(gaus,xdata,ydata,p0=[m,mean,sigma2])
plt.plot(xdata,gaus(xdata,*popt),'ro:')
plt.title("Amplitude distribution of PMT pulse")
plt.ylabel("Frequency")
plt.xlabel("Amplitude in Volt")
plt.show()
print("Mean pulse amplitude",popt[1],"V")
print("Error in pulse amplitude",math.sqrt(popt[2]),"V")


#Pulse rise time distribution

b=np.zeros(f)
for i in range(f):
    y=n[i]
    mean=sum(x*y)/sum(y)
    sigma2=sum(y*(x-mean)**2)/sum(y)
    maxindice=np.where(y==np.amax(y))
    g=maxindice[0][0]
    y1=y[g-50:g+2]
    x1=x[g-50:g+2]
    m=y[g]
    popt1,pcov1 = op.curve_fit(gaus,x1,y1,p0=[m,mean,sigma2])
    n1=-math.sqrt(-2*popt1[2]*math.log(0.1))+popt1[1]
    n2=-math.sqrt(-2*popt1[2]*math.log(0.9))+popt1[1]
    b[i]=n2-n1

ydata,bin_edges,patches=plt.hist(b,100)
xdata=np.zeros(len(ydata))
for i in range(len(ydata)):
    xdata[i]=(bin_edges[i+1]+bin_edges[i])/2
mean=sum(xdata*ydata)/sum(ydata)
sigma2=sum(ydata*(xdata-mean)**2)/sum(ydata)
maxindice=np.where(ydata==np.amax(ydata))
g=maxindice[0][0]
m=ydata[g]
popt,pcov = op.curve_fit(gaus,xdata,ydata,p0=[m,mean,sigma2])
plt.plot(xdata,gaus(xdata,*popt),'ro:')
plt.title("Rise time of PMT pulse")
plt.ylabel("Frequency")
plt.xlabel("Rise time in nsec")
plt.show()
print("Mean rise time",popt[1],"nS")
print("Error in rise time",math.sqrt(popt[2]),"nS")

























