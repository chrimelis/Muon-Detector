import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import muon_detector_helpers as md

##########################################################
# This script is used for plotting the PDF distributions 
# that were derived using probability theory arguements 
# based on the modelling assumptions of the detector 
# device and the cosmic particles (muons)
##########################################################

a=1 # square sheet side length

d=[0.001, 0.01,0.1,1,10,100,1000] # vertical distance between the detector square sheets
# Phi pdf
plt.figure()
for i in range(len(d)):
    norm_const_phi_b,error = integrate.quad(lambda x:md.helper_phi(x,a,d[i]), 0,2*np.pi)
    phi_bot_pdf = lambda x:md.helper_phi(x,a,d[i])/norm_const_phi_b
    x=np.linspace(0,2*np.pi,1000)
    y=[phi_bot_pdf(x[i]) for i in range(len(x))]
    plt.plot(x,y)
plt.legend([f'd = {d[i]}' for i in range(len(d))])

d=[0.001, 0.01,0.1,1,10,100,1000]
# x pdf
plt.figure()
for i in range(len(d)):
    norm_const_x_b,error = integrate.quad(lambda x:md.helper_x_2(x,a,d[i]), 0,a)
    x_bot_pdf = lambda x:md.helper_x_2(x,a,d[i])/norm_const_x_b
    x=np.linspace(0,a,1000)
    y=[x_bot_pdf(x[i]) for i in range(len(x))]
    plt.plot(x,y)
plt.legend([f'd = {d[i]}' for i in range(len(d))])


# Theta pdf
cos_sq = md.cosine_sq(a=0,b=np.pi/2) 

d=[0.001, 0.01,0.1,1,10,100]
# plt.figure()
plt.subplots(2,3)
for i in range(len(d)):
    theta_max = np.arctan2(np.sqrt(2)*a, d[i]) 
    normalization_const, error = integrate.quad(lambda x:cos_sq.pdf(x)*md.helper_theta(x,a,d[i]), 0, np.pi/2)
    theta_bot_pdf = lambda x:cos_sq.pdf(x)*md.helper_theta(x,a,d[i])/normalization_const
    x=np.linspace(0,theta_max,1000)
    y=[theta_bot_pdf(x[i]) for i in range(len(x))]
    plt.subplot(2,3,i+1)
    plt.plot(x,y)
    if i==0:
        x=np.linspace(0,np.pi/2,20)
        y=[cos_sq.pdf(x[i]) for i in range(len(x))]
        plt.plot(x,y,'*')
        plt.legend([f'd = {d[i]}',r'$\frac{4}{\pi}\cos^2\theta$'])
    else: 
        plt.legend([f'd = {d[i]}'])
           

plt.show()
