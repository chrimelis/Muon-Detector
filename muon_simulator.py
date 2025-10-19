import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.integrate as integrate
import muon_detector_helpers as md

# Define a particle -> (x, y, phi, theta)
# - x and y are the positions of the particles on the 1st sheet
# - phi and theta determine a direction
# n = cos(phi)sin(theta) i + sin(phi)sin(theta) j + cos(theta) k
# We restrict theta to be in the interval (pi/2, pi]
# since in order for a particle to be detected it has to come from outside the detector device

# The sheets are assumed to be square
a = 1   # the side of the square surface
# Vertical distance between the 2 sheets
d = 10
# We use "muon_generator.py" output to generate uniformly N particles on the 1st sheet of the detector (top sheet)
# and also a random downward velocity direction described by the angles phi and theta.
N = 10**5
# Load the pre-generated random samples/data 
data = np.load('data.npz')
x_top_data = data['x_top_data'] # x is uniformly distributed
y_top_data = data['y_top_data'] # y is uniformly distributed
phi_data = data['phi_data'] # phi is uniformly distributed
theta_data = data['theta_data'] # theta ~ (4/pi)*cos^2(theta)

# Theta pdf
cos_sq = md.cosine_sq(a=0,b=np.pi/2)

# theta_max: the most extreme polar angle that might occur
theta_max = np.arctan2(np.sqrt(2)*a, d)

# Bottom sheet 
x_bot_data, y_bot_data, phi_data_bot, theta_data_bot = md.propagate_particles(d,a,x_top_data,y_top_data,phi_data,theta_data)
# theta pdf
normalization_const, error = integrate.quad(lambda x:cos_sq.pdf(x)*md.helper_theta(x,a,d), 0, np.pi/2)
print(f'(Theta) Integration Error: {error}')
theta_bot_pdf = lambda x:cos_sq.pdf(x)*md.helper_theta(x,a,d)/normalization_const
# phi pdf
norm_const_phi_b,error = integrate.quad(lambda x:md.helper_phi(x,a,d), 0,2*np.pi)
print(f'(Phi) Integration Error: {error}')
phi_bot_pdf = lambda x:md.helper_phi(x,a,d)/norm_const_phi_b
# x and y pdf
norm_const_x_b,error = integrate.quad(lambda x:md.helper_x_2(x,a,d), 0, a)
print(f'(X and Y) Integration Error: {error}')
x_bot_pdf = lambda x:md.helper_x_2(x,a,d)/norm_const_x_b
y_bot_pdf = x_bot_pdf


# Hit rate
print(f"Hit both rate = {100*len(x_bot_data)/len(x_top_data)}%")

# Histograms

# Phi angle
plt.figure()
plt.suptitle(r'$N_\mathrm{top}$='+f'{N} | Rescaled Histograms | '+r'$N_\mathrm{bot}$'+f'={len(x_bot_data)}',fontsize=18)
# Top sheet
plt.subplot(1,2,1)
plt.title(fr'Top sheet: $\phi \in [0, 2\pi]$',fontsize=16)
counts_top,bins_top,_ = plt.hist(phi_data, bins='auto', density=True)
plt.legend([f'#bins={len(bins_top)-1}'])
x = np.linspace(0,2*np.pi,2)
y = [1/(2*np.pi),1/(2*np.pi)]
plt.plot(x,y)
print(f'(Top sheet) number of bins = {len(bins_top)-1}')
# Bottom sheet
plt.subplot(1,2,2)
plt.title(r'Bottom sheet: $\phi \in [0, 2\pi]$',fontsize=16)
counts_bot,bins_bot,_ = plt.hist(phi_data_bot, bins='auto', density=True)
plt.legend([f'#bins={len(bins_top)-1}'])
x=np.linspace(0,2*np.pi,1000)
y=[phi_bot_pdf(x[i]) for i in range(len(x))]
plt.plot(x,y)
print(f'(Bottom sheet) number of bins = {len(bins_bot)-1}')


# Theta angle
plt.figure()
plt.suptitle(r'$N_\mathrm{top}$='+f'{N} | Rescaled Histograms | '+r'$N_\mathrm{bot}$'+f'={len(x_bot_data)}',fontsize=18)
# Top sheet
plt.subplot(1,2,1)
plt.title(fr'Top sheet: $\theta \in [0, \pi/2)$',fontsize=16)
counts_top,bins_top,_ = plt.hist(theta_data, bins='auto', density=True)
plt.legend([f'#bins={len(bins_top)-1}'])
x = np.linspace(0,np.pi/2,1000)
y = [cos_sq.pdf(x[i]) for i in range(len(x))]
plt.plot(x,y)
print(f'(Top sheet) number of bins = {len(bins_top)-1}')
# Bottom sheet
plt.subplot(1,2,2)
plt.title(r'Bottom sheet: $\theta \in [0, \theta_{max}] \rightarrow \tan \theta_{max} = \sqrt{2}a/d$',fontsize=16)
counts_bot,bins_bot,_ = plt.hist(theta_data_bot, bins='auto', density=True)
plt.legend([f'#bins={len(bins_top)-1}'])
x=np.linspace(0,theta_max,1000)
y=[theta_bot_pdf(x[i]) for i in range(len(x))]
plt.plot(x,y)
print(f'(Bottom sheet) number of bins = {len(bins_bot)-1}')


# x-coordinate
plt.figure()
plt.suptitle(r'$N_\mathrm{top}$='+f'{N} | Rescaled Histograms | '+r'$N_\mathrm{bot}$'+f'={len(x_bot_data)}',fontsize=18)
# Top sheet
plt.subplot(1,2,1)
plt.title(fr'Top sheet: $x \in [0, a={a}]$',fontsize=16)
counts_top,bins_top,_ = plt.hist(x_top_data, bins='auto', density=True)
plt.legend([f'#bins={len(bins_top)-1}'])
x = np.linspace(0,a,2)
y = [1/a,1/a]
plt.plot(x,y)
print(f'(Top sheet) number of bins = {len(bins_top)-1}')
# Bottom sheet
plt.subplot(1,2,2)
plt.title(fr'Bottom sheet: $x \in [0, a={a}]$',fontsize=16)
counts_bot,bins_bot,_ = plt.hist(x_bot_data, bins='auto', density=True)
plt.legend([f'#bins={len(bins_top)-1}'])
x=np.linspace(0,a,1000)
y=[x_bot_pdf(x[i]) for i in range(len(x))]
plt.plot(x,y)
print(f'(Bottom sheet) number of bins = {len(bins_bot)-1}')


# y-coordinate
plt.figure()
plt.suptitle(r'$N_\mathrm{top}$='+f'{N} | Rescaled Histograms | '+r'$N_\mathrm{bot}$'+f'={len(x_bot_data)}',fontsize=18)
# Top sheet
plt.subplot(1,2,1)
plt.title(fr'Top sheet: $y \in [0, a={a}]$',fontsize=16)
counts_top,bins_top,_ = plt.hist(y_top_data, bins='auto', density=True)
plt.legend([f'#bins={len(bins_top)-1}'])
x = np.linspace(0,a,2)
y = [1/a,1/a]
plt.plot(x,y)
print(f'(Top sheet) number of bins = {len(bins_top)-1}')
# Bottom sheet
plt.subplot(1,2,2)
plt.title(fr'Bottom sheet: $y \in [0, a={a}]$',fontsize=16)
counts_bot,bins_bot,_ = plt.hist(y_bot_data, bins='auto', density=True)
plt.legend([f'#bins={len(bins_top)-1}'])
x=np.linspace(0,a,1000)
y=[y_bot_pdf(x[i]) for i in range(len(x))]
plt.plot(x,y)
print(f'(Bottom sheet) number of bins = {len(bins_bot)-1}')


# 2D plots of particles
plt.figure()
plt.suptitle(f'Hit both rate = {100*len(x_bot_data)/len(x_top_data)}%',fontsize=18)
# Top sheet
plt.subplot(1,2,1)
plt.scatter(x_top_data, y_top_data, s=1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Top sheet #particles = {len(x_top_data)}',fontsize=16)
# Bottom sheet
plt.subplot(1,2,2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f"Bottom sheet #particles = {len(x_bot_data)}",fontsize=16)
plt.scatter(x_bot_data,y_bot_data, s=1)


# 3D visualization of the particle trajectories

# number of particles whose trajectories will be visualized
n = 100
# Top position and velocity directions
x_top_data_3D = x_top_data[0:n] 
y_top_data_3D = y_top_data[0:n] 
phi_data_3D = phi_data[0:n]
theta_data_3D = theta_data[0:n]
# Bottom position
x_bot_data_3D = [0]*n
y_bot_data_3D = [0]*n
hits_both = [0]*n
for i in range(n):
    x_bot_data_3D[i], y_bot_data_3D[i], hits_both[i] = md.propagate_particle(d,a,x_top_data_3D[i],y_top_data_3D[i],phi_data_3D[i],theta_data_3D[i]) 
# pairs of x,y,z values -> used for drawing line segments between them
x_vals = list(zip(x_top_data_3D,x_bot_data_3D))
y_vals = list(zip(y_top_data_3D,y_bot_data_3D))
z_vals = [(d,0) for _ in range(n)]

# initialize new figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig.suptitle(f'Total particles = {N} | Hit both rate = {100*len(x_bot_data)/len(x_top_data)}%\nparticles shown here = {n} | depicted hit rate = {100*sum(hits_both)/n}%\nVertical distance d = {d}m | Sheet side length a = {a}m')
ax.set_xlim([-a/2, 3/2*a])
ax.set_ylim([-a/2, 3/2*a])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# loop over the pairs of points and draw line segments between each pair
for x,y,z in zip(x_vals, y_vals, z_vals):
    ax.plot(x,y,z,marker='o',color="gray", markersize=2, markerfacecolor='black',lw=0.5)

# Bottom rectangle vertices
bot_rectangle_verts = [
    [0, 0, 0],  # point A
    [a, 0, 0],  # point B
    [a, a, 0],  # point C
    [0, a, 0]   # point D
]
# Top rectangle vertices
top_rectangle_verts = [
    [0, 0, d],  # point A
    [a, 0, d],  # point B
    [a, a, d],  # point C
    [0, a, d]   # point D
]
# Wrap in a list to make each rectangle a single face
bot_face = [bot_rectangle_verts]
top_face = [top_rectangle_verts]
# Create the filled polygon
bot_rectangle = Poly3DCollection(bot_face, facecolors='skyblue', edgecolors='black', linewidths=1, alpha=0.4)
top_rectangle = Poly3DCollection(top_face, facecolors='skyblue', edgecolors='black', linewidths=1, alpha=0.4)
# Add it to the 3D figure
ax.add_collection3d(bot_rectangle)
ax.add_collection3d(top_rectangle)


plt.show()