############################################################
# This script is a fully autonomous version of the 
# Muon Detector. It creates and runs all Monte Carlo
# Simulations and compares experimental to theoretical
# results.
############################################################

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import rv_continuous
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.integrate as integrate

# Define a particle -> (x, y, phi, theta)
# - x and y are the positions of the particles on the 1st sheet
# - phi and theta determine a direction
# n = cos(phi)sin(theta) i + sin(phi)sin(theta) j + cos(theta) k
# We restrict theta to be in the interval (pi/2, pi]
# since in order for a particle to be detected it has to come from outside the detector device

# The sheets are assumed to be square
a = 1   # the side of the square surface
# Vertical distance between the 2 sheets
d = 1
# Generate uniformly N particles on the 1st sheet of the detector (top sheet)
N = 10**5
x_top_data = [random.uniform(0,a) for _ in range(N)] # x is uniformly distributed
y_top_data = [random.uniform(0,a) for _ in range(N)] # y is uniformly distributed
# Generate the velocity direction in 3D space
phi_data = [random.uniform(0,2*np.pi) for _ in range(N)]    # phi (the azimuthal angle) is uniformly distributed
# If θ' is the polar angle of spherical coordinates
# θ' = pi - theta, with theta taking values in [0,pi/2) according to the distribution theta = |v| where v ~ 2/pi * cos^2(v) with v in [-pi/2,pi/2]
# theta ~ cos^2 distribution

# Define the cosine squared distribution
class cosine_sq(rv_continuous):
    def cos_sq(self,theta_max):
        if theta_max <= 0:
            print("Error: The condition [theta_max > 0] must be true")
            exit()
        # pdf : f(theta) = A cos^2 (omega * theta),  theta in [0, theta_max] 
        # omega*theta_max = pi/2 =>
        omega = (np.pi/2)*(1/theta_max)
        A = 2/(theta_max) # normalization constant so that [int_{0}^{theta_max} f(theta) d(theta) = 1]
        f = lambda theta : A*(np.cos(omega*theta))**2   # PDF
        return f
    
    def _pdf(self, x):
        f = self.cos_sq(self.b)
        return f(x)
    
def helper_theta(x,a,d):
    """"This function computes the product of the probability that (xbot,ybot) fall into the bottom sheet of the detector 
    given that theta=x takes a const value, with (multiplied by) the pdf value of theta @theta=x
    
    It corresponds to the numerator of the bayes rule. And it will help us compute the pdf of theta_bot that is the pdf
    of the theta angles of the particles that trigger the bottom plate 
    """
    if (x < 0) or (x >= np.pi/2):
        return 0
    if x==0:
        return 1
    mu = np.tan(x)/(a/d)
    if mu<1:
        return 1+1/np.pi*mu*(mu-4)
    else:
        return 1/(np.pi/2)*float(np.heaviside([np.sqrt(2)-mu],0))*(np.arcsin(1/mu)-np.arccos(1/mu)+2*(np.sqrt(mu**2-1)-1)+1-mu**2/2)

def helper_phi(phi,a,d):
    """It computes the probability P(Xb in [0,a], Yb in [0,a] | phi = phi)
    This probability is a function of the angle phi
    
    Why is it helpful to define this function?
    The product of this function times the unrestricted pdf of the angle phi
    so that its integral is equal to 1 
    yields according to Bayes rule the pdf of the angle phi of the particles that trigger both sheets.
    """
    if np.cos(phi) == 0:
        theta1 = np.pi/2
    else:
        theta1 = np.arctan(a/d/abs(np.cos(phi)))
    if np.sin(phi) == 0:
        theta2 = np.pi/2
    else:
        theta2 = np.arctan(a/d/abs(np.sin(phi)))
    theta_max = min(theta1,theta2)
    g1 = lambda x:0.5*x+0.25*np.sin(2*x)
    g2 = lambda x:-0.25*np.cos(2*x)
    g3 = lambda x:0.5*x-0.25*np.sin(2*x)
    dg1 = g1(theta_max)-g1(0)
    dg2 = g2(theta_max)-g2(0)
    dg3 = g3(theta_max)-g3(0)
    return 4/np.pi * (dg1 -dg2*d/a*(abs(np.cos(phi))+abs(np.sin(phi))) + dg3*d**2/a**2 *abs(np.cos(phi))*abs(np.sin(phi)))

def helper_x_1(x,theta,a,d):
    """This function computes J0 (see theoretical analysis)
    as the first step towards computing 
    the pdf of the x-coordinate of the particles that trigger both sheets.
    According to the theoretical analysis (or due to symmetry)
    the pdf of this x-coordinate is identical to the pdf of the corresponding y-coordinate
    """
    if x<0 or x>a:
        return 0
    if theta<0 or theta>=np.pi/2:
        return 0
    if theta == 0:
        phi_M = np.pi
        phi_m = 0
    else:
        phi_M = np.arccos(max(-1,(x-a)/(d*np.tan(theta))))
        phi_m = np.arccos(min(1, x/(d*np.tan(theta))))
    mu = np.tan(theta)
    # 1st term
    t1 = phi_M-phi_m
    # 2nd term
    t2 = mu*(np.cos(phi_M)-np.cos(phi_m))
    # 3rd term
    t3 = 0
    if mu>1:
        phi_1 = np.arcsin(1/mu)
        phi_2 = np.pi - phi_1
        t3 = max(phi_1,phi_m)-min(phi_2,phi_M)+mu*(np.cos(max(phi_1,phi_m)) - np.cos(min(phi_2,phi_M)))
    return t1+t2+t3

def helper_x_2(x,a,d):
    """This function carries out the integration of J0 times the unrestricted pdf of angle theta
    as a partial step towards computing the pdf of the x-coordinate for the particles
    that trigger both sheets
    
    Why does it help us?
    - A normalized version of this function of x yields the 
    pdf of the x-coordinate for the particles
    that trigger both sheets.
    - According to the theoretical analysis (or due to symmetry)
    the pdf of this x-coordinate is identical to the pdf of the corresponding y-coordinate"""
    result, _ = integrate.quad(lambda theta:np.cos(theta)**2*helper_x_1(x,theta,a,d), 0,np.pi/2)
    return result

def propagate_particles(d, a, x_top, y_top, phi, theta):
    """This function computes the x and y positions on the bottom plate of the particles that trigger it.
    Inputs:
        d : (vertical) distance between the plates
        a : plate side length
        x_top : list containing the x positions of the particles that cross the top plate
        y_top : list containing the y positions of the particles that cross the top plate
    Outputs:
        x_bot : list with the x positions of the initial particles that cross the bottom plate
        y_bot : list with the y positions of the initial particles that cross the bottom plate
        phi_bot : list with the phi angles of the particles that trigger both sheets/plates
        theta_bot : list with the theta angles of the particles that trigger both sheets/plates
    Assumptions:
        - The top plate is located at z=d
        - The bottom plate is located at z=0
    """

    # If d=0, the top and bottom plates are identical
    if d==0:
        return x_top, y_top, list(range(len(x_top)))
    
    # From the given geometry it follows that the bottom plate will not be trigerred if theta > theta_max
    # where theta_max = arctan(sqrt(2)*a/d)
    theta_max = np.arctan2(np.sqrt(2)*a, d) 
    
    j = 0   # counts hits on the 2nd sheet (bottom)
    x_bot = [0]*len(x_top)
    y_bot = [0]*len(x_top)
    phi_bot = [0]*len(x_top)
    theta_bot = [0]*len(x_top)
    # Loop over each particle
    for i in range(len(x_top)):
        # If theta > theta_max => the particle will not trigger the bottom plate
        if theta[i] > theta_max:
            continue
        # The trajectory of each particle is line
        # with parametrization (x(t),y(t),z(t)) = (xtop,ytop,d) + t*(cosφ*sinθ,sinφ*sinθ,-cosθ)
        # We need to compute the t=t_bot that corresponds to z(t_bot) = 0
        # <=> d - t_bot * cosθ = 0 <=> t_bot = d/cosθ
        t_bot_i = d/np.cos(theta[i])
        x_bot_i = x_top[i] + t_bot_i*np.cos(phi[i])*np.sin(theta[i])
        y_bot_i = y_top[i] + t_bot_i*np.sin(phi[i])*np.sin(theta[i])
        if (x_bot_i < 0) or (x_bot_i > a) or (y_bot_i < 0) or (y_bot_i > a):
            continue
        x_bot[j] = x_bot_i
        y_bot[j] = y_bot_i
        phi_bot[j] = phi[i]
        theta_bot[j] = theta[i]
        j = j+1
    return x_bot[0:j], y_bot[0:j], phi_bot[0:j], theta_bot[0:j]                    

def propagate_particle(d, a, x_top, y_top, phi, theta):
    """For a specific particle with known: 
    - initial position on the top sheet (x_top, y_top, d) and 
    - velocity direction given by (cosφ*sinθ,sinφ*sinθ,-cosθ)
    this function computes the future position of this particle 
    when it reaches the plane of the bottom sheet, 
    regardless of whether it falls inside the region of the bottom sheet.
    
    Outputs:
        - x_bot: the x position on the z=0 plane
        - y_bot: the y position on the z=0 plane
        - hits_bot: True/False if it hits/or not the bottom sheet 
    """
    if theta == np.pi/2:
        print("Error theta = π/2")
        exit()
    # If d=0, the top and bottom plates are identical
    if d==0:
        return x_top, y_top, True
    # The trajectory of each particle is line
    # with parametrization (x(t),y(t),z(t)) = (xtop,ytop,d) + t*(cosφ*sinθ,sinφ*sinθ,-cosθ)
    # We need to compute the t=t_bot that corresponds to z(t_bot) = 0
    # <=> d - t_bot * cosθ = 0 <=> t_bot = d/cosθ        
    t_bot = d/np.cos(theta)
    x_bot = x_top + t_bot*np.cos(phi)*np.sin(theta) 
    y_bot = y_top + t_bot*np.sin(phi)*np.sin(theta)
    hits_both = True
    if (x_bot<0) or (x_bot>a) or (y_bot<0) or (y_bot>a):
        hits_both = False
    return x_bot, y_bot, hits_both

# Generate theta angles for all N particles
cos_sq = cosine_sq(a=0,b=np.pi/2) 
theta_data = cos_sq.rvs(size=N,random_state=None)
# theta_max: the most extreme polar angle that might occur
theta_max = np.arctan2(np.sqrt(2)*a, d)
# Bottom sheet 
x_bot_data, y_bot_data, phi_data_bot, theta_data_bot = propagate_particles(d,a,x_top_data,y_top_data,phi_data,theta_data)
# theta pdf
normalization_const, error = integrate.quad(lambda x:cos_sq.pdf(x)*helper_theta(x,a,d), 0, np.pi/2)
print(f'(Theta) Integration Error: {error}')
theta_bot_pdf = lambda x:cos_sq.pdf(x)*helper_theta(x,a,d)/normalization_const
# phi pdf
norm_const_phi_b,error = integrate.quad(lambda x:helper_phi(x,a,d), 0,2*np.pi)
print(f'(Phi) Integration Error: {error}')
phi_bot_pdf = lambda x:helper_phi(x,a,d)/norm_const_phi_b
# x and y pdf
norm_const_x_b,error = integrate.quad(lambda x:helper_x_2(x,a,d), 0, a)
print(f'(X and Y) Integration Error: {error}')
x_bot_pdf = lambda x:helper_x_2(x,a,d)/norm_const_x_b
y_bot_pdf = x_bot_pdf


# Hit rate
print(f"Hit both rate = {100*len(x_bot_data)/len(x_top_data)}%")

# Histograms

# Phi angle
plt.figure()
plt.suptitle(f'Rescaled Histograms | Samples={N}',fontsize=18)
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
plt.legend([f'#bins={len(bins_bot)-1}'])
x=np.linspace(0,2*np.pi,1000)
y=[phi_bot_pdf(x[i]) for i in range(len(x))]
plt.plot(x,y)
print(f'(Bottom sheet) number of bins = {len(bins_bot)-1}')


# Theta angle
plt.figure()
plt.suptitle(f'Rescaled Histograms | Samples={N}',fontsize=18)
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
plt.legend([f'#bins={len(bins_bot)-1}'])
x=np.linspace(0,theta_max,1000)
y=[theta_bot_pdf(x[i]) for i in range(len(x))]
plt.plot(x,y)
print(f'(Bottom sheet) number of bins = {len(bins_bot)-1}')


# x-coordinate
plt.figure()
plt.suptitle(f'Rescaled Histograms | Samples={N}',fontsize=18)
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
plt.legend([f'#bins={len(bins_bot)-1}'])
x=np.linspace(0,a,1000)
y=[x_bot_pdf(x[i]) for i in range(len(x))]
plt.plot(x,y)
print(f'(Bottom sheet) number of bins = {len(bins_bot)-1}')


# y-coordinate
plt.figure()
plt.suptitle(f'Rescaled Histograms | Samples={N}',fontsize=18)
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
plt.legend([f'#bins={len(bins_bot)-1}'])
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
    x_bot_data_3D[i], y_bot_data_3D[i], hits_both[i] = propagate_particle(d,a,x_top_data_3D[i],y_top_data_3D[i],phi_data_3D[i],theta_data_3D[i]) 
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

plt.figure()
plt.suptitle(f'Rescaled Histograms | Samples={N}',fontsize=18)
# Top sheet
plt.subplot(1,2,1)
plt.title(fr'Top sheet: $\theta \in [0, \pi/2)$',fontsize=16)
counts_top,bins_top,_ = plt.hist(theta_data, bins='auto', density=False)
plt.legend([f'#bins={len(bins_top)-1}'])
x = np.linspace(0,np.pi/2,1000)
y = [len(theta_data)*(bins_top[1]-bins_top[0])*cos_sq.pdf(x[i]) for i in range(len(x))]
plt.plot(x,y)
print(f'(Top sheet) number of bins = {len(bins_top)-1}')
# Bottom sheet
plt.subplot(1,2,2)
plt.title(r'Bottom sheet: $\theta \in [0, \theta_{max}] \rightarrow \tan \theta_{max} = \sqrt{2}a/d$',fontsize=16)
counts_bot,bins_bot,_ = plt.hist(theta_data_bot, bins=bins_top, density=False)
plt.legend([f'#bins={len(bins_bot)-1}'])
x = np.linspace(0,np.pi/2,1000)
y = [len(theta_data)*(bins_top[1]-bins_top[0])*cos_sq.pdf(x[i]) for i in range(len(x))]
plt.plot(x,y)
print(f'(Bottom sheet) number of bins = {len(bins_bot)-1}')


plt.show() 
 
# Generate velocity directions
# theta = pi - w, with w taking values in [0,pi/2) according to the distribution w = |v| where v ~ 2/pi * cos^2(v) with v in [-pi/2,pi/2]
# P(W<=w) = P (|v|<= w) = P(-w <= v <= w) = P(v<=w) - P (v <= -w) => Fw(w) = Fv(w) - Fv(-w) => fw(w) = fv(w) + fv(-w) = 4/pi cos^2(w)
# Fw = int_{0}^{w} 4/pi cos^2(x)dx = 4/pi(w/2 + 1/2 int_{0}^{w} cos(2x)dx) =  2/pi w +1/pi sin(2w)
