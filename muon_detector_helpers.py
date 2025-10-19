import numpy as np
from scipy.stats import rv_continuous
import scipy.integrate as integrate

class cosine_sq(rv_continuous):
    """"cosine_sq class refers to a Continuous Random Variable
    that follows the cosine squared distribution
    
    By inheriting the rv_continuous class
    and overriding the pdf method we create the desired distribution
    """
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