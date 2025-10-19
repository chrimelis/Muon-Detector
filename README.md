# Muon-Detector
Scintillation Detector for Cosmic Muons, **Monte Carlo Simulations**, Derivations of PDF Distributions through rigorous mathematical analysis

## Detector Characteristics

### Geometry
![Alt text](/images/scintillator_geometry.png)\
The detector comprises of two aligned identical square sheets of side a. The vertical distance between the two sheets is denoted by d. From the mathematical analysis it follows that the ratio d/a determines the behavior of the system.

### Operation
Each cosmic muon must pass through both sheets in order to trigger the recording mechanism and be detected.  

## Muon Trajectory Modelling
![Alt text](/images/muon_trajectory.png)\
Each muon's trajectory is modelled as a straight line. Muons are generated on the top sheet via a uniform distribution on the square of side length a. The velocity vector of the muon is defined using the two angles of a sperical coordinate system $(\theta, \phi)$. The azimuthal angle $\phi$, is sampled from a uniform distribution on $[0,2\pi]$. The polar angle $\theta$, ranges in $[0,\pi/2]$ but follows a $\cos^2\theta$ distribution.
The polar axis is perpendicular to both sheets of the detector and points downwards (from top to bottom). Then $\theta$ is just the angle between this axis and the velocity vector of the muon.

## Monte Carlo Simulation Examples
### Theory vs Experiment
Orange Curves: Theory\
Blue Histograms: Experiment\
=> Perfect Agreement!
![Alt text](/images/d01_theta.png)\
![Alt text](/images/d01_phi.png)\
![Alt text](/images/d01_x.png)

### 2-D Muon Plots
Positions of initial (left side) and detected muons (right side)
![Alt text](/images/d01_2D.png)\
![Alt text](/images/d1_2D.png)\
Fewer muons are detected when the distance between the plates becomes larger.
### 3-D Muon Plots
A typical 3-D representation with fewer sample points looks as follows
![Alt text](/images/d01_3D.png)\
![Alt text](/images/d1_3D.png)\
Again fewer muons are detected when the distance between the plates becomes larger.

