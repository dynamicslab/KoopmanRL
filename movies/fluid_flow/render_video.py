# Example usage: python -m movies.fluid_flow.render_video

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from custom_envs import FluidFlow
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

p = scipy.io.loadmat('./movies/fluid_flow/data/POD-COEFFS.mat')
alpha = p['alpha'] # (17000, 8)
p2 = scipy.io.loadmat('./movies/fluid_flow/data/POD-MODES.mat')
Xavg = p2['Xavg'] # (89351, 1)
Xdelta = p2['Xdelta'] # (89351, 1)
Phi = p2['Phi'] # (89351, 8)
CCcool = scipy.io.loadmat('./movies/fluid_flow/data/CCcool.mat')['CC'] # (64, 3)
cmap = ListedColormap(CCcool) # Load as a custom color map

# Video constants
fps = 10
num_seconds = 8
scale_factor = 150

# Load in saved trajectory after running
# `python -m movies.generate_trajectories`
#! Change the timestamp or directory to saved path here
video_frame_creation_timestamp = 1714233299
trajectories_npy_path = f"./video_frames/FluidFlow-v0_{video_frame_creation_timestamp}"
trajectories = np.load(f"{trajectories_npy_path}/trajectories.npy") * scale_factor
trajectory_index = 0
trajectory = trajectories[trajectory_index]

# Create frames for our video
# Frames are stored in `snapshots` array
snapshots = []
for k in range(0, trajectory.shape[0], trajectory.shape[0] // (fps*num_seconds)):
    u = Xavg[:,0] + Phi[:,0] * trajectory[k,0] + Phi[:,1] * trajectory[k,1] + Xdelta[:,0] * trajectory[k,2]
    snapshots.append(u.reshape(449,199).T)

theta = np.linspace(0, 2*np.pi, 100)
x = 49+25*np.sin(theta)
y = 99+25*np.cos(theta)

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(figsize=(8,8))

a = snapshots[0]
im = plt.imshow(a, cmap=cmap, clim=(-5,5))

def animate_func(i):
    if i % fps == 0:
        print( '.', end ='' )

    # Update flow
    im.set_array(snapshots[i])
    # Update cylinder
    plt.fill(x, y, color=[.3, .3, .3])
    plt.plot(x, y, 'k', linewidth=1.2)
    return [im]

# Create animation object
anim = FuncAnimation(
    fig,
    animate_func,
    frames=num_seconds * fps,
    interval=200 / fps # Delay between frames in ms
)

# Save the animation to a file
anim.save(f'{trajectories_npy_path}/movie.mp4', writer='ffmpeg', fps=fps*2)
