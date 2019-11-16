# Project Plan

## Feature implementations

### GPU Raytracing

I will be implementing GPU raytracing using CUDA. I will start
with a very simple implementation and progressively optimize.

#### Initial Approach

A kernel will be launched per type of primitive (launching a thread per
primitive is important for performance due to the GPU computing model). Each
thread will correspond to a single pixel and check the intersection distance to
each primitive of that type and find the minimum intersection distance if any.
Each thread will also compute the normal for each minimum distance shape if any.
Then, another kernel will be launched which will determine which exact
primitive among all primitive types has the minimum distance. Then, for
each light, a set of kernels will be launched to determine if there is a shadow.
A fixed number of recursive kernel launches will be run in a loop to compute
reflections using a kernel which computes lighting. Finally, the kernel
for computing lighting will be run at the top level to compute the value of
each pixel

#### Optimizations

It is reasonably likely that this approach won't be fast enough for real
time rendering. Here are some potential specific optimizations I might perform
in addition to general profiling and optimization.

  - Instead of finding the minimum intersection for each potential shadow,
    terminate on the first intersection.
  - Use a bounding volume hierarchy (I think this is the best data structure
    for a GPU implementation) to limit the number of computations.
  - Compute a lower resolution image and use a CNN to upsample (mostly joking
    but maybe...)

#### Resources

https://devblogs.nvidia.com/accelerated-ray-tracing-cuda/
General CUDA docs

### Physics

#### Strategy

I will assume balls stay on the ground plane. This allows for just simulating
spin, friction, sliding, and collisions (between balls and the walls). To
start, I will simply get the time between this frame and the last frame, update
ball positions, spins, and sliding velocities and then check for collisions. It
may be useful to run multiple physics steps per frame or to solve for the point
at which collisions will occur to improve realism. Pocket collisions
will also be checked for to see if a ball goes into a pocket.

#### Resources

https://en.wikipedia.org/wiki/Rolling
https://www.cs.rpi.edu/~cutler/classes/advancedgraphics/S09/final_projects/anderson.pdf

## Program flow

Per each frame, the camera position will be updated, physics will be run
to update positions and orientations. Then, GPU raytracing will be used to
render a frame. If I am able to implement a VR version of this, then 2
frames will be rendered.

The table and balls will be initialized in the standard starting position for
pool balls.

The user will have controls to pause the physics, switch the ball materials
(normal, metallic, glassy) and switch the pool table materials
(normal, mirror). I may also add multiple lighting configurations.

The user will also be able to shoot the cue ball in a direction at a velocity
and spin. At first, I will make these numeric inputs, but I may later
add more game like controls (particularly if VR is used).



## Plan of action

- Write a basic GPU raytracer which can do rendering on the level of intersect
  for spheres, cubes, and cylinders. (By 11/23) 
- Construct a simple scene which contains a very simple table constructed from
  primitives and balls (By 11/23)
- Create ball textures and table textures (By 11/29)
- Create basic physics simulation which ignores spin (By 11/29)
- Expand GPU Raytracing to be fully featured recursive raytracer (By 11/29) 
- Add some basic controls to project (By 11/29)
- Investigate using VR/implement VR support (By 12/06)
- Improve physics simulation (By 12/06)

Potential last steps before completion:

- Improve controls 
- Improve textures and table design (maybe import model and support arbitrary
  polygonal geometry)
- Accelerate raytracing if needed
- Generally improve scene appearance

## Division of labor

Just me.
