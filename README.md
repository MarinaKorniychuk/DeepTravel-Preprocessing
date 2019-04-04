## DeepTravel-Preprocessing Part

The city map (whole road network) is diveded into *N x N*  (in this example *N = 256*) disjoint equal-sized grids.

Script prepares training data for DeepTravel model:

- maps historical trajectories to  grid, generates historical T path and full G path (grid cell sequences that 
correspond to GPS points)

  Travel path G is represented by a sequence of grids it passed by *G = {g1, g2, ..., gn}*.
Recorded GPS points of the path capture the real trajectory T of G in the form *T = {t1, t2, ..., tn}*, 
each GPS point *pi = {xi, yi, ti}* (latitude *xi*, longitude *yi* and time stamp *ti*. 

- extract and aggregate historical speed and time data as short-term and long-term traffict features

  For each grid cell  *gi* speed and time spent within the *gi* data saved to the 5 minutes time bin in short-term features
and in week day bin in long-term features.


- saves generated G and T path for each trajectory and extracted short-term and ling-term traffic features to the separate files.
