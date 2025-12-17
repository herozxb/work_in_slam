# 1  The primary advantage of this monocular visual-inertial system (VINS) is to observe the metric scale, as well as roll and pitch angles

state estimation is just the getting of x from Ax=b

# 2 The first one is rigorous initialization. Due to the lack of direct distance measurements, it is difficult to directly fuse the monocular visual structure with inertial measuremen

VINS-Mono contains following features:
1) robust initialization procedure that is able to bootstrap the system from unknown initial states;
2) tightly coupled, optimization-based monocular VIO with camera–IMU extrinsic calibration and IMU bias correction;
3) online relocalization and four degrees-of-freedom (DOF) global pose graph optimization;
4) pose graph reuse that can save, load, and merge multiple local pose graphs

robust initialization, relocalization, and pose graph reuse are our technical contribution


# 3 we recover the relative rotation and up-to-scale translation between these two frames using the five-point algorithm [An Efficient Solution to the Five-Point Relative Pose Problem]
Then, we arbitrarily set the scale and triangulate all features observed in these two frames. Based on these triangulated features, a perspective-n-point (PnP) method [35] is performed to estimate poses of all other frames in the window. Finally, a global full bundle adjust-ment [36] is applied to minimize the total reprojection error of all feature observation

The simple understading 
1. initial of 1. SFM, pnp 2. IMU integration 3. least square of initial for A. b_w B. Velocity, Gravity, and Metric Scale
2. least square of A. IMU Measurement Residual and B. Visual Measurement Residual, C. Marginalization


