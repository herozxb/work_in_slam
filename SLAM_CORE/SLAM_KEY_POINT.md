# 1 	the map
# 2 	range_dis <= 3 * plane.radius
# 3 	dis_to_plane < 30 * sqrt(sigma_l)
# 4 	voxel_size, big and small, down_sample_size

# 5 	logic and i
what is the logic
# 5.1 	IMU to make lidar_cloud
# 5.2 	lidar_cloud the compare with the voxel
what is i
# 5.3 	voxel position is fixed, when it is created
# 5.4 	the lidar_point find the voxel distance

# 5.5 	it means the voxel is the key point


# 6 	after moving in the voxel, the path can recover some
# 7 	tech in control, the math not teach, the math is just the jacobian, and least square
# 8 	This approach ensures that the system remains stable by introducing regularization to the updates, mitigating issues arising from near-singular matrices or numerical instabilities. Adjust the regularization parameter ϵ as needed for your specific application.
# 8.1	near-singular matrices
# 8.2 	numerical instabilities

# 9	Embedding Manifold Structures into Kalman Filters
# 9.1	(“boxplus”) : S x R^n -> S
# 9.2	(“boxminus”) : S x S -> R^n
# 9.3 	delta_x_k+1 = F_x * delta_x_k + F_w * w, linearize of the KF into EKF, with chain rules
# 9.4 	observation function r = D*v + H*delta_x
# 9.5	arg_max_log( N( delta_x ) * N( D*v | delta_x ) ) = arg_min_g( delta_x ) = || r - H * delta_x || ^ 2 + || (“boxminus”) + j^-1 * delta_x|| ^ 2
# 9.6	d( 1/2 * ( D*v - H*delta_x )^T * R^-1 * ( D*v - H*delta_x ) ) / d ( delta_x ) = H^T * R^-1 ( D*v - H*delta_x ) = 0

# 10.0  In a Markov process, the current state depends only on the previous state, which means that the RLS algorithm can be applied using a simple recursive formula that updates the state estimate based on the previous state estimate and the new measurement.
# 10.1	I want to share with you how I understand it. It is like a “Guess Number Game”. First, I was given a number of 10, so I guess the true number is 10. Then I was given the measurements of 20, so I know that what I guessed before which is 10 is too small. So I changed my guess to be 15 for example, this margin of change is up to the confidence of measurements which is the variance σ. Now my guess is 15, which is much closer to 20. I keep “guessing” and updating the true number according to the “running” data. So you can imagine I will get more and more close to the true number.

# 11	change the R is just change the weight of the least square
# 12    e = Rq + t
# 12.1  de/dR = -R*q^ 	de/dt = I


