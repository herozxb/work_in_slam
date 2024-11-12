# 1
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3


# 2
rosbag play -r 1 -s 1 --clock  camera_2023-03-04-15-35-51.bag 	/cari_points_1:=/velodyne_points /imu_data:=/imu/data
rosbag play -r 1 -s 1 --clock  2023-09-06-16-24-11.bag		/cari_points_top:=/velodyne_points /imu:=/imu/data
rosbag play -r 1 -s 1 --clock  2023-09-06-16-24-11.bag		/cari_points_front:=/velodyne_points /imu:=/imu/data
rosbag play -r 1 -s 1 --clock  2023-09-06-16-24-11.bag		/cari_points_back:=/velodyne_points /imu:=/imu/data

rosbag play -r 1 -s 10 --clock --duration 700 2023-09-06-16-24-11.bag /cari_points_back:=/velodyne_points /imu:=/imu/data


rosbag play -r 1 -s 1 --clock  camera_2023-03-04-15-35-51.bag 	/cari_points_1:=/velodyne_points /imu_data:=/imu/data
rosbag play -r 1 -s 1 --clock  staircase.bag				/rslidar_points:=/velodyne_points
rosbag play -r 1 -s 1 --clock  2022-08-30-20-33-52_0.bag		/rslidar_points:=/velodyne_points
rosbag play -r 1 -s 1 --clock  camera_2023-03-04-15-35-51.bag 	/cari_points_1:=/livox/lidar /imu_data:=/livox/imu


# 3
# 3.1 commont it works, and stable for longer time
//const double &pcl_end_time = pcl_beg_time + pcl_out.points.back().curvature / double(1000);
# 3.2 voxel_size: 1.0

