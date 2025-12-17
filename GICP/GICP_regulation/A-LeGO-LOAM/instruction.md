#1 https://pypi.tuna.tsinghua.edu.cn/simple
E6FXYSSHXWMVX6IG6W5RCFK

#2 gtsam, vtk intel file location
set(MKL_INCLUDE_DIR "/home/deep/intel/oneapi/mkl/2022.0.2/include" )
set(MKL_LIBRARIES "/home/deep/intel/oneapi/mkl/2022.0.2/lib/intel64" )

#3 YOLO

# 4 gnuplot
dpkg -L libgnuplot-iostream-dev


# 5 download google drive
https://drive.google.com/u/0/uc?id=1_qUfwUw88rEKitUpt1kjswv7Cv4GPs0b&export=download



ya29.a0AVvZVsq9-6YP452cgDy4iHwRntZ101QgG5dZCV5NhGzqhNSyoXskCibcRcOM8NgATOz7xpZQ2NNrbmgtcQg6Mqf8RUmW0-mn6ADIpEopAEu4ALrv6ZjIYeo-usRph1rgrcNbd7ZL10BrKLrYvETYuvPumLe0aCgYKAV0SARASFQGbdwaI-NJL93WzwAl3pDQkwawvBg0163


curl -H "Authorization: Bearer YYYYY" https://www.googleapis.com/drive/v3/files/XXXXX?alt=media -o ZZZZZ 

curl -H "Authorization: Bearer ya29.a0AVvZVsplZOdHXpQ5TCEL7LxHlBwLGYKs9A38tT6XHcs1aiq2xgLCf7L3c-LBnASSHgiG_5TN8yaKk14NgElqqlq8fQv-b-WEFg8HE_AikpXbaWCBjVZyAYUHjV65cTT-ycomSrwtE15d7l5hBIwDjleCmKtZaCgYKAW4SARASFQGbdwaIjhUd9ASTlkWK0QnPTaaerA0163" https://www.googleapis.com/drive/v3/files/1_qUfwUw88rEKitUpt1kjswv7Cv4GPs0b?alt=media -o drive.zip 


In your command, replace “XXXXX” with the file ID from above, “YYYYY” with the access token from above, and “ZZZZZ” with the file name that will be saved (for example, “myFile.mp4” if you’re downloading an mp4 file).


# ORB-SLAM
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)


# Eigne not ROS package
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)


# ORB_SLAM show

$ cd ORB_SLAM3

# Search the line `ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::MONOCULAR, false);`
# the modify the 4th argument from `false` to `true`.
$ vim Examples/Monocular/mono_euroc.cc

# Rebuild example
$ cd build
$ make

# Run example again (assume that you have already prepared the dataset)
$ cd ../Examples
$ ./Monocular/mono_euroc ../Vocabulary/ORBvoc.txt ./Monocular/EuRoC.yaml PATH_TO_YOUR_MH01_DATASET ./Monocular/EuRoC_TimeStamps/MH01.txt dataset-MH01_mono

# ORB_SLAM no AR
Or, if you don't need the AR function, just delete it in the Cmakelists.

# lego-LOAM #include "opencv2/opencv.hpp" c++14

# ORB_SLAM3 
source ~/.bashrc
rosrun ORB_SLAM3 Mono Vocabulary/ORBvoc.txt Examples/Monocular/TUM2.yaml 
rosbag play -r 1 -s 125 --clock  camera_2023-03-04-15-35-51.bag /camera123/color/image_raw:=/camera/image_raw /cari_points_1:=/lslidar_point_cloud


geometry_msgs::PoseStamped pose;
pose.header.stamp = ros::Time::now();
pose.header.frame_id ="map";

cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t(); // Rotation information
cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3); // translation information
vector<float> q = ORB_SLAM2::Converter::toQuaternion(Rwc);

tf::Transform new_transform;
new_transform.setOrigin(tf::Vector3(twc.at<float>(0, 0), twc.at<float>(0, 1), twc.at<float>(0, 2)));

tf::Quaternion quaternion(q[0], q[1], q[2], q[3]);
new_transform.setRotation(quaternion);

tf::poseTFToMsg(new_transform, pose.pose);
pose_pub.publish(pose);


# Lego LOAM
#include <opencv2/imgproc.hpp>


# A Lego LOAM
roslaunch test2.launch 
source ~/.bashrc 
rosrun ORB_SLAM3 Mono Vocabulary/ORBvoc.txt Examples/Monocular/TUM2.yaml 
rosbag play -r 0.5 -s 1 --clock  camera_2023-03-04-15-35-51.bag /camera123/color/image_raw:=/camera/image_raw /cari_points_1:=/lslidar_point_cloud 


# fast_lio2 slam
roslaunch fast_lio mapping_velodyne.launch
rosbag play -r 1 -s 1 --clock  camera_2023-03-04-15-35-51.bag /camera123/color/image_raw:=/camera/image_raw /cari_points_1:=/velodyne_points /imu_data:=/imu/data


# VINS mono
roslaunch vins_estimator euroc.launch 
roslaunch vins_estimator vins_rviz.launch
rosbag play -r 0.5 -s 1 --clock  camera_2023-03-04-15-35-51.bag /camera123/color/image_raw:=/cam0/image_raw   /imu_data:=/imu0

rosbag play YOUR_PATH_TO_DATASET/MH_01_easy.bag /cam1/image_raw:=/1 /leica/position:=/2

# VINS fusion
roslaunch vins vins_rviz.launch
rosrun vins vins_node ~/catkin_ws/src/VINS-Fusion/config/euroc/euroc_mono_imu_config.yaml 
rosbag play -r 1 -s 1 --clock  camera_2023-03-04-15-35-51.bag /camera123/color/image_raw:=/cam0/image_raw  /imu_data:=/imu0


# realsense_d435i
roslaunch vins vins_rviz.launch
rosrun vins vins_node /home/deep/catkin_ws/src/VINS-Fusion/config/realsense_d435i/realsense_stereo_imu_config.yaml 

rosbag play -r 0.1 -s 1 --clock  camera_2023-03-04-15-35-51.bag /camera123/color/image_raw:=/camera/infra1/image_rect_raw /imu_data:=/camera/imu






# VINS fusion realsense D435i
roslaunch vins vins_rviz.launch
rosrun vins vins_node /home/deep/catkin_ws/src/VINS-Fusion/config/realsense_d435i/realsense_stereo_imu_config.yaml

# VINS Mono realsense D435i
roslaunch realsense2_camera rs_camera.launch 
roslaunch vins_estimator realsense_color.launch 
roslaunch vins_estimator vins_rviz.launch







# ORB_SLAM3 imu
source ~/.bashrc 
rosrun ORB_SLAM3 Mono_Inertial Vocabulary/ORBvoc.txt Examples/Monocular-Inertial/TUM-VI.yaml 
rosrun ORB_SLAM3 Mono_Inertial Vocabulary/ORBvoc.txt Examples/Monocular-Inertial/RealSense_D435i.yaml


# ORB_SLAM3 with d435i
source ~/.bashrc 
rosrun ORB_SLAM3 Mono_Inertial Vocabulary/ORBvoc.txt Examples/Monocular-Inertial/RealSense_D435i.yaml
./mono_inertial_realsense_D435i /home/deep/ORB_SLAM3/Vocabulary/ORBvoc.txt /home/deep/ORB_SLAM3/Examples/Monocular-Inertial/RealSense_D435i.yaml
Camera.width: 1280
Camera.height: 720

# ORB_SLAM3 bag
source ~/.bashrc 
rosrun ORB_SLAM3 Mono Vocabulary/ORBvoc.txt Examples/Monocular/TUM2.yaml 
rosbag play -r 0.1 -s 1 --clock  camera_2023-03-04-15-35-51.bag /camera123/color/image_raw:=/cam0/image_raw  /imu_data:=/imu0
rosbag play MH_01_easy.bag 
rosbag play -r 1 -s 1 --clock  camera_2023-03-04-15-35-51.bag /camera123/color/image_raw:=/camera/color/image_raw
 /cari_points_1:=/lslidar_point_cloud

height: 480
width: 640


# ORB_SLAM3 Mono
ORB_SLAM3/Examples/ROS/ORB_SLAM3/src/ros_mono.cc
cd /home/deep/ORB_SLAM3/Examples_old/ROS/ORB_SLAM3/build
make
rosrun ORB_SLAM3 Mono Vocabulary/ORBvoc.txt Examples/Monocular/TUM2.yaml 

./Mono ../../../Vocabulary/ORBvoc.txt ../../Monocular/TUM1.yaml


rosrun ORB_SLAM3 Mono_Inertial Vocabulary/ORBvoc.txt Examples/Monocular-Inertial/RealSense_D435i.yaml

rqt_image_view

rostopic echo /camera/color/camera_info

---
header: 
  seq: 31
  stamp: 
    secs: 1679294230
    nsecs:  38649797
  frame_id: "camera_color_optical_frame"
height: 720
width: 1280
distortion_model: "plumb_bob"
D: [0.0, 0.0, 0.0, 0.0, 0.0]
K: [913.7951049804688, 0.0, 631.6724853515625, 0.0, 912.202880859375, 356.2579650878906, 0.0, 0.0, 1.0]
R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P: [913.7951049804688, 0.0, 631.6724853515625, 0.0, 0.0, 912.202880859375, 356.2579650878906, 0.0, 0.0, 0.0, 1.0, 0.0]
binning_x: 0
binning_y: 0
roi: 
  x_offset: 0
  y_offset: 0
  height: 0
  width: 0
  do_rectify: False

K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]


./Mono ../../../Vocabulary/ORBvoc.txt ../../Monocular/TUM1.yaml



# ORB_SLAM3 Stereo
source ~/.bashrc 
source /opt/ros/noetic/setup.bash 
./Stereo ../../../Vocabulary/ORBvoc.txt ../../Stereo/D435i.yaml false

K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]

rostopic echo /camera/infra1/camera_info

header: 
  seq: 40
  stamp: 
    secs: 1679294528
    nsecs: 657775879
  frame_id: "camera_infra1_optical_frame"
height: 480
width: 848
distortion_model: "plumb_bob"
D: [0.0, 0.0, 0.0, 0.0, 0.0]
K: [417.8213806152344, 0.0, 427.784423828125, 0.0, 417.8213806152344, 236.55580139160156, 0.0, 0.0, 1.0]
R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P: [417.8213806152344, 0.0, 427.784423828125, 0.0, 0.0, 417.8213806152344, 236.55580139160156, 0.0, 0.0, 0.0, 1.0, 0.0]
binning_x: 0
binning_y: 0
roi: 
  x_offset: 0
  y_offset: 0
  height: 0
  width: 0
  do_rectify: False
---

fx = 417.8213806152344
fy = 417.8213806152344
cx = 427.784423828125
cy = 236.55580139160156

---
^Cheader: 
  seq: 15
  stamp: 
    secs: 1679294590
    nsecs: 336200953
  frame_id: "camera_infra1_optical_frame"
height: 480
width: 848
distortion_model: "plumb_bob"
D: [0.0, 0.0, 0.0, 0.0, 0.0]
K: [417.8213806152344, 0.0, 427.784423828125, 0.0, 417.8213806152344, 236.55580139160156, 0.0, 0.0, 1.0]
R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P: [417.8213806152344, 0.0, 427.784423828125, -20.895565032958984, 0.0, 417.8213806152344, 236.55580139160156, 0.0, 0.0, 0.0, 1.0, 0.0]
binning_x: 0
binning_y: 0
roi: 
  x_offset: 0
  y_offset: 0
  height: 0
  width: 0
  do_rectify: False
---

fx = 417.8213806152344
fy = 417.8213806152344
cx = 427.784423828125
cy = 236.55580139160156


# My ICP SLAM
roslaunch alego test2.launch 

rosbag play -r 0.5 -s 1 --clock  2018-05-18-14-49-12_0.bag  		/velodyne_points:=/lslidar_point_cloud 
rosbag play -r 1   -s 1 --clock  2022-12-12-06-05-20.bag  		/cari_points_1:=/lslidar_point_cloud 
rosbag play -r 1   -s 1 --clock  2023-04-12-16-13-40.bag  		/cari_points_top:=/lslidar_point_cloud 
rosbag play -r 0.5 -s 1 --clock  camera_2023-03-04-15-35-51.bag 	/cari_points_1:=/lslidar_point_cloud 
rosbag play -r 0.5 -s 1 --clock  3d_lidar.bag 			/cari_points_1:=/lslidar_point_cloud 
rosbag play -r 0.5 -s 1 --clock  hdl_400.bag 				/velodyne_points:=/lslidar_point_cloud 
rosbag play -r 0.5 -s 1 --clock  nsh_indoor_outdoor.bag  		/velodyne_points:=/lslidar_point_cloud 

# key in gicp-slam
# 1 map filter setting, make the map stable
# 2 icp to gicp
# 3 the distance of matching of gicp
# 3 make the map update slow
# 4 the Ti position thread and Ti_real position thread is compare and mixing

# lift

python main.py eval_model_iou mini --modelf=/home/deep/lift-splat-shoot/model525000.pt --dataroot=./ --gpuid=0
python main.py viz_model_preds mini --modelf=/home/deep/lift-splat-shoot/model525000.pt --dataroot=./ --map_folder=./mini --gpuid=0


