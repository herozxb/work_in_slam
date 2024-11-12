#include "alego/utility.h"
#include <std_srvs/Empty.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <pcl/filters/filter.h>
#include <pcl/filters/crop_box.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/gicp.h>

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>

using namespace boost::interprocess;
using namespace std;
using namespace gtsam;

#include <iostream>
#include <stdint.h>
#include <vector>
#include <random>
#include <cmath>

#include "../include/EKF/types.h"
#include "../include/EKF/kalman_filter.h"


#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fgicp_mt.hpp>


static constexpr size_t DIM_X{ 1 };
static constexpr size_t DIM_Z{ 2 };

static kf::KalmanFilter<DIM_X, DIM_Z> kalmanfilter;
static kf::KalmanFilter<DIM_X, DIM_Z> kalmanfilter_x;
static kf::KalmanFilter<DIM_X, DIM_Z> kalmanfilter_y;
static kf::KalmanFilter<DIM_X, DIM_Z> kalmanfilter_z;

double x = 0;
double y = 0;
double z = 0;

class LM
{
private:

  //initilize the ros node
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  
  //initialize the Subscriber
  ros::Subscriber sub_surf_last_;
  ros::Subscriber sub_corner_last_;
  ros::Subscriber sub_outlier_last_;
  ros::Subscriber sub_laser_odom_;
  
  ros::Subscriber sub_segmented_cloud_;


  //initialize the Publisher
  ros::Publisher pub_cloud_surround_;
  ros::Publisher pub_odom_aft_mapped_;
  ros::Publisher pub_keyposes_;

  ros::Publisher pub_history_keyframes_;
  ros::Publisher pub_icp_keyframes_;
  ros::Publisher pub_recent_keyframes_;

  //initialze the ServiceServer
  ros::ServiceServer srv_save_map_;


  //initilize the TransformBroadcaster
  tf::TransformBroadcaster tf_broadcaster_;


  //initialze the NonlinearFactorGraph
  NonlinearFactorGraph gtSAMgraph_;
  
  //intilize the estimater
  Values init_estimate_;
  Values opt_estimate_;
  
  //initialze the ISAM
  ISAM2 *isam_;
  
  //initialze the estimater  
  Values isam_cur_estimate_;
  
  //initialze the noise 
  noiseModel::Diagonal::shared_ptr prior_noise_;
  noiseModel::Diagonal::shared_ptr odom_noise_;
  noiseModel::Diagonal::shared_ptr constraint_noise_;


  //initialze the loop closed 
  // 回环检测相关
  bool loop_closed_;
  
  //initialze the latest and closest history id
  int latest_history_frame_id_;
  int closest_history_frame_id_;
  
  //initialze the latest and closest history frame
  PointCloudT::Ptr latest_keyframe_;
  PointCloudT::Ptr near_history_keyframes_;


  //set the mutex
  std::mutex mtx_;
  
  //set the thread of loop, main, visualization
  std::thread loop_thread_;
  std::thread main_thread_;
  std::thread visualize_thread_;


  //set search radius, search number, fitness score, loop closure enbabled
  float history_search_radius_; // 回环检测参数
  int history_search_num_;
  
  float history_fitness_score_;
  bool loop_closure_enabled_;


  //set the surface, corner and outlier 
  PointCloudT::Ptr surf_last_;
  PointCloudT::Ptr corner_last_;
  PointCloudT::Ptr outlier_last_;

  //set the surface, corner and outlier frame
  vector<PointCloudT::Ptr> corner_frames_;
  vector<PointCloudT::Ptr> surf_frames_;
  vector<PointCloudT::Ptr> outlier_frames_;


  //set the key pose 
  PointCloudT::Ptr cloud_keyposes_3d_;
  pcl::PointCloud<PointTypePose>::Ptr cloud_keyposes_6d_;

  //get the corner from the map
  PointCloudT::Ptr corner_from_map_;
  PointCloudT::Ptr surf_from_map_;
  // PointCloudT::Ptr outlier_from_map_;

  //get the corner from the map ds  
  PointCloudT::Ptr corner_from_map_ds_;
  PointCloudT::Ptr surf_from_map_ds_;

  //set teh laser corner, surface, outlier, surface total, corner ds, surface ds, outlier ds, surface total ds
  PointCloudT::Ptr laser_corner_;
  PointCloudT::Ptr laser_surf_;
  PointCloudT::Ptr laser_outlier_;
  PointCloudT::Ptr laser_surf_total_;
  PointCloudT::Ptr laser_corner_ds_;
  PointCloudT::Ptr laser_surf_ds_;
  PointCloudT::Ptr laser_outlier_ds_;
  PointCloudT::Ptr laser_surf_total_ds_;
  
  
  PointCloudT::Ptr laser_segment_;
  
  //time of the laser corner, surface, outlier, odometry
  double time_laser_corner_;
  double time_laser_surf_;
  double time_laser_outlier_;
  double time_laser_odom_;
  
  //bool of the laser corner, surface, outlier, odometry
  bool new_laser_corner_;
  bool new_laser_surf_;
  bool new_laser_outlier_;
  bool new_laser_odom_;

  //set the kdtree of corner, surface, key pose 
  pcl::KdTreeFLANN<PointT>::Ptr kd_corner_map_;
  pcl::KdTreeFLANN<PointT>::Ptr kd_surf_map_;
  pcl::KdTreeFLANN<PointT>::Ptr kd_keyposes_;
  
  //point index and point distance
  vector<int> point_idx_;
  vector<float> point_dist_;


  //VoxelGrid of corner, surface, outlier, keypose, history keyframe
  pcl::VoxelGrid<PointT> ds_corner_;
  pcl::VoxelGrid<PointT> ds_surf_;
  pcl::VoxelGrid<PointT> ds_outlier_;
  pcl::VoxelGrid<PointT> ds_keyposes_;
  pcl::VoxelGrid<PointT> ds_history_keyframes_;

  //mini distance of keyframe
  double min_keyframe_dist_;

  // 回环检测使用
  //for the loop closure of the corner, surface, outlier
  deque<PointCloudT::Ptr> recent_corner_keyframes_;
  deque<PointCloudT::Ptr> recent_surf_keyframes_;
  deque<PointCloudT::Ptr> recent_outlier_keyframes_;
  
  // keyframe search number
  int recent_keyframe_search_num_;
  
  // keyframe id
  int latest_frame_id_;
  
  // matirx of correction
  Eigen::Matrix4d correction_;

  // 无回环检测使用
  // surround key pose
  PointCloudT::Ptr surround_keyposes_;
  PointCloudT::Ptr surround_keyposes_ds_;
  
  // surround key pose id
  vector<int> surround_exist_keypose_id_;
  
  //vector to store corner, surface, outlier, key frames
  vector<PointCloudT::Ptr> surround_corner_keyframes_;
  vector<PointCloudT::Ptr> surround_surf_keyframes_;
  vector<PointCloudT::Ptr> surround_outlier_keyframes_;
  
  // the key frame search threshold
  double surround_keyframe_search_radius_;

  //ceres params_
  double params_[6];
  
  //translate, rotation from map to the odometry
  Eigen::Vector3d t_map2odom_;
  Eigen::Quaterniond q_map2odom_;
  //translate, rotation from the odometry to the laser  
  Eigen::Vector3d t_odom2laser_;
  Eigen::Quaterniond q_odom2laser_;
  //translate, rotation from map to the laser  
  Eigen::Vector3d t_map2laser_;
  Eigen::Quaterniond q_map2laser_;

  PointCloudT::Ptr cloud_template;
  
  int counter = 0;
  int counter_stable_map = 0;
  
  ros::Subscriber sub_pc_;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pre;
  
  pcl::PointCloud<pcl::PointXYZ>::Ptr map;
  pcl::PointCloud<pcl::PointXYZ>::Ptr map_final;
  pcl::PointCloud<pcl::PointXYZ>::Ptr map_final_boxxed;
  pcl::PointCloud<pcl::PointXYZ>::Ptr map_final_boxxed_2;
  pcl::PointCloud<pcl::PointXYZ>::Ptr map_fixed;
    
  Eigen::Matrix4f Ti;
  Eigen::Matrix4f Ti_of_map;
  Eigen::Matrix4f Ti_of_map_real;
  Eigen::Matrix4f Ti_translation;
  Eigen::Matrix4f Ti_real;
  
  
  ros::Publisher pub_odom_aft_mapped_2;
  ros::Publisher pub_odom_aft_mapped_3;   
  ros::Publisher pub_odom_aft_mapped_kalman; 
    
public:

  LM(ros::NodeHandle nh) : nh_(nh)
  {
    onInit();
  }
  
  void onInit()
  {
    TicToc t_init;

    ROS_INFO("--------- LaserMapping init --------------");

    //initialize all the array
    surf_last_.reset(new PointCloudT);
    corner_last_.reset(new PointCloudT);
    outlier_last_.reset(new PointCloudT);
    cloud_keyposes_3d_.reset(new PointCloudT);
    cloud_keyposes_6d_.reset(new pcl::PointCloud<PointTypePose>);
    corner_from_map_.reset(new PointCloudT);
    surf_from_map_.reset(new PointCloudT);
    // outlier_from_map_.reset(new PointCloudT);
    corner_from_map_ds_.reset(new PointCloudT);
    surf_from_map_ds_.reset(new PointCloudT);
    laser_corner_.reset(new PointCloudT);
    laser_surf_.reset(new PointCloudT);
    laser_outlier_.reset(new PointCloudT);
    laser_surf_total_.reset(new PointCloudT);
    laser_corner_ds_.reset(new PointCloudT);
    laser_surf_ds_.reset(new PointCloudT);
    laser_outlier_ds_.reset(new PointCloudT);
    laser_surf_total_ds_.reset(new PointCloudT);
    kd_surf_map_.reset(new pcl::KdTreeFLANN<PointT>);
    kd_corner_map_.reset(new pcl::KdTreeFLANN<PointT>);
    kd_keyposes_.reset(new pcl::KdTreeFLANN<PointT>);

    laser_segment_.reset(new PointCloudT);
    
    //initialize bool
    new_laser_surf_ = new_laser_corner_ = new_laser_outlier_ = new_laser_corner_ = false;

    //VoxelGrid set the filter size 
    ds_corner_.setLeafSize(0.4, 0.4, 0.4);
    ds_surf_.setLeafSize(0.8, 0.8, 0.8);
    ds_outlier_.setLeafSize(1.0, 1.0, 1.0);
    ds_keyposes_.setLeafSize(1.0, 1.0, 1.0);
    ds_history_keyframes_.setLeafSize(0.4, 0.4, 0.4);

    // set the min distance
    min_keyframe_dist_ = 1.0;

    // set surround_keyposes_ds_
    surround_keyposes_ds_.reset(new PointCloudT);
    //surround_keyposes_ds_.reset(new PointCloudT);

    // keyframe search number equals 50
    recent_keyframe_search_num_ = 50;
    
    // keyframe search radius equals 50.
    surround_keyframe_search_radius_ = 50.;
    
    //set the latest frame id to -1
    latest_frame_id_ = -1;

    //set the params to zero
    for (int i = 0; i < 6; ++i)
    {
      params_[i] = 0.;
    }
    
    //set the transformation to zero
    t_map2odom_.setZero();
    q_map2odom_.setIdentity();
    t_odom2laser_.setZero();
    q_odom2laser_.setIdentity();
    t_map2laser_.setZero();
    q_map2laser_.setIdentity();

    //ISAM is a optimazing algorithm of gtsam
    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam_ = new ISAM2(parameters);
    
    // vector of gtsam
    gtsam::Vector Vector6(6);
    Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-6;
    
    //variance of the noise
    prior_noise_ = noiseModel::Diagonal::Variances(Vector6);
    odom_noise_ = noiseModel::Diagonal::Variances(Vector6);


    //for loop closure
    loop_closed_ = false;
    
    //latest history frame id
    latest_history_frame_id_ = -1;
    
    //latest key frame
    latest_keyframe_.reset(new PointCloudT);
    
    //nearest history key frame
    near_history_keyframes_.reset(new PointCloudT);
    
    //search radius
    history_search_radius_ = 10.;
    //search number
    history_search_num_ = 25;
    
    //fitness score
    history_fitness_score_ = 0.3;
    
    //loop closre enabled or not
    loop_closure_enabled_ = true;
    correction_.setIdentity();

    //all the publiser
    pub_cloud_surround_ = nh_.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 10);
    pub_odom_aft_mapped_ = nh_.advertise<nav_msgs::Odometry>("/odom_aft_mapped", 10);
    pub_keyposes_ = nh_.advertise<sensor_msgs::PointCloud2>("/keyposes", 10);
    pub_recent_keyframes_ = nh_.advertise<sensor_msgs::PointCloud2>("/recent_keyframes", 10);
    pub_history_keyframes_ = nh_.advertise<sensor_msgs::PointCloud2>("/history_keyframes", 10);
    pub_icp_keyframes_ = nh_.advertise<sensor_msgs::PointCloud2>("/icp_keyframes", 10);
    //srv_save_map_ = nh_.advertiseService("/save_map", &LM::saveMapCB, this);
    
    
    //all the subsriber
    sub_surf_last_ = nh_.subscribe<sensor_msgs::PointCloud2>("/surf_last", 10, boost::bind(&LM::surfLastHandler, this, _1));
    sub_corner_last_ = nh_.subscribe<sensor_msgs::PointCloud2>("/corner_last", 10, boost::bind(&LM::cornerLastHandler, this, _1));
    sub_outlier_last_ = nh_.subscribe<sensor_msgs::PointCloud2>("/outlier", 10, boost::bind(&LM::outlierLastHandler, this, _1));
    sub_laser_odom_ = nh_.subscribe<nav_msgs::Odometry>("/odom/lidar", 10, boost::bind(&LM::laserOdomHandler, this, _1));
    
    //sub_segmented_cloud_ = nh_.subscribe<sensor_msgs::PointCloud2>("/segmented_cloud", 10, boost::bind(&LM::segmented_cloudHandler, this, _1));

    //thread for loop
    //loop_thread_ = std::thread(&LM::loopClosureThread, this);
    //thread for visualizaton
    //visualize_thread_ = std::thread(&LM::visualizeGlobalMapThread, this);

    ROS_INFO("LaserMapping onInit end: %.3fms", t_init.toc());
    
    cloud_template.reset(new PointCloudT);
    
    sub_pc_ = nh_.subscribe<sensor_msgs::PointCloud2>("/lslidar_point_cloud", 10, boost::bind(&LM::main_callback, this, _1));
    
    cloud_pre.reset(new pcl::PointCloud<pcl::PointXYZ>);
    map.reset(new pcl::PointCloud<pcl::PointXYZ>);
    map_final.reset(new pcl::PointCloud<pcl::PointXYZ>);
    map_final_boxxed.reset(new pcl::PointCloud<pcl::PointXYZ>);    
    map_final_boxxed_2.reset(new pcl::PointCloud<pcl::PointXYZ>);   
    map_fixed.reset(new pcl::PointCloud<pcl::PointXYZ>);    
    
    Ti = Eigen::Matrix4f::Identity ();
    Ti_of_map = Eigen::Matrix4f::Identity ();
    Ti_of_map_real = Eigen::Matrix4f::Identity ();
    Ti_translation = Eigen::Matrix4f::Identity ();
    Ti_real = Eigen::Matrix4f::Identity ();
    
    
    pub_odom_aft_mapped_2 = nh_.advertise<nav_msgs::Odometry>("/odom_aft_mapped_2", 10);
    pub_odom_aft_mapped_3 = nh_.advertise<nav_msgs::Odometry>("/odom_aft_mapped_3", 10);
    pub_odom_aft_mapped_kalman = nh_.advertise<nav_msgs::Odometry>("/odom_aft_mapped_kalman", 10);
  }
  
  
  
  //process the input cloud
  void main_callback(const sensor_msgs::PointCloud2ConstPtr &msg)
  {
  
  
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>(5,1));
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_boxxed_for_local (new pcl::PointCloud<pcl::PointXYZ>(5,1));
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_filtered (new pcl::PointCloud<pcl::PointXYZ>(5,1));
    //PointCloudT::Ptr cloud_in(new PointCloudT());
    pcl::fromROSMsg(*msg, *cloud_in);
    // ROS_INFO("cloud_in size: %d", cloud_in->points.size());

    if( counter == 0 )
    {
        // Initialize the map
    	*map_final = *cloud_in;
    	//*map_fixed = *cloud_in;
    	
    }
    else
    {
        // 1.0 remove the Nan points
	// remove the NaN points and remove the points out of distance
	std::vector<int> indices;
	pcl::removeNaNFromPointCloud(*cloud_in, *cloud_in, indices);
	
	
	// 1.1 Voxel the cloud
	
	// Create the filtering object
	pcl::VoxelGrid<pcl::PointXYZ> sor_input;
	sor_input.setInputCloud (cloud_in);
	sor_input.setLeafSize (0.1f, 0.1f, 0.1f);
	sor_input.filter (*cloud_in_filtered);
	
	// 1.2 boxxed the cloud
	
	//cout<<"==================cropbox_cloud_in====================="<<endl;
	*cloud_in_boxxed_for_local = *cloud_in_filtered;
	
        pcl::CropBox<pcl::PointXYZ> boxFilter_for_in;
	float x_min_for_in = - 50, y_min_for_in = - 50, z_min_for_in = + 0 ;
	float x_max_for_in = + 50, y_max_for_in = + 50, z_max_for_in = + 50;
	
	boxFilter_for_in.setMin(Eigen::Vector4f(x_min_for_in, y_min_for_in, z_min_for_in, 1.0));
	boxFilter_for_in.setMax(Eigen::Vector4f(x_max_for_in, y_max_for_in, z_max_for_in, 1.0));

	boxFilter_for_in.setInputCloud(cloud_in_boxxed_for_local);
	boxFilter_for_in.filter(*cloud_in_boxxed_for_local);
	

	/*
	// 2.0 gicp of the cloud_in voxeled and pre cloud
	pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
	gicp.setMaxCorrespondenceDistance(1.0);
	gicp.setTransformationEpsilon(0.001);
	gicp.setMaximumIterations(1000);
	
	gicp.setInputSource(cloud_in_filtered);
	gicp.setInputTarget(cloud_pre);
	pcl::PointCloud<pcl::PointXYZ> Final;
	gicp.align(Final);
	//*/
	
	pcl::PointCloud<pcl::PointXYZ> Final;
	
	if(counter < 3 )
	{
		// 1st icp for cloud_in [now] and cloud_pre [last]
		pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
		icp.setInputSource(cloud_in_filtered);
		icp.setInputTarget(cloud_pre);
		icp.align(Final);
			
		// 2.2 get the x,y,z of the odometry of the trajectory
		Ti = icp.getFinalTransformation () * Ti;
	}
	else
	{
		// 2.0 gicp of the cloud_in voxeled and pre cloud
		pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
		gicp.setMaxCorrespondenceDistance(1.0);
		gicp.setTransformationEpsilon(0.001);
		gicp.setMaximumIterations(1000);
		
		gicp.setInputSource(cloud_in_filtered);
		gicp.setInputTarget(cloud_pre);
		gicp.align(Final);
		Ti = gicp.getFinalTransformation () * Ti;
	
	}
	//*/
	//std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
	//std::cout << icp.getFinalTransformation() << std::endl;


	// 2.1 output the cloud_in_boxxed_for_local and cloud_pre
	//brown is the input cloud of right now frame
	sensor_msgs::PointCloud2Ptr msg_second(new sensor_msgs::PointCloud2);
	//cout<<"==================cloud_in====================="<<endl;
	pcl::toROSMsg(*cloud_in_boxxed_for_local, *msg_second);
	msg_second->header.stamp.fromSec(0);
	msg_second->header.frame_id = "map";
	pub_cloud_surround_.publish(msg_second);

/*
	//blue is cloud_pre, the last frame
	pcl::toROSMsg(*cloud_pre, *msg_second);
	msg_second->header.stamp.fromSec(0);
	msg_second->header.frame_id = "map";
	pub_recent_keyframes_.publish(msg_second); 
//*/

	//cout<<"==================Final====================="<<endl;
	//pcl::toROSMsg(Final, *msg_second);
	//msg_second->header.stamp.fromSec(0);
	//msg_second->header.frame_id = "map";
	//pub_history_keyframes_.publish(msg_second);    




	
	Ti_translation(0,3) = Ti(0,3);
	Ti_translation(1,3) = Ti(1,3);
	Ti_translation(2,3) = Ti(2,3);
	
	/*
	
	pcl::PointCloud<pcl::PointXYZ>::Ptr output (new pcl::PointCloud<pcl::PointXYZ>(5,1));
	pcl::transformPointCloud (*cloud_in, *output, Ti);    


	pcl::toROSMsg(*output, *msg_second);
	msg_second->header.stamp.fromSec(0);
	msg_second->header.frame_id = "map";
	pub_history_keyframes_.publish(msg_second);  
	
	//*/	
	
	// 3.0 voxed the map_final
	// Create the filtering object
	pcl::VoxelGrid<pcl::PointXYZ> sor;
	sor.setInputCloud (map_final);
	sor.setLeafSize (0.1f, 0.1f, 0.1f);
	sor.filter (*map_final);
	
	
	if( counter_stable_map % 20 == 0 )
	{
		// 3.1 boxxed the map_final
		pcl::CropBox<pcl::PointXYZ> boxFilter;
		float x_min = Ti(0,3) - 50, y_min = Ti(1,3) - 50, z_min = Ti(2,3) - 0;
		float x_max = Ti(0,3) + 50, y_max = Ti(1,3) + 50, z_max = Ti(2,3) + 50;

		boxFilter.setMin(Eigen::Vector4f(x_min, y_min, z_min, 1.0));
		boxFilter.setMax(Eigen::Vector4f(x_max, y_max, z_max, 1.0));

		boxFilter.setInputCloud(map_final);
		boxFilter.filter(*map_final_boxxed);
	}
	counter_stable_map++;
        //3.2 output the map_final boxxed
	//red is the map_final
	pcl::toROSMsg(*map_final, *msg_second);
	msg_second->header.stamp.fromSec(0);
	msg_second->header.frame_id = "map";
	pub_icp_keyframes_.publish(msg_second);  
	
	/*
	//cout<<"==================FinalTransformation_of_map====================="<<endl;
	// 2nd icp for cloud_in_cropbox_for_local and map_final_boxxed
        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp_for_map;
	icp_for_map.setInputSource(cloud_in_cropbox_for_local);
	icp_for_map.setInputTarget(map_final_boxxed);
	//pcl::PointCloud<pcl::PointXYZ> Final;
	icp_for_map.align(Final);
	//std::cout << "has converged:" << icp_for_map.hasConverged() << " score: " << icp_for_map.getFitnessScore() << std::endl;
	//std::cout << icp_for_map.getFinalTransformation() << std::endl;
	//*/
	
	//3.3 get the gicp of the cloud in boxxed and the map boxxed
	
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_boxxed_translate_to_near_mapboxxed (new pcl::PointCloud<pcl::PointXYZ>(5,1));
	pcl::transformPointCloud (*cloud_in_boxxed_for_local, *cloud_in_boxxed_translate_to_near_mapboxxed, Ti);   

	//blue is cloud_pre, the last frame
	pcl::toROSMsg(*cloud_in_boxxed_translate_to_near_mapboxxed, *msg_second);
	msg_second->header.stamp.fromSec(0);
	msg_second->header.frame_id = "map";
	pub_recent_keyframes_.publish(msg_second); 
		
	
	pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp_for_map;
	gicp_for_map.setMaxCorrespondenceDistance(10.0);
	gicp_for_map.setTransformationEpsilon(0.01);
	gicp_for_map.setRotationEpsilon(0.01);
	gicp_for_map.setMaximumIterations(1000);
	
	gicp_for_map.setInputSource(cloud_in_boxxed_translate_to_near_mapboxxed);
	gicp_for_map.setInputTarget(map_final_boxxed);
	gicp_for_map.align(Final);
	
	Ti_of_map = gicp_for_map.getFinalTransformation (); // * Ti_of_map;
	
	//Eigen::Matrix4f rotationMatrix = gicp_for_add_to_map_final.getFinalTransformation ();
	Eigen::Matrix4f rotation_matrix = Ti_of_map;

	//double roll = atan2( rotationMatrix(2,1),rotationMatrix(2,2) )/3.1415926*180;
	//std::cout<<"roll is " << roll <<std::endl;
	//double pitch = atan2( -rotationMatrix(2,0), std::pow( rotationMatrix(2,1)*rotationMatrix(2,1) +rotationMatrix(2,2)*rotationMatrix(2,2) ,0.5  )  )/3.1415926*180;
	//std::cout<<"pitch is " << pitch <<std::endl;
	double yaw_of_cloud_ti_to_map = atan2( rotation_matrix(1,0),rotation_matrix(0,0) )/3.1415926*180;
	std::cout<<"yaw is " << yaw_of_cloud_ti_to_map <<std::endl;
	
	//cout<<"===========Ti_of_map=============="<<endl;
	//cout<<Ti_of_map<<endl;
	//cout<<"===========Ti_of_map_real=============="<<endl;
	//cout<<Ti_of_map_real<<endl;
	//Ti_of_map_real = Ti_of_map * Ti_of_map_real;
	//Ti_of_map is the right now translation of the input cloud to the cloud, but not add what is start from point

	Ti_real = Ti_of_map * Ti;
		
	if( abs( Ti_of_map(0,3) ) > 0.2 || abs( Ti_of_map(1,3) ) > 0.2 || abs( yaw_of_cloud_ti_to_map ) > 1 )
	{
	        cout<<"===========Ti_real=============="<<endl;
	        cout<<Ti_of_map<<endl;
		cout<<Ti<<endl;
		cout<<Ti_real<<endl;
		Ti = Ti_real;

	}

	
	
	/*
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_map_translate (new pcl::PointCloud<pcl::PointXYZ>(5,1));
	pcl::transformPointCloud (*cloud_in_filtered, *cloud_map_translate, Ti_real);    
	

	
	pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp_final_and_map;
	gicp_final_and_map.setMaxCorrespondenceDistance(1.0);
	gicp_final_and_map.setTransformationEpsilon(0.001);
	gicp_final_and_map.setMaximumIterations(1000);
	
	gicp_final_and_map.setInputSource(cloud_map_translate);
	gicp_final_and_map.setInputTarget(map_final);
	gicp_final_and_map.align(Final);
	
	/*
	//3rd icp for cloud_map_translate and map_final
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp_final_and_map;
	icp_final_and_map.setInputSource(cloud_map_translate);
	icp_final_and_map.setInputTarget(map_final);
	//pcl::PointCloud<pcl::PointXYZ> Final;
	icp_final_and_map.align(Final);
	//std::cout << "has converged:" << icp_final_and_map.hasConverged() << " score: " << icp_final_and_map.getFitnessScore() << std::endl;
	//std::cout << icp_final_and_map.getFinalTransformation() << std::endl;
	//*/
	
	/*
	pcl::PointCloud<pcl::PointXYZ>::Ptr Final_translate (new pcl::PointCloud<pcl::PointXYZ>(5,1));
	pcl::transformPointCloud (*cloud_map_translate, *Final_translate, gicp_final_and_map.getFinalTransformation ()); 
	//*/
	 

	pcl::PointCloud<pcl::PointXYZ>::Ptr Final_cloud_translate (new pcl::PointCloud<pcl::PointXYZ>(5,1));
	pcl::transformPointCloud (*cloud_in_filtered, *Final_cloud_translate, Ti_real); 
	//*/

	

	
	if( counter % 5 == 0 )
        {
        
            
	    // 3.1 boxxed the map_final
	    pcl::CropBox<pcl::PointXYZ> box_filter;
	    float x_min = Ti_real(0,3) - 50, y_min = Ti_real(1,3) - 50, z_min = Ti_real(2,3) - 50;
	    float x_max = Ti_real(0,3) + 50, y_max = Ti_real(1,3) + 50, z_max = Ti_real(2,3) + 50;

	    box_filter.setMin(Eigen::Vector4f(x_min, y_min, z_min, 1.0));
	    box_filter.setMax(Eigen::Vector4f(x_max, y_max, z_max, 1.0));

	    box_filter.setInputCloud(map_final);
	    box_filter.filter(*map_final_boxxed_2);
	    //*/
        
	    pcl::PointCloud<pcl::PointXYZ> Final_for_add_to_map;
	    
	    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp_for_add_to_map_final;
	    gicp_for_add_to_map_final.setMaxCorrespondenceDistance(5.0);
	    gicp_for_add_to_map_final.setTransformationEpsilon(0.001);
	    gicp_for_add_to_map_final.setMaximumIterations(1000);

	    gicp_for_add_to_map_final.setInputSource(Final_cloud_translate);
	    gicp_for_add_to_map_final.setInputTarget(map_final_boxxed_2);
	    gicp_for_add_to_map_final.align(Final_for_add_to_map);
	    
            
            
            /*
	    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp_for_add_to_map_final;
	    icp_for_add_to_map_final.setInputSource(Final_translate);
	    icp_for_add_to_map_final.setInputTarget(map_final);
	   //pcl::PointCloud<pcl::PointXYZ> Final;
	    icp_for_add_to_map_final.align(Final_for_add_to_map);
            //*/
            
            std::cout << "============icp==============" <<std::endl;
            std::cout << gicp_for_add_to_map_final.getFinalTransformation () <<std::endl;
            
            /*
	    //Eigen::Matrix4f rotationMatrix = gicp_for_add_to_map_final.getFinalTransformation ();
	    Eigen::Matrix4f rotationMatrix = Ti_of_map;

	    double roll = atan2( rotationMatrix(2,1),rotationMatrix(2,2) )/3.1415926*180;
	    std::cout<<"roll is " << roll <<std::endl;
	    double pitch = atan2( -rotationMatrix(2,0), std::pow( rotationMatrix(2,1)*rotationMatrix(2,1) +rotationMatrix(2,2)*rotationMatrix(2,2) ,0.5  )  )/3.1415926*180;
	    std::cout<<"pitch is " << pitch <<std::endl;
	    double yaw = atan2( rotationMatrix(1,0),rotationMatrix(0,0) )/3.1415926*180;
	    std::cout<<"yaw is " << yaw <<std::endl;
            //*/
            //if( abs( gicp_for_add_to_map_final.getFinalTransformation ()(0,3) ) < 0.9 && abs( gicp_for_add_to_map_final.getFinalTransformation ()(1,3) ) < 0.9 && abs( gicp_for_add_to_map_final.getFinalTransformation ()(2,3) ) < 0.9 && abs(yaw) < 5 && abs(roll) < 2 && abs(pitch) < 2 )//&& abs( gicp_for_add_to_map_final.getFinalTransformation ()(2,3) ) < 0.5 )
            
            
            if( abs( gicp_for_add_to_map_final.getFinalTransformation ()(0,3) ) < 0.2 && abs( gicp_for_add_to_map_final.getFinalTransformation ()(1,3) ) < 0.2 ) //  && abs( gicp_for_add_to_map_final.getFinalTransformation ()(2,3) ) < 0.01 )
            {
            
		Ti = gicp_for_add_to_map_final.getFinalTransformation () * Ti;
		
		*map_final += Final_for_add_to_map;
		
		pcl::toROSMsg(Final_for_add_to_map, *msg_second);
		msg_second->header.stamp.fromSec(0);
		msg_second->header.frame_id = "map";
		pub_history_keyframes_.publish(msg_second);   
	    }
	    
	}
	
	/*
	if( sqrt( Ti_real(0,3)*Ti_real(0,3) + Ti_real(1,3)*Ti_real(1,3) ) > 50 )
	{
		Ti = Eigen::Matrix4f::Identity ();
		Ti_of_map = Eigen::Matrix4f::Identity ();
		Ti_real = Eigen::Matrix4f::Identity ();
		
		counter = 0;
		counter_stable_map = 0;
		
		map_final.reset(new pcl::PointCloud<pcl::PointXYZ>);
		map_final_boxxed.reset(new pcl::PointCloud<pcl::PointXYZ>);    
		map_final_boxxed_2.reset(new pcl::PointCloud<pcl::PointXYZ>); 
	
	
	}
	//*/
	
	
    }
    
    counter++;
    
    *cloud_pre = *cloud_in_filtered;
  
  //*/   
  
  }

  //get the surface information
  void surfLastHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
  {
    laser_surf_->clear();
    pcl::fromROSMsg(*msg, *laser_surf_);
    time_laser_surf_ = msg->header.stamp.toSec();
    new_laser_surf_ = true;
  }
  
  //get the corner information
  void cornerLastHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
  {
    laser_corner_->clear();
    pcl::fromROSMsg(*msg, *laser_corner_);
    time_laser_corner_ = msg->header.stamp.toSec();
    new_laser_corner_ = true;
  }
  
      
  //get the outlier information
  void outlierLastHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
  {
    laser_outlier_->clear();
    pcl::fromROSMsg(*msg, *laser_outlier_);
    time_laser_outlier_ = msg->header.stamp.toSec();
    new_laser_outlier_ = true;
    

    
  }
  
  //get the odometry information
  void laserOdomHandler(const nav_msgs::OdometryConstPtr &msg)
  {
  
    // get the time
    time_laser_odom_ = msg->header.stamp.toSec();
    new_laser_odom_ = true;
    
    //odometry to laser
    t_odom2laser_(0) = msg->pose.pose.position.x;
    t_odom2laser_(1) = msg->pose.pose.position.y;
    t_odom2laser_(2) = msg->pose.pose.position.z;
    q_odom2laser_.w() = msg->pose.pose.orientation.w;
    q_odom2laser_.x() = msg->pose.pose.orientation.x;
    q_odom2laser_.y() = msg->pose.pose.orientation.y;
    q_odom2laser_.z() = msg->pose.pose.orientation.z;
    
    //map to laser
    t_map2laser_ = q_map2odom_ * t_odom2laser_ + t_map2odom_;
    q_map2laser_ = q_map2odom_ * q_odom2laser_;
    
    //publish the message of map to laser
    if (pub_odom_aft_mapped_.getNumSubscribers() > 0)
    {
      nav_msgs::OdometryPtr msg(new nav_msgs::Odometry);
      msg->header.stamp.fromSec(time_laser_odom_);
      msg->header.frame_id = "map";
      msg->child_frame_id = "/laser";
      msg->pose.pose.position.x = t_map2laser_.x();
      msg->pose.pose.position.y = t_map2laser_.y();
      msg->pose.pose.position.z = t_map2laser_.z();
      msg->pose.pose.orientation.w = q_map2laser_.w();
      msg->pose.pose.orientation.x = q_map2laser_.x();
      msg->pose.pose.orientation.y = q_map2laser_.y();
      msg->pose.pose.orientation.z = q_map2laser_.z();
      pub_odom_aft_mapped_.publish(msg);
    }
    
    //publish the transformation
    //tf::Transform tf_m2o;
    //tf_m2o.setOrigin(tf::Vector3(t_map2odom_.x(), t_map2odom_.y(), t_map2odom_.z()));
    //tf_m2o.setRotation(tf::Quaternion(q_map2odom_.x(), q_map2odom_.y(), q_map2odom_.z(), q_map2odom_.w()));
    //tf_broadcaster_.sendTransform(tf::StampedTransform(tf_m2o, ros::Time::now(), "map", "/odom"));
  }

  
  void downsampleCurrentScan()
  {
    TicToc t_ds;
    
    //clear the corner
    laser_corner_ds_->clear();
    
    //filter the laser_corner_ds_
    ds_corner_.setInputCloud(laser_corner_);
    ds_corner_.filter(*laser_corner_ds_);
    
    //clear the surface
    laser_surf_ds_->clear();
    
    //filter the laser_surf_ds_
    ds_surf_.setInputCloud(laser_surf_);
    ds_surf_.filter(*laser_surf_ds_);
    
    //clear the outlier
    laser_outlier_ds_->clear();
    
    //filter the laser_outlier_ds_
    ds_outlier_.setInputCloud(laser_outlier_);
    ds_outlier_.filter(*laser_outlier_ds_);
    
    //clear the laser_surf_total_
    laser_surf_total_->clear();
    laser_surf_total_ds_->clear();
    
    //added to the laser_surf_total_
    *laser_surf_total_ += *laser_surf_ds_;
    *laser_surf_total_ += *laser_outlier_ds_;
    
    //filter the laser_surf_total_ds_
    ds_surf_.setInputCloud(laser_surf_total_);
    ds_surf_.filter(*laser_surf_total_ds_);
    
    
    /*
    ROS_INFO("downsampleCurrentScan: %.3fms", t_ds.toc());
    ROS_INFO("before downsample: corner size %d, surf size %d", laser_corner_->points.size(), laser_surf_total_->points.size() + laser_outlier_->points.size());
    ROS_INFO("after downsample: corner size %d, surf size %d", laser_corner_ds_->points.size(), laser_surf_total_ds_->points.size());
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>(5,1));
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>(5,1));

    if (pcl::io::loadPCDFile<pcl::PointXYZ> ("/home/deep/catkin_ws/src/work_in_job/A-LeGO-LOAM/000010.pcd", *cloud_in) == -1) //* load the file
    {
      PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
      //return (-1);
    }
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud (new pcl::PointCloud<pcl::PointXYZ>(5,1));
    pcl::Indices indices;
    pcl::removeNaNFromPointCloud(*cloud_in, *output_cloud, indices);
    
    cloud_in = output_cloud;
    
    /*
    std::cout << "Loaded "
              << cloud_in->width * cloud_in->height
              << " data points from test_pcd.pcd with the following fields: "
              << std::endl;
    for (const auto& point: *cloud_in)
      std::cout << "    " << point.x
                << " "    << point.y
                << " "    << point.z << std::endl;
    //*/
    
    /*
    if (pcl::io::loadPCDFile<pcl::PointXYZ> ("/home/deep/catkin_ws/src/work_in_job/A-LeGO-LOAM/000050.pcd", *cloud_out) == -1) //* load the file
    {
      PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
      //return (-1);
    }
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud_2 (new pcl::PointCloud<pcl::PointXYZ>(5,1));
    pcl::removeNaNFromPointCloud(*cloud_out, *output_cloud_2, indices);

    cloud_out = output_cloud_2;
    //*/
    
    // Fill in the CloudIn data
    //for (auto& point : *cloud_in)
    //{
    //    point.x = 1024 * rand() / (RAND_MAX + 1.0f);
    //    point.y = 1024 * rand() / (RAND_MAX + 1.0f);
    //    point.z = 1024 * rand() / (RAND_MAX + 1.0f);
    //}

    //std::cout << "Saved " << cloud_in->size () << " data points to input:" << std::endl;

    //for (auto& point : *cloud_in)
    //    std::cout << point << std::endl;

    //*cloud_out = *cloud_in;

    //std::cout << "size:" << cloud_out->size() << std::endl;
    //for (auto& point : *cloud_out)
    //    point.x += 0.7f;

    //std::cout << "Transformed " << cloud_in->size () << " data points:" << std::endl;

    //for (auto& point : *cloud_out)
    //    std::cout << point << std::endl;

    //*/

/*
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(cloud_out);
    icp.setInputTarget(cloud_in);
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);
    std::cout << "has converged:" << icp.hasConverged() << " score: " <<
    icp.getFitnessScore() << std::endl;
    std::cout << icp.getFinalTransformation() << std::endl;
    

    PointCloudT::Ptr pc_map(new PointCloudT);
    sensor_msgs::PointCloud2Ptr msg(new sensor_msgs::PointCloud2);


    cout<<"==================cloud_in====================="<<endl;
    pcl::toROSMsg(*cloud_in, *msg);
    msg->header.stamp.fromSec(0);
    msg->header.frame_id = "map";
    pub_cloud_surround_.publish(msg);
    
    cout<<"==================cloud_out====================="<<endl;
    pcl::toROSMsg(*cloud_out, *msg);
    msg->header.stamp.fromSec(0);
    msg->header.frame_id = "map";
    pub_recent_keyframes_.publish(msg);    
    
    //cout<<"==================Final====================="<<endl;
    //pcl::toROSMsg(Final, *msg);
    //msg->header.stamp.fromSec(0);
    //msg->header.frame_id = "map";
    //pub_history_keyframes_.publish(msg);   
    
    
    
    Ti = icp.getFinalTransformation () * Ti;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr output (new pcl::PointCloud<pcl::PointXYZ>(5,1));
    //pcl::transformPointCloud (*cloud_out, *output, icp.getFinalTransformation());
    pcl::transformPointCloud (*cloud_out, *output, Ti);    
    
    cout<<"==================FinalTransformation====================="<<endl;
    pcl::toROSMsg(*output, *msg);
    msg->header.stamp.fromSec(0);
    msg->header.frame_id = "map";
    pub_history_keyframes_.publish(msg);   
    
    
        
//*/

    //odometry to laser
    //t_odom2laser_(0) = msg->pose.pose.position.x;
    //t_odom2laser_(1) = msg->pose.pose.position.y;
    //t_odom2laser_(2) = msg->pose.pose.position.z;
    //q_odom2laser_.w() = msg->pose.pose.orientation.w;
    //q_odom2laser_.x() = msg->pose.pose.orientation.x;
    //q_odom2laser_.y() = msg->pose.pose.orientation.y;
    //q_odom2laser_.z() = msg->pose.pose.orientation.z;
    
    
    //t_map2odom_(0) = 0;
    //t_map2odom_(1) = 0;
    //t_map2odom_(2) = 0;
            
    //q_map2odom_.x() = 0;
    //q_map2odom_.y() = 0;
    //q_map2odom_.z() = 0;
    //q_map2odom_.w() = 1;
    
    //map to laser
    //t_map2laser_ = q_map2odom_ * t_odom2laser_ + t_map2odom_;
    //q_map2laser_ = q_map2odom_ * q_odom2laser_;




  }

  void transformUpdate()
  {
  

    shared_memory_object shdmem(open_or_create, "Boost", read_write);
    shdmem.truncate(1024);
    mapped_region region(shdmem, read_write);
    //std::cout << std::hex << region.get_address() << std::endl;
    //std::cout << std::dec << region.get_size() << std::endl;
    double* i1 = static_cast<double*>(region.get_address());


    //ROS_WARN("========================================transform[0]=============================================");
    //ROS_WARN("in map [x] : %f", *i1);
    //ROS_WARN("in map [y] : %f", *(i1+1));
    //ROS_WARN("in map [z] : %f", *(i1+2));
  
    // q_0=-0.00666253, q_1=-0.0385051, q_2=0.0329403, q_3=0.998693
    //Quaternion (const tfScalar &x, const tfScalar &y, const tfScalar &z, const tfScalar &w)
    //tf::Quaternion q(*(i1+3), *(i1+4), *(i1+5), *(i1+6));
    tf::Quaternion q( -0.00666253 , -0.0385051, 0.0329403, 0.998693);
    tf::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    params_[0] = (*i1)*((-45));
    params_[1] = (*(i1+1))*80;    
    params_[2] = (*(i1+2))* (-0.45);
    params_[3] = 0;    
    params_[4] = 0;    
    params_[5] = yaw;
    
    //ROS_WARN(" yaw [0] : %f", yaw);
    
    
    PointT this_pose_3d;

    this_pose_3d.x = params_[0];
    this_pose_3d.y = params_[1];
    this_pose_3d.z = params_[2];
    this_pose_3d.intensity = cloud_keyposes_3d_->points.size() + 0.1;
    
    cloud_keyposes_3d_->points.push_back(this_pose_3d);
          
    q_map2laser_ = Eigen::AngleAxisd( params_[5], Eigen::Vector3d::UnitZ() ) * Eigen::AngleAxisd( params_[4], Eigen::Vector3d::UnitY() ) * Eigen::AngleAxisd( params_[3], Eigen::Vector3d::UnitX() );
    t_map2laser_.x() = params_[0];
    t_map2laser_.y() = params_[1];
    t_map2laser_.z() = params_[2];
    
    // get the odometry to laser to the map to odometry, by map to laser of R and T
    q_map2odom_ = q_map2laser_ * q_odom2laser_.inverse();
    t_map2odom_ = t_map2laser_ - q_map2odom_ * t_odom2laser_;
    
    
    
    //cout<<"=============================Odometry================================"<<endl;
    //cout<< Ti_real <<endl;
    
    nav_msgs::OdometryPtr msg_2(new nav_msgs::Odometry);
    msg_2->header.stamp.fromSec(time_laser_odom_);
    msg_2->header.frame_id = "map";
    msg_2->child_frame_id = "/laser";
    msg_2->pose.pose.position.x = Ti_real(0,3);
    msg_2->pose.pose.position.y = Ti_real(1,3);
    msg_2->pose.pose.position.z = Ti_real(2,3);
    msg_2->pose.pose.orientation.w = 1;
    msg_2->pose.pose.orientation.x = 0;
    msg_2->pose.pose.orientation.y = 0;
    msg_2->pose.pose.orientation.z = 0;
    pub_odom_aft_mapped_2.publish(msg_2);
    
    //cout<<"=============================Odometry_2================================"<<endl;

    nav_msgs::OdometryPtr msg_3(new nav_msgs::Odometry);
    msg_3->header.stamp.fromSec(time_laser_odom_);
    msg_3->header.frame_id = "map";
    msg_3->child_frame_id = "/laser";
    msg_3->pose.pose.position.x = Ti(0,3);
    msg_3->pose.pose.position.y = Ti(1,3);
    msg_3->pose.pose.position.z = Ti(2,3);
    msg_3->pose.pose.orientation.w = 1;
    msg_3->pose.pose.orientation.x = 0;
    msg_3->pose.pose.orientation.y = 0;
    msg_3->pose.pose.orientation.z = 0;
    pub_odom_aft_mapped_3.publish(msg_3);
    
    
    kf::Matrix<1, 1>  matrix_F{ kf::Matrix<1, 1>::Identity() };  // 1x1
    matrix_F << 1.0F;
    
    kf::Matrix<1, 1>  matrix_Q{ kf::Matrix<1, 1>::Identity() };  // 1x1
    matrix_Q << 0.05F;
    
    kf::Matrix<2, 2>  matrix_R { kf::Matrix<2, 2>::Identity() }; // 2x2
    matrix_R << 2, 0, 0, 1;
    
    kf::Matrix<2, 1>  matrix_H { kf::Matrix<2, 1>::Random(2,1) };// 2x1
    matrix_H << 1.0F, 1.0F;
    
    
    //std::cout<<"======================kalmanfilter============================="<<std::endl;
    kalmanfilter_x.vector_x() << 1.0F;
    kalmanfilter_x.matrix_P() << 1.0F;
    const kf::Vector<2> vector_z_of_x { Ti_real(0,3), this_pose_3d.x };
    kalmanfilter_x.predict(matrix_F, matrix_Q );
    x = kalmanfilter_x.correct(vector_z_of_x, matrix_R, matrix_H);


    kalmanfilter_y.vector_x() << 1.0F;
    kalmanfilter_y.matrix_P() << 1.0F;
    const kf::Vector<2> vector_z_of_y { Ti_real(1,3), this_pose_3d.y };
    kalmanfilter_y.predict(matrix_F, matrix_Q );
    y = kalmanfilter_y.correct(vector_z_of_y, matrix_R, matrix_H);
    
    //const kf::Vector<2> vector_z_of_z { Ti_real(2,3), this_pose_3d.z };
    //kalmanfilter_z.predict(matrix_F, matrix_Q );
    //double z = kalmanfilter_z.correct(vector_z_of_z, matrix_R, matrix_H);

    z = this_pose_3d.z;
    nav_msgs::OdometryPtr msg_kalman(new nav_msgs::Odometry);
    msg_kalman->header.stamp.fromSec(time_laser_odom_);
    msg_kalman->header.frame_id = "map";
    msg_kalman->child_frame_id = "/laser";
    
    msg_kalman->pose.pose.position.x = Ti_of_map_real(0,3);
    msg_kalman->pose.pose.position.y = Ti_of_map_real(1,3);
    msg_kalman->pose.pose.position.z = Ti_of_map_real(2,3);
    msg_kalman->pose.pose.orientation.w = 1;
    msg_kalman->pose.pose.orientation.x = 0;
    msg_kalman->pose.pose.orientation.y = 0;
    msg_kalman->pose.pose.orientation.z = 0;
    pub_odom_aft_mapped_kalman.publish(msg_kalman);
    
    
    
    
  }

  //publish the message
  void publish()
  {
    if (pub_keyposes_.getNumSubscribers() > 0)
    {
      sensor_msgs::PointCloud2Ptr msg(new sensor_msgs::PointCloud2);
      pcl::toROSMsg(*cloud_keyposes_3d_, *msg);
      msg->header.stamp.fromSec(time_laser_odom_);
      msg->header.frame_id = "map";
      pub_keyposes_.publish(msg);
    }
  }

  //main loop thread
  void mainLoop()
  {
    ros::Duration dura(0.01);
    while (ros::ok())
    {
      
      downsampleCurrentScan();
      transformUpdate();
      publish();
      
      dura.sleep();
      ros::spinOnce();
    }
  }
  
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "LM");
  ros::NodeHandle nh;
  LM lm(nh);
  lm.mainLoop();
  ros::spin();
  return 0;
}
