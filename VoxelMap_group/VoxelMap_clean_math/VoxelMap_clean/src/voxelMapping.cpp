#include "IMU_Processing.hpp"
#include "preprocess.h"
#include "voxel_map_util.hpp"
#include <Eigen/Core>
#include <common_lib.h>
#include <csignal>
#include <fstream>
#include <geometry_msgs/Vector3.h>
#include <math.h>
#include <mutex>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <so3_math.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf2_msgs/TFMessage.h>
#include <thread>
#include <unistd.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <voxel_map/States.h>

#define INIT_TIME (0.0)
int scanIdx = 0;

vect3 position_lidar;

//nh.param
string lidar_topic, imu_topic;
double ranging_cov = 0.0;
double angle_cov = 0.0;
double gyroscope_cov_scale, accelerater_cov_scale;
// params for imu
bool imu_en = true;
std::vector<double> extrinT;
std::vector<double> extrinR;
int  NUM_MAX_ITERATIONS;
int max_points_size = 50;
int max_cov_points_size = 50;
std::vector<double> layer_point_size;
int max_layer = 0;
double max_voxel_size = 1.0;
double filter_size_surface_min;
double min_eigen_value = 0.003;
shared_ptr<Preprocess> p_preprocess(new Preprocess());
bool calib_laser = false;
bool publish_voxel_map = false;
int publish_max_voxel_layer = 0;
bool publish_point_cloud = false;
int pub_point_cloud_skip = 1;
bool dense_map_en = true;

// surf feature in map
PointCloudXYZI::Ptr feature_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feature_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr laser_cloud_origin(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laser_cloud_no_effect(new PointCloudXYZI(100000, 1));
pcl::VoxelGrid<PointType> down_size_filter_surface;

bool flg_reset, flg_exit = false;
condition_variable sig_buffer;
void SigHandle(int sig) {
  flg_exit = true;
  ROS_WARN("catch sig %d", sig);
  sig_buffer.notify_all();
}

std::vector<int> layer_size;

double lidar_time_offset = 0.0;
mutex mtx_buffer;
double last_timestamp_lidar, last_timestamp_imu = -1.0;

deque<PointCloudXYZI::Ptr> lidar_buffer;
deque<double> time_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

int publish_count = 0;

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    auto time_offset = lidar_time_offset;
    //std::printf("lidar offset:%f\n", lidar_time_offset);
    //ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() + time_offset < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_preprocess->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    //ROS_INFO("p_preprocess->process[0] %.6f", ptr->points.back().curvature);
    time_buffer.push_back(msg->header.stamp.toSec() + time_offset);
    last_timestamp_lidar = msg->header.stamp.toSec() + time_offset;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}


void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) {
  publish_count++;
  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

  double timestamp = msg->header.stamp.toSec();

  mtx_buffer.lock();

  if (timestamp < last_timestamp_imu) {
    ROS_ERROR("imu loop back, clear buffer");
    imu_buffer.clear();
    flg_reset = true;
  }

  last_timestamp_imu = timestamp;

  imu_buffer.push_back(msg);
  cout << "got imu: " << timestamp << " imu size " << imu_buffer.size() << endl;
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}


nav_msgs::Path path;

geometry_msgs::Quaternion geometry_quaternion;

// estimator inputs and output;
MeasureGroup Measures;


double lidar_mean_scantime = 0.0;
int    scan_num = 0;

bool lidar_pushed = false;
double lidar_end_time = 0;
bool flg_first_scan = true;
double first_lidar_time = 0;


double total_distance = 0;


bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }
    //cout<<"=========[0]==========="<<endl;
    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
            //ROS_INFO("lidar_end_time[0] %.6f", lidar_end_time);
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            //ROS_INFO("lidar_end_time[1] %.6f", lidar_end_time);
        }
        else
        {
            //ROS_INFO("lidar_end_time[2.0] %.6f", meas.lidar_beg_time);
            //ROS_INFO("lidar_end_time[2.1] %.6f", meas.lidar->points.size());
            //ROS_INFO("lidar_end_time[2.1] %.6f", meas.lidar->points.back().curvature);
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
            //ROS_INFO("lidar_end_time[2] %.6f", lidar_end_time);

        }

        meas.lidar_end_time = lidar_end_time;
        //ROS_INFO("lidar_end_time[3] %.6f", lidar_end_time);

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }
    //cout<<"=========[1]==========="<<endl;
    //cout<<imu_buffer.size()<<endl;
    //cout<<"=========[1.0]==========="<<endl;
    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    //ROS_INFO("imu_time %.6f", imu_time);
    //ROS_INFO("lidar_end_time %.6f", lidar_end_time);
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();

        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
	//ROS_INFO("imu_time %.6f", imu_time);
	//ROS_INFO("lidar_end_time %.6f", lidar_end_time);
        cout<<imu_buffer.size()<<endl;
    }
    //cout<<"=========[2]==========="<<endl;
    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}



V3D euler_current;
V3D position_last(Zero3d);

// estimator inputs and output;
//MeasureGroup Measures;

//struct MeasureGroup // Lidar data and imu dates for the curent process
//{
//  MeasureGroup() { this->lidar.reset(new PointCloudXYZI()); };
//  double lidar_beg_time;
//  PointCloudXYZI::Ptr lidar;
//  deque<sensor_msgs::Imu::ConstPtr> imu;
//};


StatesGroup state;


// 7 state

//StatesGroup() {
//  this->rot_end = M3D::Identity();
//  this->pos_end = Zero3d;
//  this->vel_end = Zero3d;
//  this->bias_g = Zero3d;
//  this->bias_a = Zero3d;
//  this->gravity = Zero3d;
//  this->cov = Matrix<double, DIM_STATE, DIM_STATE>::Identity() * INIT_COV;
//};





// project the lidar scan to world frame
void point_body_to_world(PointType const *const pi, PointType *const po) {
  V3D p_body(pi->x, pi->y, pi->z);
  p_body = p_body + Lidar_offset_to_IMU;
  V3D p_global(state.rot_end * (p_body) + state.pos_end);
  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

const bool var_contrast(pointWithCov &x, pointWithCov &y) {
  return (x.cov.diagonal().norm() < y.cov.diagonal().norm());
};

geometry_msgs::PoseStamped msg_body_pose;

template <typename T> void set_posestamp(T &out) {
  out.position.x = state.pos_end(0);
  out.position.y = state.pos_end(1);
  out.position.z = state.pos_end(2);
  out.orientation.x = geometry_quaternion.x;
  out.orientation.y = geometry_quaternion.y;
  out.orientation.z = geometry_quaternion.z;
  out.orientation.w = geometry_quaternion.w;
}

void publish_path(const ros::Publisher pubPath) {
  set_posestamp(msg_body_pose.pose);
  msg_body_pose.header.stamp = ros::Time::now();
  msg_body_pose.header.frame_id = "camera_init";
  path.poses.push_back(msg_body_pose);
  pubPath.publish(path);
}

void RGB_point_body_to_world(PointType const *const pi, PointType *const po) {

  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(state.rot_end * (p_body) + state.pos_end);
  
  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  
  po->intensity = pi->intensity;
  po->curvature = pi->curvature;
  
  po->normal_x = pi->normal_x;
  po->normal_y = pi->normal_y;
  po->normal_z = pi->normal_z;
  
  float intensity = pi->intensity;
  
  intensity = intensity - floor(intensity);

  int reflection_map = intensity * 10000;
}

void publish_frame_world(const ros::Publisher &pubLaserCloudFullRes, const int point_skip) {

  PointCloudXYZI::Ptr laserCloudFullRes(dense_map_en ? feature_undistort : feature_down_body);
  int size = laserCloudFullRes->points.size();
  
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
  
  for (int i = 0; i < size; i++) {
    RGB_point_body_to_world(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
  }
  
  PointCloudXYZI::Ptr laserCloudWorldPub(new PointCloudXYZI);
  
  for (int i = 0; i < size; i += point_skip) {
    laserCloudWorldPub->points.push_back(laserCloudWorld->points[i]);
  }
  
  sensor_msgs::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*laserCloudWorldPub, laserCloudmsg);
  laserCloudmsg.header.stamp =
      ros::Time::now(); //.fromSec(last_timestamp_lidar);
  laserCloudmsg.header.frame_id = "camera_init";
  pubLaserCloudFullRes.publish(laserCloudmsg);
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "voxelMapping");
  ros::NodeHandle nh;


  //-----------------------------------------iniitlize the variable[start]-----------------------------------------------//



  // cummon params
  nh.param<string>("common/lid_topic", lidar_topic, "/velodyne_points");
  nh.param<string>("common/imu_topic", imu_topic, "/imu/data");  
  
  // noise model params
  nh.param<double>("noise_model/ranging_cov", ranging_cov, 0.02);
  nh.param<double>("noise_model/angle_cov", angle_cov, 0.05);
  nh.param<double>("noise_model/gyr_cov_scale", gyroscope_cov_scale, 0.1);
  nh.param<double>("noise_model/acc_cov_scale", accelerater_cov_scale, 0.1);

  // imu params, current version does not support imu
  nh.param<bool>("imu/imu_en", imu_en, false);
  nh.param<vector<double>>("imu/extrinsic_T", extrinT, vector<double>());
  nh.param<vector<double>>("imu/extrinsic_R", extrinR, vector<double>());

  // mapping algorithm params
  nh.param<int>("mapping/max_iteration", NUM_MAX_ITERATIONS, 4);
  nh.param<int>("mapping/max_points_size", max_points_size, 100);
  nh.param<int>("mapping/max_cov_points_size", max_cov_points_size, 100);
  nh.param<vector<double>>("mapping/layer_point_size", layer_point_size, vector<double>());
  nh.param<int>("mapping/max_layer", max_layer, 2);
  nh.param<double>("mapping/voxel_size", max_voxel_size, 1.0);
  nh.param<double>("mapping/down_sample_size", filter_size_surface_min, 0.2);
  nh.param<double>("mapping/plannar_threshold", min_eigen_value, 0.01);

  // preprocess params
  nh.param<double>("preprocess/blind", p_preprocess->blind, 0.01);
  nh.param<bool>("preprocess/calib_laser", calib_laser, false);
  nh.param<int>("preprocess/lidar_type", p_preprocess->lidar_type, AVIA);
  nh.param<int>("preprocess/scan_line", p_preprocess->N_SCANS, 16);
  nh.param<int>("preprocess/point_filter_num", p_preprocess->point_filter_num, 1);

  // visualization params
  nh.param<bool>("visualization/pub_voxel_map", publish_voxel_map, false);
  nh.param<int>("visualization/publish_max_voxel_layer", publish_max_voxel_layer, 0);
  nh.param<bool>("visualization/pub_point_cloud", publish_point_cloud, true);
  nh.param<int>("visualization/pub_point_cloud_skip", pub_point_cloud_skip, 1);
  nh.param<bool>("visualization/dense_map_enable", dense_map_en, false);

  cout << "p_preprocess->lidar_type " << p_preprocess->lidar_type << endl;
  
  
  for (int i = 0; i < layer_point_size.size(); i++) {
    layer_size.push_back(layer_point_size[i]);
  }

  ros::Subscriber sub_pcl = nh.subscribe(lidar_topic, 200000, standard_pcl_cbk);
  ros::Subscriber sub_imu;
  
  if (imu_en) {
    sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
  }

  ros::Publisher pubLaserCloudFullRes =  nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
  ros::Publisher pubLaserCloudEffect =   nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100);
  ros::Publisher pubOdomAftMapped =      nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
  ros::Publisher pubPath = 		  nh.advertise<nav_msgs::Path>("/path", 10);
  ros::Publisher voxel_map_pub =         nh.advertise<visualization_msgs::MarkerArray>("/planes", 10000);

  path.header.stamp = ros::Time::now();
  path.header.frame_id = "camera_init";

  /*** variables definition ***/
  
  // #define DIM_STATE (18)  // Dimension of states (Let Dim(SO(3)) = 3)
  VD(DIM_STATE) solution; // DIM_STATE = 18
  MD(DIM_STATE, DIM_STATE) H_Transpose_multiply_R_inverse_multiply_H, I_STATE, G;
  
  V3D R_add, T_add;
  
  StatesGroup state_propagate;
  
  PointCloudXYZI::Ptr corrospend_norm_vector(new PointCloudXYZI(100000, 1));
  
  double deltaT, deltaR = 0;
  down_size_filter_surface.setLeafSize( filter_size_surface_min, filter_size_surface_min, filter_size_surface_min );
  
  shared_ptr<ImuProcess> p_imu(new ImuProcess());
  p_imu->imu_en = imu_en;
  
  Eigen::Vector3d extra_T_parameter;
  Eigen::Matrix3d extra_R_parameter;
  
  extra_T_parameter << extrinT[0], extrinT[1], extrinT[2];
  extra_R_parameter << extrinR[0], extrinR[1], extrinR[2], extrinR[3], extrinR[4], extrinR[5], extrinR[6], extrinR[7], extrinR[8];
  
  p_imu->set_extrinsic(extra_T_parameter, extra_R_parameter);

  // Current version do not support imu.
  if (imu_en) {
    std::cout << "use imu" << std::endl;
  } else {
    std::cout << "no imu" << std::endl;
  }

  p_imu->set_gyr_cov_scale(V3D(gyroscope_cov_scale, gyroscope_cov_scale, gyroscope_cov_scale));
  p_imu->set_acc_cov_scale(V3D(accelerater_cov_scale, accelerater_cov_scale, accelerater_cov_scale));
  p_imu->set_gyr_bias_cov(V3D(0.00001, 0.00001, 0.00001));
  p_imu->set_acc_bias_cov(V3D(0.00001, 0.00001, 0.00001));


  H_Transpose_multiply_R_inverse_multiply_H.setZero();
  I_STATE.setIdentity();
  G.setZero();

  signal(SIGINT, SigHandle);
  ros::Rate rate(5000);
  bool status = ros::ok();

  // initialize the VoxelMap
  // for Plane Map
  bool init_map = false;
  std::unordered_map<VOXEL_LOC, OctoTree *> voxel_map;

  bool flag_EKF_initlized, flag_EKF_converged, EKF_stop_flag = 0, is_first_frame = true;

  //-----------------------------------------iniitlize the variable[end]-----------------------------------------------//



  while (status) {
  
    if (flg_exit)
      break;
      
    ros::spinOnce();
    
    // 5 kinds of covarience
    
    // 1. lidar covarience of point 
    // 2. world covarience of point
    // 3. point to plane covarience, [n,q], n is normal, q is center
    // 4. distance covarience, is 3 + 2
    // 5. z = h(x) covarience, is 3 + 1      
      
    if (sync_packages(Measures)) {
      // std::cout << "sync once" << std::endl;
      
      // the first scan
      if (flg_first_scan)
      {
            first_lidar_time = Measures.lidar_beg_time;
            p_imu->first_lidar_time = first_lidar_time;
            flg_first_scan = false;
            continue;
      }
      

      // [ IMU DONE ]
      //imu process to feature undistort, and move the pcl cloud to world coordinate
      p_imu->Process(Measures, state, feature_undistort);  
          
      state_propagate = state;

      // something wrong
      if (feature_undistort->empty() || (feature_undistort == NULL))
      {
        ROS_WARN("No point, skip this scan!\n");
        continue;
      }
      
      // =========== [step 1, get the input lidar point with variance in lidar frame, initilize voxel map with covariance ] =========== //
      
      // if the lidar is working, then we initialize the EKF
      flag_EKF_initlized = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;
            
      cout<<"===========[0]============="<<endl;

      if (flag_EKF_initlized && !init_map) {
      
        cout<<"===========[1]============="<<endl;
        pcl::PointCloud<pcl::PointXYZI>::Ptr world_lidar( new pcl::PointCloud<pcl::PointXYZI>);
        Eigen::Quaterniond q(state.rot_end);
        
        // feature_undistort to world_lidar
        transform_lidar_to_world(state, p_imu, feature_undistort, world_lidar);
        
        std::vector<pointWithCov> pv_list;
        
        for (size_t i = 0; i < world_lidar->size(); i++) {
        
          pointWithCov pv;
          
          pv.point << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
          
          V3D point_this( feature_undistort->points[i].x, feature_undistort->points[i].y, feature_undistort->points[i].z);
          
          // if z=0, error will occur in calcBodyCov. To be solved
          if (point_this[2] == 0) {
            point_this[2] = 0.001;
          }
          
	  //cout<<"========calcBodyCov=========="<<endl;
	  //cout<<point_this.size()<<endl; // = 3          
          
	  // 1. lidar covarience of point 
	  // 2. world covarience of point
	  // 3. point to plane covarience, [n,q], n is normal, q is center
	  // 4. distance covarience, is 3 + 2
	  // 5. z = h(x) covarience, is 3 + 1


          // 1. lidar covarience of point 
          // cov = cov_lidar
          M3D cov_lidar;  
          calcBodyCov(point_this, ranging_cov, angle_cov, cov_lidar);


          point_this += Lidar_offset_to_IMU;
          
          M3D point_crossmat;
          
          point_crossmat << SKEW_SYM_MATRX(point_this);
          
          // 2. world covarience of point
          M3D cov_world = state.rot_end * cov_lidar * state.rot_end.transpose() +
                (-point_crossmat) * state.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose() +
                state.cov.block<3, 3>(3, 3);
          
          // cov = cov_world      
          pv.cov = cov_world;
          
          pv_list.push_back(pv);
          
          Eigen::Vector3d sigma_pv = pv.cov.diagonal();
          sigma_pv[0] = sqrt(sigma_pv[0]);
          sigma_pv[1] = sqrt(sigma_pv[1]);
          sigma_pv[2] = sqrt(sigma_pv[2]);
        }

	cout<<"========pv_list.size()=========="<<endl;
	cout<<pv_list.size()<<endl;      //16026


        // build the initilized map by pv_list
        buildVoxelMap(pv_list, max_voxel_size, max_layer, layer_size, max_points_size, max_points_size, min_eigen_value, voxel_map);
                      
        std::cout << "build voxel map" << std::endl;
        scanIdx++;
        
        if (1) {
          pubVoxelMap(voxel_map, publish_max_voxel_layer, voxel_map_pub);
        }
        
        init_map = true;
        continue;
        
      }      
      
      
      // down sample the point cloud
      down_size_filter_surface.setInputCloud(feature_undistort);
      down_size_filter_surface.filter(*feature_down_body);
      
      std::cout << " feats size:" << feature_undistort->size() << ", down size:" << feature_down_body->size() << std::endl;

      // sort the point cloud by the time
      sort(feature_down_body->points.begin(), feature_down_body->points.end(), time_list);
      
      int rematch_number = 0;
      bool nearest_search_en = true;

      std::vector<M3D> body_variance;
      std::vector<M3D> crossmat_list;      
      
      
       
      // =========== [step 2, run the esekf kalman filter to get the state and covariance ] =========== //

      /*** iterated state estimation ***/
      auto calculate_point_cov_start = std::chrono::high_resolution_clock::now();
      
      // ------CORE[ 2.1 get the lidar point covriance ]------ //
      // get the body variance
      for (size_t i = 0; i < feature_down_body->size(); i++) 
      {
      
        V3D point_this(feature_down_body->points[i].x, feature_down_body->points[i].y, feature_down_body->points[i].z);
        
        if (point_this[2] == 0) {
          point_this[2] = 0.001;
        }
        
	// 1. lidar covarience of point 
	// 2. world covarience of point
	// 3. point to plane covarience, [n,q], n is normal, q is center
	// 4. distance covarience, is 3 + 2
	// 5. z = h(x) covarience, is 3 + 1
        
        // 1. lidar covarience of point 
        M3D cov_lidar; // cov = cov_lidar
        
        calcBodyCov(point_this, ranging_cov, angle_cov, cov_lidar);
        
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_this);
        
        crossmat_list.push_back(point_crossmat);
        body_variance.push_back(cov_lidar);
        
      }
 
 
      auto calc_point_cov_end = std::chrono::high_resolution_clock::now();
      
      double calc_point_cov_time = std::chrono::duration_cast<std::chrono::duration<double>>( calc_point_cov_end - calculate_point_cov_start).count() * 1000;
      
      // =========== [step 2.1, run the iterated esekf kalman filter is started ] =========== //
      
      // NUM_MAX_ITERATIONS = 10, the IEKF is started
      for (int iterCount = 0; iterCount < NUM_MAX_ITERATIONS; iterCount++) 
      {  
        laser_cloud_origin->clear();
        laser_cloud_no_effect->clear();
        corrospend_norm_vector->clear();
        

        std::vector<ptpl> point_to_plane_list;
        /** LiDAR match based on 3 sigma criterion **/

        vector<pointWithCov> pv_list;
        std::vector<M3D> var_list;
        
        pcl::PointCloud<pcl::PointXYZI>::Ptr world_lidar( new pcl::PointCloud<pcl::PointXYZI> );
        transform_lidar_to_world(state, p_imu, feature_down_body, world_lidar);
      
              
        // ------CORE[ 2.2 get the world point covriance ]------ //
        for (size_t i = 0; i < feature_down_body->size(); i++) {
        
          pointWithCov pv;
          
          pv.point << feature_down_body->points[i].x, feature_down_body->points[i].y, feature_down_body->points[i].z;
          pv.point_world << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
          

	  // 1. lidar covarience of point 
	  // 2. world covarience of point
	  // 3. point to plane covarience, [n,q], n is normal, q is center
	  // 4. distance covarience, is 3 + 2
	  // 5. z = h(x) covarience, is 3 + 1
          
          // 1. lidar covarience of point 
          M3D cov_lidar = body_variance[i];
          
          M3D point_crossmat = crossmat_list[i];
          M3D R_var = state.cov.block<3, 3>(0, 0);
          M3D T_var = state.cov.block<3, 3>(3, 3);
          // transformLiDARCovarianceToWorldCovariance()
          
          // 2. world covarience of point
          // cov = cov_world   the uncertainty of the LiDAR point of world coordinate
          M3D cov_world = state.rot_end * cov_lidar * state.rot_end.transpose() + (-point_crossmat) * R_var * (-point_crossmat.transpose()) + T_var;
                
          pv.cov = cov_world;
          pv_list.push_back(pv);
          var_list.push_back(cov_world);
          
        }
        
        std::vector<V3D> non_match_list;
        // build the Voxel map
        BuildResidualListOMP(voxel_map, max_voxel_size, 3.0, max_layer, pv_list, point_to_plane_list, non_match_list);
        int effect_feature_number = 0;
        
        // get all the residual
        for (int i = 0; i < point_to_plane_list.size(); i++) {
        
          PointType pi_body;
          PointType pi_world;
          PointType plane;
          
          pi_body.x = point_to_plane_list[i].point(0);
          pi_body.y = point_to_plane_list[i].point(1);
          pi_body.z = point_to_plane_list[i].point(2);
          
          point_body_to_world(&pi_body, &pi_world);
          
          plane.x = point_to_plane_list[i].normal(0);
          plane.y = point_to_plane_list[i].normal(1);
          plane.z = point_to_plane_list[i].normal(2);
          
          effect_feature_number++;
          float distance = (pi_world.x * plane.x + pi_world.y * plane.y + pi_world.z * plane.z + point_to_plane_list[i].d);
          
          plane.intensity = distance;
          
          laser_cloud_origin->push_back(pi_body);
          corrospend_norm_vector->push_back(plane);
          
        }
  

        /*** Computation of Measuremnt Jacobian matrix H and measurents vector * ***/
         
        // Get the  H_ and  H.transpose() * R.inverse()
        // ------CORE[ 2.3  Measuremnt Jacobian matrix H ]------ //
        MatrixXd Hsub( effect_feature_number, 6);
        MatrixXd Hsub_Transpose_multiply_R_inverse( 6, effect_feature_number );
        
        VectorXd R_inverse( effect_feature_number );
        VectorXd measurement_vector( effect_feature_number );
        
        double max_distance = 0;      
        
        // ---------------------for every point--------------------- //
        

        for (int i = 0; i < effect_feature_number; i++) {
        
          const PointType &laser_point = laser_cloud_origin->points[i];
          V3D point_this(laser_point.x, laser_point.y, laser_point.z);
          
          
          // 1. lidar covarience of point 
	  // 2. world covarience of point
	  // 3. point to plane covarience, [n,q], n is normal, q is center
	  // 4. distance covarience, is 3 + 2
	  // 5. z = h(x) covarience, is 3 + 1
          
          // 1. lidar covarience of point 
          M3D cov_lidar;
          
          calcBodyCov(point_this, ranging_cov, angle_cov, cov_lidar);
    
          // 1. lidar covarience of point 
          cov_lidar = state.rot_end * cov_lidar * state.rot_end.transpose();
          
          M3D point_crossmat;
          point_crossmat << SKEW_SYM_MATRX(point_this);
          
          const PointType &norm_point = corrospend_norm_vector->points[i];
          V3D norm_vector(norm_point.x, norm_point.y, norm_point.z);
          
          V3D point_world = state.rot_end * point_this + state.pos_end;
          
          // /*** get the normal vector of closest surface/corner ***/
          
          // 3. point to plane covarience, [n,q], n is normal, q is center
          Eigen::Matrix<double, 1, 6> J_nq;
          
          J_nq.block<1, 3>(0, 0) = point_world - point_to_plane_list[i].center;
          J_nq.block<1, 3>(0, 3) = -point_to_plane_list[i].normal;
          
          // 4. distance covarience, is 3 + 2
          double sigma_l = J_nq * point_to_plane_list[i].plane_cov * J_nq.transpose();
          R_inverse(i) = 1.0 / (sigma_l + norm_vector.transpose() * cov_lidar * norm_vector);
          
          
          laser_cloud_origin->points[i].intensity = sqrt(R_inverse(i));
          laser_cloud_origin->points[i].normal_x =  corrospend_norm_vector->points[i].intensity;
          laser_cloud_origin->points[i].normal_y =  sqrt(sigma_l);
          laser_cloud_origin->points[i].normal_z =  sqrt(norm_vector.transpose() * cov_lidar * norm_vector);
          laser_cloud_origin->points[i].curvature = sqrt(sigma_l + norm_vector.transpose() * cov_lidar * norm_vector);

          /*** calculate the Measuremnt Jacobian matrix H ***/
          
          // A is [ d(d) / d(R), d(d) / d(t) ]
          // d = n^T(Rxp+t)
          // d(d) / d(R) = d(d) / d(p_world) * d(p_world) / d(R) 
          // d(p_world) / d(R) = p_crose x R 
          
          V3D A(point_crossmat * state.rot_end.transpose() * norm_vector);
          Hsub.row(i) << VEC_FROM_ARRAY(A), norm_point.x, norm_point.y, norm_point.z;
          
          Hsub_Transpose_multiply_R_inverse.col(i) << A[0] * R_inverse(i),         A[1] * R_inverse(i),         A[2] * R_inverse(i), 
                                                      norm_point.x * R_inverse(i), norm_point.y * R_inverse(i), norm_point.z * R_inverse(i);
          /*** Measuremnt: distance to the closest surface/corner ***/
          measurement_vector(i) = -norm_point.intensity;


          if ( max_distance < measurement_vector(i) )
          {
               max_distance = measurement_vector(i);
          }
          
          //float distance = (pi_world.x * pl.x + pi_world.y * pl.y + pi_world.z * pl.z + point_to_plane_list[i].d);
          //pl.intensity = distance;
          
        }
        
        MatrixXd K(DIM_STATE, effect_feature_number);
        EKF_stop_flag = false;
        flag_EKF_converged = false;  
   
   
      /*** Iterative Kalman Filter Update ***/
        
        // ------CORE[ 2.4 Kalman Filter Update by Measuremnt Jacobian matrix H ]------ //
        if (!flag_EKF_initlized) {
        
          cout << "===================Initiallizing LiDar===================" << endl;
          /*** only run in initialization period ***/
          
          MatrixXd H_init(MD(9, DIM_STATE)::Zero());
          MatrixXd z_init(VD(9)::Zero());
          
          H_init.block<3, 3>(0, 0) = M3D::Identity();
          H_init.block<3, 3>(3, 3) = M3D::Identity();
          H_init.block<3, 3>(6, 15) = M3D::Identity();
          
          z_init.block<3, 1>(0, 0) = -Log(state.rot_end);
          z_init.block<3, 1>(3, 0) = -state.pos_end;

	  // [DONE]
          auto H_init_Transpose = H_init.transpose();
          
          // [DONE]
          auto &&K_init =
              state.cov * H_init_Transpose * ( H_init * state.cov * H_init_Transpose + 0.0001 * MD(9, 9)::Identity() ).inverse();
          // [DONE]
          solution = K_init * z_init;

          state.resetpose();
          
          EKF_stop_flag = true;
          
        } else {
        
          // [DONE]
          auto &&Hsub_Transpose = Hsub.transpose();
          
          // [DONE]
          H_Transpose_multiply_R_inverse_multiply_H.block<6, 6>(0, 0) = Hsub_Transpose_multiply_R_inverse * Hsub;
          
          // [DONE]
          MD(DIM_STATE, DIM_STATE) &&Q_inverse = ( 1.0 * H_Transpose_multiply_R_inverse_multiply_H + 1.0 * (state.cov).inverse() + 0.00001 * Hsub_Transpose * Hsub ).inverse(); // DIM_STATE=18
          // [DONE]
          K = Q_inverse.block<DIM_STATE, 6>(0, 0) * Hsub_Transpose_multiply_R_inverse;
          // [DONE]
          auto vector_box_minus = state_propagate - state;
          
          //cout<<"=============Hsub============="<<endl;
          //cout<<Hsub<<endl;
          //MatrixXd Hsub( effect_feature_number, 6);
          
          // MatrixXd Hsub( effect_feature_number, 6);
          // MatrixXd Hsub_Transpose_multiply_R_inverse( 6, effect_feature_number );
          // VectorXd R_inverse( effect_feature_number );
          // VectorXd measurement_vector( effect_feature_number );
          // MatrixXd K(DIM_STATE, effect_feature_number);
          
          //solution = K * measurement_vector + vector_box_minus - K * Hsub * vector_box_minus.block<6, 1>(0, 0);
          solution = -1 * vector_box_minus + K * ( measurement_vector - Hsub * ( -1 ) * vector_box_minus.block<6, 1>(0, 0) );

          //float distance = (pi_world.x * pl.x + pi_world.y * pl.y + pi_world.z * pl.z + point_to_plane_list[i].d);
          //pl.intensity = distance;
          
	  // 7 items
	  //float64[] rot_end      # the estimated attitude (rotation matrix) at the end lidar point
	  //float64[] pos_end      # the estimated position at the end lidar point (world frame)
	  //float64[] vel_end      # the estimated velocity at the end lidar point (world frame)
	  //float64[] bias_gyr     # gyroscope bias
	  //float64[] bias_acc     # accelerator bias
	  //float64[] gravity      # the estimated gravity acceleration
	  //float64[] cov          # states covariance
	  // Pose6D[] IMUpose      # 6D pose at each imu measurements

          //cout<<"==========v[0]_before_eskf============"<<endl; 
          //cout<<state.vel_end<<endl;
          
          
          //V3D d_velocity(Zero3d);
          
          //d_velocity(0) = 1 * solution.block<3, 1>(6, 0)(0);
          //d_velocity(1) = 1 * solution.block<3, 1>(6, 0)(1);
          //d_velocity(2) = 1 * solution.block<3, 1>(6, 0)(2);
          
          //solution.block<3, 1>(6, 0) = d_velocity;
          
          state += solution;

          R_add = solution.block<3, 1>(0, 0);
          T_add = solution.block<3, 1>(3, 0);


          //cout<<"========================R,t======================"<<endl; 
          //cout<<state.rot_end<<","<<endl;     
          //cout<<state.pos_end<<","<<endl;
          //counter++;
          //cout<<"counter = ["<<counter<<"]"<<endl;
          cout<<"==========velocity[0]_after_eskf============"<<endl; 
          cout<<state.vel_end<<endl;
          cout<<"==========dv============"<<endl; 
          cout<<solution.block<3, 1>(6, 0)<<endl;
          cout<<"==========dT============"<<endl; 
          cout<<solution.block<3, 1>(3, 0)<<endl;
          //cout<<"=========measurement_vector========="<<endl;
          //cout<<measurement_vector(0)<<endl;
          //cout<<measurement_vector(1)<<endl;
          //cout<<measurement_vector(2)<<endl;
          //cout<<max_distance<<endl;
          
	  //if( abs(state.vel_end(0)) > 2 || abs(state.vel_end(1)) > 2 || abs(state.vel_end(2)) > 0.5 )
	  //{
	    //state_point.vel << state_point.vel[0]/5.0, state_point.vel[1]/5.0, state_point.vel[2]/5.0;
	    //state.vel_end << 0.0, 0.0, 0.0;
	    //state_point.rot.coeffs()[0] = 0.0;
	    //state_point.rot.coeffs()[1] = 0.0;
	    //state_point.rot.coeffs()[2] = 0.0;
	    //state_point.rot.coeffs()[3] = 1.0;
	  //}
          
          
          
          if ((R_add.norm() * 57.3 < 0.01) && (T_add.norm() * 100 < 0.015)) {
            flag_EKF_converged = true;
          }

          deltaR = R_add.norm() * 57.3;
          deltaT = T_add.norm() * 100;
          
        }           

        euler_current = RotMtoEuler(state.rot_end);
        
        /*** Rematch Judgement ***/
        
        nearest_search_en = false;
        
        if (flag_EKF_converged || ((rematch_number == 0) && (iterCount == (NUM_MAX_ITERATIONS - 2)))) {
          nearest_search_en = true;
          rematch_number++;
        }

        /*** Convergence Judgements and Covariance Update ***/
        
        if (!EKF_stop_flag && (rematch_number >= 2 || (iterCount == NUM_MAX_ITERATIONS - 1))) {
            
          if (flag_EKF_initlized) {
          
            /*** Covariance Update ***/
            G.setZero();
            G.block<DIM_STATE, 6>(0, 0) = K * Hsub;
            
            state.cov = (I_STATE - G) * state.cov;
            total_distance += (state.pos_end - position_last).norm();
            
            position_last = state.pos_end;

            geometry_quaternion = tf::createQuaternionMsgFromRollPitchYaw( euler_current(0), euler_current(1), euler_current(2) );

            VD(DIM_STATE) K_sum = K.rowwise().sum();
            VD(DIM_STATE) P_diag = state.cov.diagonal();
          }
          
          EKF_stop_flag = true;
          
        }

        if (EKF_stop_flag)
          break;        
      
      
      }
      
      position_lidar = state.pos_end;
      
      // ======================================================[step 3, update the voxel map of covariance ]========================================================= //
	    
      /*** add the  points to the voxel map ***/
      
      pcl::PointCloud<pcl::PointXYZI>::Ptr world_lidar( new pcl::PointCloud<pcl::PointXYZI> );
      transform_lidar_to_world(state, p_imu, feature_down_body, world_lidar);
      std::vector<pointWithCov> pv_list;
      
      for (size_t i = 0; i < world_lidar->size(); i++) {
      
        pointWithCov pv;
        pv.point << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
        
        M3D point_crossmat = crossmat_list[i];
        M3D cov_body = body_variance[i];
        
        cov_body = state.rot_end * cov_body * state.rot_end.transpose() + (-point_crossmat) * state.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose() + state.cov.block<3, 3>(3, 3);
        pv.cov = cov_body;
        pv_list.push_back(pv);
        
      }
      
      std::sort(pv_list.begin(), pv_list.end(), var_contrast);
      
      updateVoxelMapOMP(pv_list, max_voxel_size, max_layer, layer_size, max_points_size, max_points_size, min_eigen_value, voxel_map);      
      
      publish_path(pubPath);
      
      if (publish_voxel_map) {
        pubVoxelMap(voxel_map, publish_max_voxel_layer, voxel_map_pub);
      }
      
      if (true) {
      //if (publish_point_cloud) {
        publish_frame_world(pubLaserCloudFullRes, pub_point_cloud_skip);
      }
      
    }
      
  }


  return 0;
}
