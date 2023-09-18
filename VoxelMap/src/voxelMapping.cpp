#include "IMU_Processing.hpp"
#include "preprocess.h"
#include "voxel_map_util.hpp"
#include <Eigen/Core>
#include <common_lib.h>
#include <csignal>
#include <cv_bridge/cv_bridge.h>
#include <fstream>
#include <geometry_msgs/Vector3.h>
#include <image_transport/image_transport.h>
#include <livox_ros_driver/CustomMsg.h>
#include <math.h>
#include <mutex>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <opencv2/opencv.hpp>
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
#define CALIB_ANGLE_COV (0.01)

bool calib_laser = false;
bool write_kitti_log = false;
std::string result_path = "";

// params for imu
bool imu_en = true;
std::vector<double> extrinT;
std::vector<double> extrinR;

// params for publish function
bool publish_voxel_map = false;
int publish_max_voxel_layer = 0;
bool publish_point_cloud = false;
int pub_point_cloud_skip = 1;

double intensity_min_thr = 0.0, intensity_max_thr = 1.0;

// params for voxel mapping algorithm
double min_eigen_value = 0.003;
int max_layer = 0;

int max_cov_points_size = 50;
int max_points_size = 50;
double sigma_num = 2.0;
double max_voxel_size = 1.0;
std::vector<int> layer_size;
// Eigen::Vector3d layer_size(20, 10, 10);
// avia
// velodyne
// Eigen::Vector3d layer_size(10, 6, 6);

// record point usage
double mean_effect_points = 0;

//? ds
double mean_ds_points = 0;
double mean_raw_points = 0;

// record time
double undistort_time_mean = 0;
double down_sample_time_mean = 0;
double calculate_cov_time_mean = 0;
double scan_match_time_mean = 0;
double ekf_solve_time_mean = 0;
double map_update_time_mean = 0;

mutex mtx_buffer;

// ? 
condition_variable sig_buffer;


Eigen::Vector3d last_odometry(0, 0, 0);
Eigen::Matrix3d last_R = Eigen::Matrix3d::Zero();
double trajectory_len = 0;

string map_file_path, lidar_topic, imu_topic;
int scanIdx = 0;

int iterCount, feature_down_size, NUM_MAX_ITERATIONS, laser_cloud_valid_number, effect_feature_number, time_log_counter, publish_count = 0;

double first_lidar_time = 0;
double lidar_end_time = 0;

double residual_mean_last = 0.05;
double total_distance = 0;
double gyroscope_cov_scale, accelerater_cov_scale;
double last_timestamp_lidar, last_timestamp_imu = -1.0;
double filter_size_corner_min, filter_size_surface_min, fov_deg;
double map_incremental_time, kdtree_search_time, total_time, scan_match_time, solve_time;
bool lidar_pushed, flg_reset, flg_exit = false;
bool dense_map_en = true;
bool flg_first_scan = true;

deque<PointCloudXYZI::Ptr> lidar_buffer;
deque<double> time_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

// surf feature in map
PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr cube_points_add(new PointCloudXYZI());
PointCloudXYZI::Ptr feature_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feature_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feature_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr map_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr norm_vector(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laser_cloud_origin(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laser_cloud_no_effect(new PointCloudXYZI(100000, 1));
pcl::VoxelGrid<PointType> down_size_filter_surface;
pcl::VoxelGrid<PointType> down_size_filter_map;

V3D euler_current;
V3D position_last(Zero3d);

// estimator inputs and output;
MeasureGroup Measures;

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

nav_msgs::Path path;
nav_msgs::Odometry odometry_after_mapped;
geometry_msgs::Quaternion geometry_quaternion;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_preprocess(new Preprocess());

void SigHandle(int sig) {
  flg_exit = true;
  ROS_WARN("catch sig %d", sig);
  sig_buffer.notify_all();
}

const bool intensity_contrast(PointType &x, PointType &y) {
  return (x.intensity > y.intensity);
};

const bool var_contrast(pointWithCov &x, pointWithCov &y) {
  return (x.cov.diagonal().norm() < y.cov.diagonal().norm());
};

inline void dump_lio_state_to_log(FILE *fp) {

  V3D rot_ang(Log(state.rot_end));
  fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
  fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2)); // Angle
  fprintf(fp, "%lf %lf %lf ", state.pos_end(0), state.pos_end(1),
          state.pos_end(2)); // Pos
  fprintf(fp, "%lf %lf %lf ", state.vel_end(0), state.vel_end(1),
          state.vel_end(2)); // Vel
  fprintf(fp, "%lf %lf %lf ", state.bias_g(0), state.bias_g(1),
          state.bias_g(2)); // omega
  fprintf(fp, "%lf %lf %lf %lf ", scan_match_time, solve_time,
          map_incremental_time,
          total_time); // scan match, ekf, map incre, total
  fprintf(fp, "%d %d %d", feature_undistort->points.size(),
          feature_down_body->points.size(),
          effect_feature_number); // raw point number, effect number
  fprintf(fp, "\r\n");
  fflush(fp);
  
}

inline void kitti_log(FILE *fp) {

  Eigen::Matrix4d T_lidar_to_cam;
  T_lidar_to_cam << 0.00042768, -0.999967, -0.0080845, -0.01198, -0.00721062,
      0.0080811998, -0.99994131, -0.0540398, 0.999973864, 0.00048594,
      -0.0072069, -0.292196, 0, 0, 0, 1.0;
  V3D rot_ang(Log(state.rot_end));
  MD(4, 4) T;
  T.block<3, 3>(0, 0) = state.rot_end;
  T.block<3, 1>(0, 3) = state.pos_end;
  T(3, 0) = 0;
  T(3, 1) = 0;
  T(3, 2) = 0;
  T(3, 3) = 1;
  T = T_lidar_to_cam * T * T_lidar_to_cam.inverse();
  for (int i = 0; i < 3; i++) {
    if (i == 2)
      fprintf(fp, "%lf %lf %lf %lf", T(i, 0), T(i, 1), T(i, 2), T(i, 3));
    else
      fprintf(fp, "%lf %lf %lf %lf ", T(i, 0), T(i, 1), T(i, 2), T(i, 3));
  }
  fprintf(fp, "\n");
  // Eigen::Quaterniond q(state.rot_end);
  // fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf \r\n", state.pos_end[0],
  //         state.pos_end[1], state.pos_end[2], q.w(), q.x(), q.y(), q.z());
  fflush(fp);
}

// 8 items
// ((vect3, pos)) ((SO3, rot)) ((SO3, offset_R_L_I)) ((vect3, offset_T_L_I)) ((vect3, vel)) ((vect3, bg)) ((vect3, ba)) ((S2, grav))

// 7 items
//float64[] rot_end      # the estimated attitude (rotation matrix) at the end lidar point
//float64[] pos_end      # the estimated position at the end lidar point (world frame)
//float64[] vel_end      # the estimated velocity at the end lidar point (world frame)
//float64[] bias_gyr     # gyroscope bias
//float64[] bias_acc     # accelerator bias
//float64[] gravity      # the estimated gravity acceleration
//float64[] cov          # states covariance
// Pose6D[] IMUpose        # 6D pose at each imu measurements


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

template <typename T>
void point_body_to_world(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po) {
  V3D p_body(pi[0], pi[1], pi[2]);
  p_body = p_body + Lidar_offset_to_IMU;
  V3D p_global(state.rot_end * (p_body) + state.pos_end);
  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
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
double lidar_time_offset = 0.0;
void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    auto time_offset = lidar_time_offset;
    //std::printf("lidar offset:%f\n", lidar_time_offset);
    ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
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

void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
  mtx_buffer.lock();
  // cout << "got feature" << endl;
  if (msg->header.stamp.toSec() < last_timestamp_lidar) {
    ROS_ERROR("lidar loop back, clear buffer");
    lidar_buffer.clear();
  }
  // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_preprocess->process(msg, ptr);
  lidar_buffer.push_back(ptr);
  time_buffer.push_back(msg->header.stamp.toSec());
  last_timestamp_lidar = msg->header.stamp.toSec();

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

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }
    cout<<"=========[0]==========="<<endl;
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
    cout<<"=========[1]==========="<<endl;
    cout<<imu_buffer.size()<<endl;
    cout<<"=========[1.0]==========="<<endl;
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
    cout<<"=========[2]==========="<<endl;
    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
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

void publish_effect_world(const ros::Publisher &pubLaserCloudEffect, const ros::Publisher &pubPointWithCov, const std::vector<ptpl> &point_to_plane_list) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr effect_cloud_world(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effect_feature_number, 1));
  visualization_msgs::MarkerArray ma_line;
  visualization_msgs::Marker m_line;
  m_line.type = visualization_msgs::Marker::LINE_LIST;
  m_line.action = visualization_msgs::Marker::ADD;
  m_line.ns = "lines";
  m_line.color.a = 0.5; // Don't forget to set the alpha!
  m_line.color.r = 1.0;
  m_line.color.g = 1.0;
  m_line.color.b = 1.0;
  m_line.scale.x = 0.01;
  m_line.pose.orientation.w = 1.0;
  m_line.header.frame_id = "camera_init";
  for (int i = 0; i < point_to_plane_list.size(); i++) {
    Eigen::Vector3d p_c = point_to_plane_list[i].point;
    Eigen::Vector3d p_w = state.rot_end * (p_c) + state.pos_end;
    pcl::PointXYZRGB pi;
    pi.x = p_w[0];
    pi.y = p_w[1];
    pi.z = p_w[2];
    // float v = laserCloudWorld->points[i].intensity / 200;
    // v = 1.0 - v;
    // uint8_t r, g, b;
    // mapJet(v, 0, 1, r, g, b);
    // pi.r = r;
    // pi.g = g;
    // pi.b = b;
    effect_cloud_world->points.push_back(pi);
    m_line.points.clear();
    geometry_msgs::Point p;
    p.x = p_w[0];
    p.y = p_w[1];
    p.z = p_w[2];
    m_line.points.push_back(p);
    p.x = point_to_plane_list[i].center(0);
    p.y = point_to_plane_list[i].center(1);
    p.z = point_to_plane_list[i].center(2);
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
  }
  int max_num = 20000;
  for (int i = point_to_plane_list.size(); i < max_num; i++) {
    m_line.color.a = 0;
    ma_line.markers.push_back(m_line);
    m_line.id++;
  }
  pubPointWithCov.publish(ma_line);

  sensor_msgs::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*effect_cloud_world, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp =
      ros::Time::now(); //.fromSec(last_timestamp_lidar);
  laserCloudFullRes3.header.frame_id = "camera_init";
  pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_no_effect(const ros::Publisher &pubLaserCloudNoEffect) {
  sensor_msgs::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laser_cloud_no_effect, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp =
      ros::Time::now(); //.fromSec(last_timestamp_lidar);
  laserCloudFullRes3.header.frame_id = "camera_init";
  pubLaserCloudNoEffect.publish(laserCloudFullRes3);
}

void publish_effect(const ros::Publisher &pubLaserCloudEffect) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr effect_cloud_world(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effect_feature_number, 1));
  for (int i = 0; i < effect_feature_number; i++) {
    RGB_point_body_to_world(&laser_cloud_origin->points[i], &laserCloudWorld->points[i]);
    pcl::PointXYZRGB pi;
    pi.x = laserCloudWorld->points[i].x;
    pi.y = laserCloudWorld->points[i].y;
    pi.z = laserCloudWorld->points[i].z;
    float v = laserCloudWorld->points[i].intensity / 100;
    v = 1.0 - v;
    uint8_t r, g, b;
    mapJet(v, 0, 1, r, g, b);
    pi.r = r;
    pi.g = g;
    pi.b = b;
    effect_cloud_world->points.push_back(pi);
  }

  sensor_msgs::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp =
      ros::Time::now(); //.fromSec(last_timestamp_lidar);
  laserCloudFullRes3.header.frame_id = "camera_init";
  pubLaserCloudEffect.publish(laserCloudFullRes3);
}

template <typename T> void set_posestamp(T &out) {
  out.position.x = state.pos_end(0);
  out.position.y = state.pos_end(1);
  out.position.z = state.pos_end(2);
  out.orientation.x = geometry_quaternion.x;
  out.orientation.y = geometry_quaternion.y;
  out.orientation.z = geometry_quaternion.z;
  out.orientation.w = geometry_quaternion.w;
}

void publish_odometry(const ros::Publisher &pubOdomAftMapped) {
  odometry_after_mapped.header.frame_id = "camera_init";
  odometry_after_mapped.child_frame_id = "aft_mapped";
  odometry_after_mapped.header.stamp =
      ros::Time::now(); // ros::Time().fromSec(last_timestamp_lidar);
  set_posestamp(odometry_after_mapped.pose.pose);
  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion q;
  transform.setOrigin(
      tf::Vector3(state.pos_end(0), state.pos_end(1), state.pos_end(2)));
  q.setW(geometry_quaternion.w);
  q.setX(geometry_quaternion.x);
  q.setY(geometry_quaternion.y);
  q.setZ(geometry_quaternion.z);
  transform.setRotation(q);
  br.sendTransform(tf::StampedTransform(transform, odometry_after_mapped.header.stamp, "camera_init", "aft_mapped"));
  pubOdomAftMapped.publish(odometry_after_mapped);
}

void publish_mavros(const ros::Publisher &mavros_pose_publisher) {
  msg_body_pose.header.stamp = ros::Time::now();
  msg_body_pose.header.frame_id = "camera_odom_frame";
  set_posestamp(msg_body_pose.pose);
  mavros_pose_publisher.publish(msg_body_pose);
}

void publish_path(const ros::Publisher pubPath) {
  set_posestamp(msg_body_pose.pose);
  msg_body_pose.header.stamp = ros::Time::now();
  msg_body_pose.header.frame_id = "camera_init";
  path.poses.push_back(msg_body_pose);
  pubPath.publish(path);
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "voxelMapping");
  ros::NodeHandle nh;

  double ranging_cov = 0.0;
  double angle_cov = 0.0;
  std::vector<double> layer_point_size;

  // cummon params
  nh.param<string>("common/lid_topic", lidar_topic, "/livox/lidar");
  nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");

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
  nh.param<vector<double>>("mapping/layer_point_size", layer_point_size,
                           vector<double>());
  nh.param<int>("mapping/max_layer", max_layer, 2);
  nh.param<double>("mapping/voxel_size", max_voxel_size, 1.0);
  nh.param<double>("mapping/down_sample_size", filter_size_surface_min, 0.2);
  // std::cout << "filter_size_surface_min:" << filter_size_surface_min << std::endl;
  nh.param<double>("mapping/plannar_threshold", min_eigen_value, 0.01);

  // preprocess params
  nh.param<double>("preprocess/blind", p_preprocess->blind, 0.01);
  nh.param<bool>("preprocess/calib_laser", calib_laser, false);
  nh.param<int>("preprocess/lidar_type", p_preprocess->lidar_type, AVIA);
  nh.param<int>("preprocess/scan_line", p_preprocess->N_SCANS, 16);
  nh.param<int>("preprocess/point_filter_num", p_preprocess->point_filter_num, 1);

  // visualization params
  nh.param<bool>("visualization/pub_voxel_map", publish_voxel_map, false);
  nh.param<int>("visualization/publish_max_voxel_layer",
                publish_max_voxel_layer, 0);
  nh.param<bool>("visualization/pub_point_cloud", publish_point_cloud, true);
  nh.param<int>("visualization/pub_point_cloud_skip", pub_point_cloud_skip, 1);
  nh.param<bool>("visualization/dense_map_enable", dense_map_en, false);

  // result params
  nh.param<bool>("Result/write_kitti_log", write_kitti_log, 'false');
  nh.param<string>("Result/result_path", result_path, "");
  cout << "p_preprocess->lidar_type " << p_preprocess->lidar_type << endl;
  
  
  for (int i = 0; i < layer_point_size.size(); i++) {
    layer_size.push_back(layer_point_size[i]);
  }

  ros::Subscriber sub_pcl = p_preprocess->lidar_type == AVIA ? nh.subscribe(lidar_topic, 200000, livox_pcl_cbk) : nh.subscribe(lidar_topic, 200000, standard_pcl_cbk);
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
  VD(DIM_STATE) solution;
  MD(DIM_STATE, DIM_STATE) G, H_Transpose_multiply_R_inverse_multiply_H, I_STATE;
  
  V3D R_add, T_add;
  
  StatesGroup state_propagate;
  
  PointCloudXYZI::Ptr corrospend_norm_vector(new PointCloudXYZI(100000, 1));
  
  int frame_number = 0;
  double deltaT, deltaR, average_time_consumption = 0;
  bool flg_EKF_inited, flg_EKF_converged, EKF_stop_flg = 0, is_first_frame = true;
  
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

  G.setZero();
  H_Transpose_multiply_R_inverse_multiply_H.setZero();
  I_STATE.setIdentity();

  /*** debug record ***/
  // FILE for kitti
  FILE *fp_kitti;

  // open FILE
  if (write_kitti_log) {
    fp_kitti = fopen(result_path.c_str(), "w");
  }

  signal(SIGINT, SigHandle);
  ros::Rate rate(5000);
  bool status = ros::ok();

  // initialize the VoxelMap
  // for Plane Map
  bool init_map = false;
  std::unordered_map<VOXEL_LOC, OctoTree *> voxel_map;
  last_R << 1, 0, 0, 0, 1, 0, 0, 0, 1;

  while (status) {
  
    if (flg_exit)
      break;
      
      
    ros::spinOnce();
    
    
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
      
      // time
      std::cout << "scanIdx:" << scanIdx << std::endl;
      // std::cout << " init rot cov:" << std::endl
      //           << state.cov.block<3, 3>(0, 0) << std::endl;
      auto undistort_start = std::chrono::high_resolution_clock::now();
      
      
      // [ IMU DONE ]
      //imu process to feature undistort
      p_imu->Process(Measures, state, feature_undistort);
      
      auto undistort_end  = std::chrono::high_resolution_clock::now();
      auto undistort_time = std::chrono::duration_cast<std::chrono::duration<double>>( undistort_end - undistort_start ).count() * 1000;
        
      state_propagate = state;

      // something wrong
      if (feature_undistort->empty() || (feature_undistort == NULL))
      {
        ROS_WARN("No point, skip this scan!\n");
        continue;
      }


      // =========== [step 1, get the input lidar point with variance in lidar frame, initilize voxel map with covariance ] =========== //
      
      flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;
      cout<<"=====================world_lidar->size[0]======================"<<endl;
      if (flg_EKF_inited && !init_map) {
        cout<<"=====================world_lidar->size[1]======================"<<endl;
        pcl::PointCloud<pcl::PointXYZI>::Ptr world_lidar( new pcl::PointCloud<pcl::PointXYZI>);
        Eigen::Quaterniond q(state.rot_end);
        
        // feature_undistort to world_lidar
        transformLidar(state, p_imu, feature_undistort, world_lidar);
        std::vector<pointWithCov> pv_list;
        
        cout<<"=====================world_lidar->size======================"<<endl;
        cout<<world_lidar->size()<<endl;
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
          
          // cov = cov_lidar
          M3D cov;
          calcBodyCov(point_this, ranging_cov, angle_cov, cov);


          point_this += Lidar_offset_to_IMU;
          
          M3D point_crossmat;
          
          point_crossmat << SKEW_SYM_MATRX(point_this);
          cov = state.rot_end * cov * state.rot_end.transpose() +
                (-point_crossmat) * state.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose() +
                state.cov.block<3, 3>(3, 3);
          
          // cov = cov_world      
          pv.cov = cov;
          
          pv_list.push_back(pv);
          
          Eigen::Vector3d sigma_pv = pv.cov.diagonal();
          sigma_pv[0] = sqrt(sigma_pv[0]);
          sigma_pv[1] = sqrt(sigma_pv[1]);
          sigma_pv[2] = sqrt(sigma_pv[2]);
        }

	cout<<"========buildVoxelMap=========="<<endl;
	cout<<pv_list.size()<<endl;      //16026
	//while(1){}
	

        buildVoxelMap(pv_list, max_voxel_size, max_layer, layer_size, max_points_size, max_points_size, min_eigen_value, voxel_map);
                      
        std::cout << "build voxel map" << std::endl;


        scanIdx++;
        
        if (1) {
          pubVoxelMap(voxel_map, publish_max_voxel_layer, voxel_map_pub);
        }
        
        init_map = true;
        continue;
        
      }

      /*** downsample the feature points in a scan ***/
      auto t_downsample_start = std::chrono::high_resolution_clock::now();
      
      down_size_filter_surface.setInputCloud(feature_undistort);
      down_size_filter_surface.filter(*feature_down_body);
      
      auto t_downsample_end = std::chrono::high_resolution_clock::now();
      
      std::cout << " feats size:" << feature_undistort->size() << ", down size:" << feature_down_body->size() << std::endl;
                
      auto t_downsample = std::chrono::duration_cast<std::chrono::duration<double>>( t_downsample_end - t_downsample_start).count() * 1000;

      sort(feature_down_body->points.begin(), feature_down_body->points.end(), time_list);

      int rematch_number = 0;
      bool nearest_search_en = true;
      double total_residual;

      scan_match_time = 0.0;

      std::vector<M3D> body_variance;
      std::vector<M3D> crossmat_list;
      
      
      // =========== [step 2, run the esekf kalman filter to get the state and covariance ] =========== //

      /*** iterated state estimation ***/
      auto calculate_point_cov_start = std::chrono::high_resolution_clock::now();
      
      for (size_t i = 0; i < feature_down_body->size(); i++) 
      {
      
        V3D point_this(feature_down_body->points[i].x, feature_down_body->points[i].y, feature_down_body->points[i].z);
        
        if (point_this[2] == 0) {
          point_this[2] = 0.001;
        }
        
        M3D cov; // cov = cov_lidar
        
        calcBodyCov(point_this, ranging_cov, angle_cov, cov);
        
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_this);
        
        crossmat_list.push_back(point_crossmat);
        body_variance.push_back(cov);
        
      }
      
      auto calc_point_cov_end = std::chrono::high_resolution_clock::now();
      
      double calc_point_cov_time = std::chrono::duration_cast<std::chrono::duration<double>>( calc_point_cov_end - calculate_point_cov_start).count() * 1000;
          

      for (iterCount = 0; iterCount < NUM_MAX_ITERATIONS; iterCount++) {
      
        laser_cloud_origin->clear();
        laser_cloud_no_effect->clear();
        corrospend_norm_vector->clear();
        
        total_residual = 0.0;

        std::vector<ptpl> point_to_plane_list;
        /** LiDAR match based on 3 sigma criterion **/

        vector<pointWithCov> pv_list;
        std::vector<M3D> var_list;
        
        pcl::PointCloud<pcl::PointXYZI>::Ptr world_lidar( new pcl::PointCloud<pcl::PointXYZI> );
        transformLidar(state, p_imu, feature_down_body, world_lidar);
        
        for (size_t i = 0; i < feature_down_body->size(); i++) {
        
          pointWithCov pv;
          
          pv.point << feature_down_body->points[i].x, feature_down_body->points[i].y, feature_down_body->points[i].z;
          pv.point_world << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
          
          M3D cov = body_variance[i];
          
          M3D point_crossmat = crossmat_list[i];
          M3D R_var = state.cov.block<3, 3>(0, 0);
          M3D T_var = state.cov.block<3, 3>(3, 3);
          // transformLiDARCovarianceToWorldCovariance()
          cov = state.rot_end * cov * state.rot_end.transpose() + (-point_crossmat) * R_var * (-point_crossmat.transpose()) + T_var;
                
          pv.cov = cov;
          pv_list.push_back(pv);
          var_list.push_back(cov);
          
        }
        
        auto scan_match_time_start = std::chrono::high_resolution_clock::now();

        std::vector<V3D> non_match_list;
        BuildResidualListOMP(voxel_map, max_voxel_size, 3.0, max_layer, pv_list, point_to_plane_list, non_match_list);
        
        auto scan_match_time_end = std::chrono::high_resolution_clock::now();

        effect_feature_number = 0;
        
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
          
          total_residual += fabs(distance);
        }
        
        residual_mean_last = total_residual / effect_feature_number;
        
        scan_match_time += std::chrono::duration_cast<std::chrono::duration<double>>( scan_match_time_end - scan_match_time_start).count() * 1000;

        auto t_solve_start = std::chrono::high_resolution_clock::now();

        /*** Computation of Measuremnt Jacobian matrix H and measurents vector
         * ***/
         
        MatrixXd Hsub( effect_feature_number, 6);
        MatrixXd Hsub_Transpose_multiply_R_inverse( 6, effect_feature_number );
        
        VectorXd R_inverse( effect_feature_number );
        VectorXd measurement_vector( effect_feature_number );
        
        double max_distance = 0;

        for (int i = 0; i < effect_feature_number; i++) {
        
          const PointType &laser_point = laser_cloud_origin->points[i];
          V3D point_this(laser_point.x, laser_point.y, laser_point.z);
          
          M3D cov;
          
          if (calib_laser) {
            calcBodyCov(point_this, ranging_cov, CALIB_ANGLE_COV, cov);
          } else {
            calcBodyCov(point_this, ranging_cov, angle_cov, cov);
          }

          cov = state.rot_end * cov * state.rot_end.transpose();
          
          M3D point_crossmat;
          point_crossmat << SKEW_SYM_MATRX(point_this);
          
          const PointType &norm_point = corrospend_norm_vector->points[i];
          V3D norm_vector(norm_point.x, norm_point.y, norm_point.z);
          
          V3D point_world = state.rot_end * point_this + state.pos_end;
          
          // /*** get the normal vector of closest surface/corner ***/
          Eigen::Matrix<double, 1, 6> J_nq;
          
          J_nq.block<1, 3>(0, 0) = point_world - point_to_plane_list[i].center;
          J_nq.block<1, 3>(0, 3) = -point_to_plane_list[i].normal;
          double sigma_l = J_nq * point_to_plane_list[i].plane_cov * J_nq.transpose();
          
          
          R_inverse(i) = 1.0 / (sigma_l + norm_vector.transpose() * cov * norm_vector);
          
          
          laser_cloud_origin->points[i].intensity = sqrt(R_inverse(i));
          laser_cloud_origin->points[i].normal_x =  corrospend_norm_vector->points[i].intensity;
          laser_cloud_origin->points[i].normal_y =  sqrt(sigma_l);
          laser_cloud_origin->points[i].normal_z =  sqrt(norm_vector.transpose() * cov * norm_vector);
          laser_cloud_origin->points[i].curvature = sqrt(sigma_l + norm_vector.transpose() * cov * norm_vector);

          /*** calculate the Measuremnt Jacobian matrix H ***/
          V3D A(point_crossmat * state.rot_end.transpose() * norm_vector);
          Hsub.row(i) << VEC_FROM_ARRAY(A), norm_point.x, norm_point.y, norm_point.z;
          Hsub_Transpose_multiply_R_inverse.col(i) << A[0] * R_inverse(i), A[1] * R_inverse(i), A[2] * R_inverse(i), 
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
        EKF_stop_flg = false;
        flg_EKF_converged = false;
        
        /*** Iterative Kalman Filter Update ***/
        if (!flg_EKF_inited) {
        
          cout << "||||||||||Initiallizing LiDar||||||||||" << endl;
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
          
          EKF_stop_flg = true;
          
        } else {
        
          // [DONE]
          auto &&Hsub_Transpose = Hsub.transpose();
          
          // [DONE]
          H_Transpose_multiply_R_inverse_multiply_H.block<6, 6>(0, 0) = Hsub_Transpose_multiply_R_inverse * Hsub;
          
          // [DONE]
          MD(DIM_STATE, DIM_STATE) &&Q_inverse = (H_Transpose_multiply_R_inverse_multiply_H + (state.cov).inverse()).inverse(); // DIM_STATE=18
          // [DONE]
          K = Q_inverse.block<DIM_STATE, 6>(0, 0) * Hsub_Transpose_multiply_R_inverse;
          // [DONE]
          auto vector_box_minus = state_propagate - state;
          
          // MatrixXd Hsub( effect_feature_number, 6);
          // MatrixXd Hsub_Transpose_multiply_R_inverse( 6, effect_feature_number );
          // VectorXd R_inverse( effect_feature_number );
          // VectorXd measurement_vector( effect_feature_number );
          
          // MatrixXd K(DIM_STATE, effect_feature_number);
          
          //solution = K * measurement_vector + vector_box_minus - K * Hsub * vector_box_minus.block<6, 1>(0, 0);
          solution = vector_box_minus + K * ( measurement_vector - Hsub * vector_box_minus.block<6, 1>(0, 0) );
          
          
          
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
          
          state += solution;

          R_add = solution.block<3, 1>(0, 0);
          T_add = solution.block<3, 1>(3, 0);


          //cout<<"========================R,t======================"<<endl; 
          //cout<<state.rot_end<<","<<endl;     
          //cout<<state.pos_end<<","<<endl;
          cout<<"==========v[1]_after_eskf============"<<endl; 
          cout<<state.vel_end<<endl;
          cout<<"==========dv============"<<endl; 
          cout<<solution.block<3, 1>(6, 0)<<endl;
          
          cout<<"=========measurement_vector========="<<endl;
          cout<<measurement_vector(0)<<endl;
          cout<<measurement_vector(1)<<endl;
          cout<<measurement_vector(2)<<endl;
          cout<<max_distance<<endl;
          
                    
          if ((R_add.norm() * 57.3 < 0.01) && (T_add.norm() * 100 < 0.015)) {
            flg_EKF_converged = true;
          }

          deltaR = R_add.norm() * 57.3;
          deltaT = T_add.norm() * 100;
          
        }
        
        euler_current = RotMtoEuler(state.rot_end);
        
        /*** Rematch Judgement ***/
        
        nearest_search_en = false;
        
        if (flg_EKF_converged ||
            ((rematch_number == 0) && (iterCount == (NUM_MAX_ITERATIONS - 2)))) {
            
          nearest_search_en = true;
          rematch_number++;
          
        }

        /*** Convergence Judgements and Covariance Update ***/
        
        if (!EKF_stop_flg &&
            (rematch_number >= 2 || (iterCount == NUM_MAX_ITERATIONS - 1))) {
            
          if (flg_EKF_inited) {
          
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
          
          EKF_stop_flg = true;
          
        }
        
        auto t_solve_end = std::chrono::high_resolution_clock::now();
        solve_time += std::chrono::duration_cast<std::chrono::duration<double>>( t_solve_end - t_solve_start).count() * 1000;

        if (EKF_stop_flg)
          break;
      }

      // ======================================================[step 3, update the voxel map of covariance ]========================================================= //
	    
      /*** add the  points to the voxel map ***/
      
      auto map_incremental_start = std::chrono::high_resolution_clock::now();
      
      pcl::PointCloud<pcl::PointXYZI>::Ptr world_lidar( new pcl::PointCloud<pcl::PointXYZI> );
      
      transformLidar(state, p_imu, feature_down_body, world_lidar);
      std::vector<pointWithCov> pv_list;
      
      for (size_t i = 0; i < world_lidar->size(); i++) {
      
        pointWithCov pv;
        pv.point << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
        
        M3D point_crossmat = crossmat_list[i];
        M3D cov = body_variance[i];
        
        cov = state.rot_end * cov * state.rot_end.transpose() + (-point_crossmat) * state.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose() + state.cov.block<3, 3>(3, 3);
        pv.cov = cov;
        pv_list.push_back(pv);
        
      }
      
      std::sort(pv_list.begin(), pv_list.end(), var_contrast);
      
      updateVoxelMapOMP(pv_list, max_voxel_size, max_layer, layer_size, max_points_size, max_points_size, min_eigen_value, voxel_map);
      
      auto map_incremental_end = std::chrono::high_resolution_clock::now();
      map_incremental_time = std::chrono::duration_cast<std::chrono::duration<double>>( map_incremental_end - map_incremental_start).count() * 1000;

      total_time = t_downsample + scan_match_time + solve_time + map_incremental_time + undistort_time + calc_point_cov_time;
                   
      /******* Publish functions:  *******/
      publish_odometry(pubOdomAftMapped);
      publish_path(pubPath);
      
      tf::Transform transform;
      tf::Quaternion q;
      
      transform.setOrigin( tf::Vector3(state.pos_end(0), state.pos_end(1), state.pos_end(2)));
      
      q.setW(geometry_quaternion.w);
      q.setX(geometry_quaternion.x);
      q.setY(geometry_quaternion.y);
      q.setZ(geometry_quaternion.z);
      
      transform.setRotation(q);
      transformLidar(state, p_imu, feature_down_body, world_lidar);
      
      sensor_msgs::PointCloud2 pub_cloud;
      pcl::toROSMsg(*world_lidar, pub_cloud);
      
      pub_cloud.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
      pub_cloud.header.frame_id = "camera_init";
      
      if (true) {
      //if (publish_point_cloud) {
        publish_frame_world(pubLaserCloudFullRes, pub_point_cloud_skip);
      }

      if (publish_voxel_map) {
        pubVoxelMap(voxel_map, publish_max_voxel_layer, voxel_map_pub);
      }

      publish_effect(pubLaserCloudEffect);

      frame_number++;
      mean_raw_points =	mean_raw_points * (frame_number - 1) / frame_number + (double)(feature_undistort->size()) / frame_number;
      mean_ds_points  = 	mean_ds_points * (frame_number - 1) / frame_number + (double)(feature_down_body->size()) / frame_number;
      mean_effect_points = 	mean_effect_points * (frame_number - 1) / frame_number + (double)effect_feature_number / frame_number;
      
      undistort_time_mean = 		undistort_time_mean * (frame_number - 1) / frame_number + (undistort_time) / frame_number;
      down_sample_time_mean = 	down_sample_time_mean * (frame_number - 1) / frame_number + (t_downsample) / frame_number;
      calculate_cov_time_mean = 	calculate_cov_time_mean * (frame_number - 1) / frame_number + (calc_point_cov_time) / frame_number;
      scan_match_time_mean = 		scan_match_time_mean * (frame_number - 1) / frame_number + (scan_match_time) / frame_number;
      ekf_solve_time_mean = 		ekf_solve_time_mean * (frame_number - 1) / frame_number + (solve_time) / frame_number;
      map_update_time_mean = 		map_update_time_mean * (frame_number - 1) / frame_number + (map_incremental_time) / frame_number;
      average_time_consumption = 	average_time_consumption * (frame_number - 1) / frame_number + (total_time) / frame_number;

      time_log_counter++;
      cout << "[ Time ]: "
           << "average undistort: " << undistort_time_mean << std::endl;
      cout << "[ Time ]: "
           << "average down sample: " << down_sample_time_mean << std::endl;
      cout << "[ Time ]: "
           << "average calc cov: " << calculate_cov_time_mean << std::endl;
      cout << "[ Time ]: "
           << "average scan match: " << scan_match_time_mean << std::endl;
      cout << "[ Time ]: "
           << "average solve: " << ekf_solve_time_mean << std::endl;
      cout << "[ Time ]: "
           << "average map incremental: " << map_update_time_mean << std::endl;
      cout << "[ Time ]: "
           << " average total " << average_time_consumption << endl;

      if (write_kitti_log) {
        kitti_log(fp_kitti);
      }

      scanIdx++;
    }
    
    status = ros::ok();
    rate.sleep();
  }
  return 0;
}
