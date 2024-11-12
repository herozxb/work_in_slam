// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include "voxel_map_util.hpp"
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/crop_box.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/gicp.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <thread>



#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/narf.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/gicp.h>
#include <pcl/range_image/range_image_planar.h>
#include <pcl/visualization/cloud_viewer.h>

#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

ros::Publisher pub_history_keyframes_;
ros::Publisher pub_recent_keyframes_;
ros::Publisher pub_icp_keyframes_;
ros::Publisher pub_cloud_surround_;


#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_gicp_st.hpp>

// observability
Eigen::Vector3d scales;

// Global Parameters
const float max_correspondence_distance = 0.07f;
const int max_iterations = 5;

// Initial Guess (Identity matrix)
Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();

std::mutex result_mutex;

// A structure to hold the result of each GICP process
struct GICPResult {
    Eigen::Matrix4f transformation;
    double alignment_error;
    // Other relevant data (e.g., inliers, covariance, etc.)
};

void runGICP(pcl::PointCloud<pcl::PointXYZ>::Ptr source,
             pcl::PointCloud<pcl::PointXYZ>::Ptr target,
             GICPResult &result,
             float max_correspondence_dist = 0.05,
             int max_iterations = 50,
             Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity()) {
    
/*
    clock_t start = clock(); // Get the current time in clock ticks

    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
    pcl::PointCloud<pcl::PointXYZ> aligned;
    
    gicp.setMaxCorrespondenceDistance(max_correspondence_dist);
    gicp.setMaximumIterations(max_iterations);
    gicp.setInputSource(source);
    gicp.setInputTarget(target);
    
    gicp.align(aligned, initial_guess);

    //std::lock_guard<std::mutex> lock(result_mutex);
    if (gicp.hasConverged()) {
        result.transformation = gicp.getFinalTransformation();
        result.alignment_error = gicp.getFitnessScore();
        std::cout << "============= GICP =============" << std::endl;
        std::cout << gicp.getFinalTransformation() << std::endl;
        std::cout << gicp.getFitnessScore() << std::endl;
        
    } else {
        std::cerr << "GICP failed to converge" << std::endl;
    }
    

    clock_t end = clock(); // Get the current time in clock ticks

    double runningTime = (double)(end - start) / CLOCKS_PER_SEC; // Convert clock ticks to seconds

    std::cout << "Running time:[0] " << runningTime << " seconds" << std::endl;
   //*/ 
    
    
    //clock_t start_1 = clock(); // Get the current time in clock ticks

    // Initialize fgicp_mt object (multi-threaded FastGICP)
    fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>::Ptr fast_gicp_speed(new fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>());

    // Set number of threads for multi-threading
    fast_gicp_speed->setNumThreads(1); // Adjust number of threads as needed
    fast_gicp_speed->setMaxCorrespondenceDistance(max_correspondence_dist); // Set maximum correspondence distance
    fast_gicp_speed->setMaximumIterations(max_iterations); // Set maximum iterations

    // Set input source and target clouds
    fast_gicp_speed->setInputSource(source);
    fast_gicp_speed->setInputTarget(target);

    // Align the source cloud to the target cloud
    //pcl::PointCloud<pcl::PointXYZ> aligned(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ> aligned;
    fast_gicp_speed->align(aligned);

    // Local results to avoid locking during the whole process
    Eigen::Matrix4f transformation;
    double alignment_error;
    bool converged = fast_gicp_speed->hasConverged();

    if (converged) {
        transformation = fast_gicp_speed->getFinalTransformation();
        alignment_error = fast_gicp_speed->getFitnessScore();

        std::cout << "============= fast GICP =============" << std::endl;
        std::cout << transformation << std::endl;
        std::cout << alignment_error << std::endl;
        
    } else {
        //std::cerr << "GICP failed to converge" << std::endl;
    }

    // Lock only when updating the shared resource
    {
        std::lock_guard<std::mutex> lock(result_mutex);
        if (converged) {
            result.transformation = transformation;
            result.alignment_error = alignment_error;
        }
    }
    
        
    //clock_t end_1 = clock(); // Get the current time in clock ticks

    //double runningTime_1 = (double)(end_1 - start_1) / CLOCKS_PER_SEC; // Convert clock ticks to seconds

    //std::cout << "Running time:[1] " << runningTime_1 << " seconds" << std::endl;
    
    //*/
    
    /*
    clock_t start_1 = clock(); // Get the current time in clock ticks
    // Initialize fgicp_mt object (multi-threaded FastGICP)
    fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>::Ptr fast_gicp_speed(new fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>());

    // Set number of threads for multi-threading
    fast_gicp_speed->setNumThreads(3); // or however many threads you want to use
    fast_gicp_speed->setMaxCorrespondenceDistance(max_correspondence_dist); // Adjust the value as needed
    fast_gicp_speed->setMaximumIterations(max_iterations);
    
    

    fast_gicp_speed->setInputSource(source);
    fast_gicp_speed->setInputTarget(target);

    //pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());

    // Align the source cloud to the target cloud
    pcl::PointCloud<pcl::PointXYZ> aligned;
    fast_gicp_speed->align(aligned);
    
    std::lock_guard<std::mutex> lock(result_mutex);
    if (fast_gicp_speed->hasConverged()) {
        result.transformation = fast_gicp_speed->getFinalTransformation();
        result.alignment_error = fast_gicp_speed->getFitnessScore();
        std::cout << "============= fast GICP =============" << std::endl;
        std::cout << fast_gicp_speed->getFinalTransformation() << std::endl;
        std::cout << fast_gicp_speed->getFitnessScore() << std::endl;
        
    } else {
        std::cerr << "GICP failed to converge" << std::endl;
    }
    
    

    // Get the transformation matrix
    //std::cout << "============= fast GICP =============" << std::endl;
    //Eigen::Matrix4f transformation = fast_gicp_speed->getFinalTransformation();
    //std::cout << "Transformation Matrix: \n" << transformation << std::endl;
    
    clock_t end_1 = clock(); // Get the current time in clock ticks

    double runningTime_1 = (double)(end_1 - start_1) / CLOCKS_PER_SEC; // Convert clock ticks to seconds

    std::cout << "Running time:[1] " << runningTime_1 << " seconds" << std::endl;
	//*/
}



/*
// Function to select the best GICP result
GICPResult selectBestResult(const std::vector<GICPResult>& results) {
    GICPResult best_result;
    best_result.alignment_error = std::numeric_limits<double>::max();

    for (const auto& result : results) {
        if (result.alignment_error < best_result.alignment_error) {
            best_result = result;
        }
    }
    return best_result;
}

//*/

// Helper function to compare two transformation matrices within a threshold
bool areTransformationsSimilar(const Eigen::Matrix4f &T1, const Eigen::Matrix4f &T2, float threshold) {
    return (T1 - T2).norm() < threshold;
}

// Function to select the most common GICP result
GICPResult selectBestResult(const std::vector<GICPResult>& results, float threshold = 7e-2) {
    std::map<int, int> countMap;  // Map to count occurrences of each result
    int bestIndex = -1;
    int maxCount = 0;
    

    for (size_t i = 0; i < results.size(); ++i) {
        int count = 0;
        for (size_t j = 0; j < results.size(); ++j) {
            
            if( i != j )
            {
		    cout<<"(T1 - T2).norm() = "<< (results[i].transformation - results[j].transformation).norm()<<endl;;
		    if (areTransformationsSimilar(results[i].transformation, results[j].transformation, threshold)) {
		        count++;
		    }
            
            }
        }

        if (count > maxCount) {
            maxCount = count;
            bestIndex = i;
        }
    }

    cout<<"maxCount="<<maxCount<<endl;
    cout<<"bestIndex="<<bestIndex<<endl;

    if (bestIndex != -1) {
        return results[bestIndex];
    } else {
        // If no result is found, return the first one as a fallback
        GICPResult error;
        
        error.alignment_error = 0;
        
        return error;
    }
}






V3D pos_imu;

bool not_update_vexolmap = false;

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   time_sync_en = false, extrinsic_est_en = true, path_en = true;
double lidar_time_offset = 0.0;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string  lid_topic, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_surf_min = 0;
double total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;


vector<vector<int>>  pointSearchInd_surf;
vector<PointVector>  Nearest_Points;
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
//PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
//PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;
std::vector<M3D> var_down_body;

pcl::VoxelGrid<PointType> downSizeFilterSurf;

std::vector<float> nn_dist_in_feats;
std::vector<float> nn_plane_std;
PointCloudXYZI::Ptr feats_with_correspondence(new PointCloudXYZI());

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

// params for voxel mapping algorithm
double min_eigen_value = 0.003;
int max_layer = 0;

int max_cov_points_size = 50;
int max_points_size = 50;
double sigma_num = 2.0;
double max_voxel_size = 1.0;
std::vector<int> layer_size;

double ranging_cov = 0.0;
double angle_cov = 0.0;
std::vector<double> layer_point_size;

bool publish_voxel_map = false;
int publish_max_voxel_layer = 0;

std::unordered_map<VOXEL_LOC, OctoTree *> voxel_map;

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());


pcl::PointCloud<pcl::PointXYZ>::Ptr map_final;
pcl::PointCloud<pcl::PointXYZ>::Ptr map_final_boxed;  
pcl::PointCloud<pcl::PointXYZ>::Ptr map_final_boxed_2; 
pcl::PointCloud<pcl::PointXYZ>::Ptr map_final_boxed_cov; 
pcl::PointCloud<pcl::PointXYZ>::Ptr Final;
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pre;
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_filtered;
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_boxed_for_local;

Eigen::Matrix4f Ti;   
Eigen::Matrix4f Ti_real;
Eigen::Matrix4f Ti_of_map;
Eigen::Matrix4d T;

int counter_loop;
int counter_stable_map;

ros::Publisher pub_odom_aft_mapped_2;
ros::Publisher direction_pub;
ros::Publisher marker_pub; 
ros::Publisher eigenvalue_pub;
ros::Publisher marker_pub_Hessien;

ros::Publisher mean_dir_pub;
ros::Publisher std_dir_pub;


void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

const bool var_contrast(pointWithCov &x, pointWithCov &y) {
    return (x.cov.diagonal().norm() < y.cov.diagonal().norm());
};

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
//    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;

    po->curvature = pi->curvature;
    po->normal_x = pi->normal_x;
}



pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_store_intensity(new pcl::PointCloud<pcl::PointXYZ>);

// Define HighIntensityFeature structure
struct HighIntensityFeature {
    Eigen::Vector3d point;
    double intensity;
};

// Function to detect high-intensity features
std::vector<HighIntensityFeature> detectHighIntensityFeatures(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, double intensity_threshold, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_store_edge ) {
    std::vector<HighIntensityFeature> highIntensityFeatures;

    for (const auto& point : cloud->points) {
        if (point.intensity > intensity_threshold) {
            HighIntensityFeature feature;
            feature.point = Eigen::Vector3d(point.x, point.y, point.z);
            feature.intensity = point.intensity;
            highIntensityFeatures.push_back(feature);
            
            //cloud_store_edge->points.push_back(point);
        }
    }

    return highIntensityFeatures;
}


void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    auto time_offset = lidar_time_offset;
//    std::printf("lidar offset:%f\n", lidar_time_offset);
    mtx_buffer.lock();
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() + time_offset < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec() + time_offset);
    last_timestamp_lidar = msg->header.stamp.toSec() + time_offset;
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();

    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);

    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    if (timestamp < last_timestamp_imu)
    {
//        ROS_WARN("imu loop back, clear buffer");
//        imu_buffer.clear();
        ROS_WARN("imu loop back, ignoring!!!");
        ROS_WARN("current T: %f, last T: %f", timestamp, last_timestamp_imu);
        return;
    }
    // 剔除异常数据
    if (std::abs(msg->angular_velocity.x) > 10
        || std::abs(msg->angular_velocity.y) > 10
        || std::abs(msg->angular_velocity.z) > 10) {
        ROS_WARN("Large IMU measurement!!! Drop Data!!! %.3f  %.3f  %.3f",
                 msg->angular_velocity.x,
                 msg->angular_velocity.y,
                 msg->angular_velocity.z
        );
        return;
    }

//    // 如果是第一帧 拿过来做重力对齐
//    // TODO 用多帧平均的重力
//    if (is_first_imu) {
//        double acc_vec[3] = {msg_in->linear_acceleration.x, msg_in->linear_acceleration.y, msg_in->linear_acceleration.z};
//
//        R__world__o__initial = SO3(g2R(Eigen::Vector3d(acc_vec)));
//
//        is_first_imu = false;
//    }

    last_timestamp_imu = timestamp;

    mtx_buffer.lock();

    imu_buffer.push_back(msg);
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

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {

//            std::printf("\nFirst 100 points: \n");
//            for(int i=0; i < 100; ++i){
//                std::printf("%f ", meas.lidar->points[i].curvature  / double(1000));
//            }
//
//            std::printf("\n Last 100 points: \n");
//            for(int i=100; i >0; --i){
//                std::printf("%f ", meas.lidar->points[meas.lidar->size() - i - 1].curvature / double(1000));
//            }
//            std::printf("last point offset time: %f\n", meas.lidar->points.back().curvature / double(1000));
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
//            lidar_end_time = meas.lidar_beg_time + (meas.lidar->points[meas.lidar->points.size() - 20]).curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
//            std::printf("pcl_bag_time: %f\n", meas.lidar_beg_time);
//            std::printf("lidar_end_time: %f\n", lidar_end_time);
        }

        meas.lidar_end_time = lidar_end_time;
//        std::printf("Scan start timestamp: %f, Scan end time: %f\n", meas.lidar_beg_time, meas.lidar_end_time);

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());


void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI laserCloudWorld;
        for (int i = 0; i < size; i++)
        {
            PointType const * const p = &laserCloudFullRes->points[i];
            if(p->intensity < 5){
                continue;
            }
//            if (p->x < 0 and p->x > -4
//                    and p->y < 1.5 and p->y > -1.5
//                            and p->z < 2 and p->z > -1) {
//                continue;
//            }
            PointType p_world;

            RGBpointBodyToWorld(p, &p_world);
//            if (p_world.z > 1) {
//                continue;
//            }
            laserCloudWorld.push_back(p_world);
//            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
//                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

}




// Function to compute the smoothness of local surface for each point
std::vector<float> computeSmoothness(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int k_neighbors, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_store ) {
    std::vector<float> smoothness_values(cloud->points.size(), 0.0f);

    // KdTree for finding neighbors
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    // Temporary variables
    std::vector<int> pointIdxNKNSearch(k_neighbors);
    std::vector<float> pointNKNSquaredDistance(k_neighbors);

    // Iterate over each point
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        pcl::PointXYZ searchPoint = cloud->points[i];

        // Find k-nearest neighbors
        if (kdtree.nearestKSearch(searchPoint, k_neighbors, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
            Eigen::Vector3f X_L_k_i(searchPoint.x, searchPoint.y, searchPoint.z);
            Eigen::Vector3f sum_diffs(0.0f, 0.0f, 0.0f);

            for (size_t j = 1; j < pointIdxNKNSearch.size(); ++j) { // skip the point itself (j starts from 1)
                pcl::PointXYZ neighborPoint = cloud->points[pointIdxNKNSearch[j]];
                Eigen::Vector3f X_L_k_j(neighborPoint.x, neighborPoint.y, neighborPoint.z);
                sum_diffs += (X_L_k_i - X_L_k_j);
            }

            float norm_X_L_k_i = X_L_k_i.norm();
            float norm_sum_diffs = sum_diffs.norm();

            float c = (norm_X_L_k_i > 0 && pointIdxNKNSearch.size() > 1) ? (norm_sum_diffs / (pointIdxNKNSearch.size() - 1)) / norm_X_L_k_i : 0.0f;
            smoothness_values[i] = c;
            
            if( smoothness_values[i] > 0.07 )
            {
            	cout<<"====================smoothness_values======================="<<endl;
            	cout<<smoothness_values[i]<<endl;
            	
            	cloud_store_intensity->points.push_back(searchPoint);
            }
        }
    }

    return smoothness_values;
}

// Define CornerFeature structure
struct CornerFeature {
    Eigen::Vector3d point;
    Eigen::Vector3d direction; // direction of the largest eigenvalue
};

// Function to compute covariance matrix of a local neighborhood
Eigen::Matrix3d computeCovarianceMatrix(const std::vector<Eigen::Vector3d> &points) {
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();

    for (const auto &point : points) {
        mean += point;
    }
    mean /= points.size();

    for (const auto &point : points) {
        Eigen::Vector3d diff = point - mean;
        covariance += diff * diff.transpose();
    }
    covariance /= points.size();
    return covariance;
}

// Function to detect corner features
std::vector<CornerFeature> detectCornerFeatures(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double threshold , pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_store ) {
    std::vector<CornerFeature> cornerFeatures;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    int K = 3; // Number of nearest neighbors

    for (size_t i = 0; i < cloud->points.size(); ++i) {
        std::vector<int> pointIdxNKNSearch(K);
        std::vector<float> pointNKNSquaredDistance(K);

        if (kdtree.nearestKSearch(cloud->points[i], K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
            std::vector<Eigen::Vector3d> neighborhood;

            for (size_t j = 0; j < pointIdxNKNSearch.size(); ++j) {
                pcl::PointXYZ pt = cloud->points[pointIdxNKNSearch[j]];
                neighborhood.emplace_back(pt.x, pt.y, pt.z);
            }

            if (neighborhood.size() > 3) {
                Eigen::Matrix3d covariance = computeCovarianceMatrix(neighborhood);
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigenSolver(covariance);

                // Check the largest eigenvalue to determine if the point is a corner
                double lambda1 = eigenSolver.eigenvalues()[2]; // largest eigenvalue
                double lambda2 = eigenSolver.eigenvalues()[1]; // second largest eigenvalue

                if (lambda1 > 3 * lambda2) { // simple threshold to determine if it is a corner
                    CornerFeature feature;
                    feature.point = Eigen::Vector3d(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
                    feature.direction = eigenSolver.eigenvectors().col(2); // direction of the largest eigenvalue
                    cornerFeatures.push_back(feature);
                    
		     cout<<"====================detectCornerFeatures======================="<<endl;
		     cout<< feature.point <<endl;
		     cout<< feature.direction <<endl;

		     cloud_store_intensity->points.push_back(cloud->points[i]);
                    
                    
                }
            }
        }
    }
    return cornerFeatures;
}

pcl::ModelCoefficients::Ptr coefficients_of_plane(new pcl::ModelCoefficients);

pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>);

pcl::ModelCoefficients::Ptr findPlaneInPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double radius, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane) {
    if (cloud->points.size() < 3) {
        std::cerr << "Not enough points to fit a plane." << std::endl;
        return NULL;
    }

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(radius);

    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size() == 0) {
        std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
        return NULL;
    }

    Eigen::Vector3d plane_normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
    double d = coefficients->values[3];

    Eigen::Vector3d plane_center(0, 0, 0);
    for (const auto& idx : inliers->indices) {
        plane_center += Eigen::Vector3d(cloud->points[idx].x, cloud->points[idx].y, cloud->points[idx].z);
    }
    plane_center /= inliers->indices.size();

    std::cout << "Plane center: " << plane_center.transpose() << std::endl;
    std::cout << "Plane normal: " << plane_normal.transpose() << std::endl;
    
    // Extract the points that belong to the plane (inliers)
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);  // Set to true if you want to remove inliers and keep only outliers

    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>);
    extract.filter(*cloud_plane);
    
    
    
    
    
    
    return coefficients;
}

// Define the size of the space and voxels
const float VOXEL_SIZE = 3.0f;
const int GRID_X = 30; // 300 / 3
const int GRID_Y = 20;  // 99 / 3
const int GRID_Z = 10;  // 99 / 3
//*/

/*
// Define the size of the space and voxels
const float VOXEL_SIZE = 2.0f;
const int GRID_X = 50;  // 100 / 2
const int GRID_Y = 50;  // 100 / 2
const int GRID_Z = 10;  // 20  / 2
//*/

// Helper function to calculate voxel index
inline int getVoxelIndex(int x, int y, int z) {
    return x + y * GRID_X + z * GRID_X * GRID_Y;
}

// Function to check if the points within a voxel form a plane
bool isVoxelPlanar(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double threshold, pcl::ModelCoefficients::Ptr coefficients_of_plane ) {
    if (cloud->points.size() < 3) {
        return false;
    }

    cout<<"===[isVoxelPlanar]===="<<endl;
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(threshold);

    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);


    if (inliers->indices.empty()) {
        coefficients_of_plane = NULL;
        return false;
    }

    cout<<"=====[inliers->indices.size()]===="<<endl;
    cout<<inliers->indices.size()<<endl;
    coefficients_of_plane->values[3] = coefficients->values[3];
    coefficients_of_plane->values[0] = coefficients->values[0];
    coefficients_of_plane->values[1] = coefficients->values[1];
    coefficients_of_plane->values[2] = coefficients->values[2];
    
    cout<<"=========[coefficients_of_plane]======="<<endl;
    cout<<coefficients_of_plane->values[3]<<endl;
    cout<<coefficients_of_plane->values[0]<<endl;
    cout<<coefficients_of_plane->values[1]<<endl;
    cout<<coefficients_of_plane->values[2]<<endl;
    //cout<<"====================================================[plane_normal][3]============================================================"<<endl;
    return true;
}

// Function to compute the distance from a point to a plane
double pointToPlaneDistance(const Eigen::Vector3d& point, const Eigen::Vector3d& plane_normal, const Eigen::Vector3d& plane_center) {
    return std::fabs(plane_normal.dot(point - plane_center));
}


std::map<int, pcl::ModelCoefficients::Ptr> voxel_plane_coefficient;
    
    
    
std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr>   make_voxel_map( pcl::PointCloud<pcl::PointXYZ>::Ptr cloud )
{


    std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr> voxel_map;

    //cout<<"====================================================[0]============================================================"<<endl;
    //cout<<cloud->size()<<endl;
    
    // Assign points to voxels
    for ( int i = 0; i < cloud->size(); i++ ) {
        int x = static_cast<int>(cloud->points[i].x / VOXEL_SIZE);
        int y = static_cast<int>(cloud->points[i].y / VOXEL_SIZE);
        int z = static_cast<int>(cloud->points[i].z / VOXEL_SIZE);

        int voxel_index = getVoxelIndex(x, y, z);

        if (voxel_map.find(voxel_index) == voxel_map.end()) {
        	voxel_map[voxel_index] = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        }
        voxel_map[voxel_index]->points.push_back(cloud->points[i]);
        
        //cout<<"====================================================[1]============================================================"<<endl;
        if (voxel_plane_coefficient.find(voxel_index) == voxel_plane_coefficient.end()) {
        	voxel_plane_coefficient[voxel_index] = pcl::ModelCoefficients::Ptr(new pcl::ModelCoefficients);
        }
        
        //voxel_plane_coefficient[voxel_index] = pcl::ModelCoefficients::Ptr(new pcl::ModelCoefficients);
	// Ensure the vector has the correct size for plane coefficients
	voxel_plane_coefficient[voxel_index]->values.resize(4);

	//cout<<"====================================================[2]============================================================"<<endl;
	//cout<< voxel_plane_coefficient[voxel_index]<<endl;
	//cout<< voxel_index <<endl;
	voxel_plane_coefficient[voxel_index]->values[3] = -100;
	voxel_plane_coefficient[voxel_index]->values[0] = -100;
	voxel_plane_coefficient[voxel_index]->values[1] = -100;
	voxel_plane_coefficient[voxel_index]->values[2] = -100;
        
        
    }


    return voxel_map;
} 


void make_voxel_plane_coefficient( std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr> voxel_map )
{

    pcl::ModelCoefficients::Ptr coefficients_of_plane_of_voxel(new pcl::ModelCoefficients);
    coefficients_of_plane_of_voxel->values.resize(4);

    //cout<<"====================================================[2]============================================================"<<endl;
    coefficients_of_plane_of_voxel->values[3] = -100;
    coefficients_of_plane_of_voxel->values[0] = -100;
    coefficients_of_plane_of_voxel->values[1] = -100;
    coefficients_of_plane_of_voxel->values[2] = -100;

    // Check each voxel if it has a planar surface
    for (const auto& voxel_pair : voxel_map) {
    
	cout<<"===id_of_voxel==="<<endl;
	cout<<voxel_pair.first<<endl;
    	
    	
        if (isVoxelPlanar( voxel_pair.second, 0.05, coefficients_of_plane_of_voxel )) {

            //cout<<"====================================================voxel_plane_coefficient[1]============================================================"<<endl;
            //cout<<voxel_pair.first<<endl;
            //cout<<coefficients_of_plane_of_voxel->values[3]<<endl;
            //cout<<coefficients_of_plane_of_voxel->values[0]<<endl;
            //cout<<coefficients_of_plane_of_voxel->values[1]<<endl;
            //cout<<coefficients_of_plane_of_voxel->values[2]<<endl;
            //cout<<"====================================================voxel_plane_coefficient[2]============================================================"<<endl;
            
            if( coefficients_of_plane_of_voxel != NULL )
            {
            	//cout<<"====================================================voxel_plane_coefficient[3]============================================================"<<endl;
            	voxel_plane_coefficient[voxel_pair.first]->values[3] = coefficients_of_plane_of_voxel->values[3];
            	voxel_plane_coefficient[voxel_pair.first]->values[0] = coefficients_of_plane_of_voxel->values[0];
            	voxel_plane_coefficient[voxel_pair.first]->values[1] = coefficients_of_plane_of_voxel->values[1];
            	voxel_plane_coefficient[voxel_pair.first]->values[2] = coefficients_of_plane_of_voxel->values[2];
            	//cout<<"====================================================voxel_plane_coefficient[4]============================================================"<<endl;
            	
            }
            else
            {
            	//cout<<"====================================================NULL============================================================"<<endl;
            	voxel_plane_coefficient[voxel_pair.first] = NULL;
            }

        }
        
        //cout<<"====================================================voxel_plane_coefficient[5]============================================================"<<endl;
    }


}
    
// Main function to process the point cloud
void processPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {


    std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr> voxel_map = make_voxel_map(cloud);
	
    /*
    cout<<"====================================================[0]============================================================"<<endl;
    cout<<cloud->size()<<endl;
    
    
    // Assign points to voxels
    for ( int i = 0; i < cloud->size(); i++ ) {
        int x = static_cast<int>(cloud->points[i].x / VOXEL_SIZE);
        int y = static_cast<int>(cloud->points[i].y / VOXEL_SIZE);
        int z = static_cast<int>(cloud->points[i].z / VOXEL_SIZE);

        int voxel_index = getVoxelIndex(x, y, z);

        if (voxel_map.find(voxel_index) == voxel_map.end()) {
        	voxel_map[voxel_index] = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        }
        voxel_map[voxel_index]->points.push_back(cloud->points[i]);
        
        //cout<<"====================================================[1]============================================================"<<endl;
        if (voxel_plane_coefficient.find(voxel_index) == voxel_plane_coefficient.end()) {
        	voxel_plane_coefficient[voxel_index] = pcl::ModelCoefficients::Ptr(new pcl::ModelCoefficients);
        }
        
        //voxel_plane_coefficient[voxel_index] = pcl::ModelCoefficients::Ptr(new pcl::ModelCoefficients);
	// Ensure the vector has the correct size for plane coefficients
	voxel_plane_coefficient[voxel_index]->values.resize(4);

	cout<<"====================================================[2]============================================================"<<endl;
	//cout<< voxel_plane_coefficient[voxel_index]<<endl;
	cout<< voxel_index <<endl;
	voxel_plane_coefficient[voxel_index]->values[3] = -100;
	voxel_plane_coefficient[voxel_index]->values[0] = -100;
	voxel_plane_coefficient[voxel_index]->values[1] = -100;
	voxel_plane_coefficient[voxel_index]->values[2] = -100;
        
        
    }
    //*/
    cout<<"===[voxel_map.size()]==="<<endl;

    cout<<voxel_map.size()<<endl;
    cout<<voxel_plane_coefficient.size()<<endl;
    
    make_voxel_plane_coefficient(voxel_map);
    
    /*
    pcl::ModelCoefficients::Ptr coefficients_of_plane_of_voxel(new pcl::ModelCoefficients);
    coefficients_of_plane_of_voxel->values.resize(4);

    cout<<"====================================================[2]============================================================"<<endl;
    coefficients_of_plane_of_voxel->values[3] = -100;
    coefficients_of_plane_of_voxel->values[0] = -100;
    coefficients_of_plane_of_voxel->values[1] = -100;
    coefficients_of_plane_of_voxel->values[2] = -100;

    // Check each voxel if it has a planar surface
    for (const auto& voxel_pair : voxel_map) {
    
	cout<<"====================================================voxel_pair.first[0]============================================================"<<endl;
	cout<<voxel_pair.first<<endl;
    	
        if (isVoxelPlanar( voxel_pair.second, 0.1, coefficients_of_plane_of_voxel )) {

            cout<<"====================================================voxel_plane_coefficient[1]============================================================"<<endl;
            cout<<voxel_pair.first<<endl;
            cout<<coefficients_of_plane_of_voxel->values[3]<<endl;
            cout<<coefficients_of_plane_of_voxel->values[0]<<endl;
            cout<<coefficients_of_plane_of_voxel->values[1]<<endl;
            cout<<coefficients_of_plane_of_voxel->values[2]<<endl;
            
            if( coefficients_of_plane_of_voxel != NULL )
            {
            	voxel_plane_coefficient[voxel_pair.first]->values[3] = coefficients_of_plane_of_voxel->values[3];
            	voxel_plane_coefficient[voxel_pair.first]->values[0] = coefficients_of_plane_of_voxel->values[0];
            	voxel_plane_coefficient[voxel_pair.first]->values[1] = coefficients_of_plane_of_voxel->values[1];
            	voxel_plane_coefficient[voxel_pair.first]->values[2] = coefficients_of_plane_of_voxel->values[2];
            	
            	cout<<"====================================================coefficients_of_plane_of_voxel[voxel_pair.first]->values[1]============================================================"<<endl;
            	cout<<"voxel_pair.first="<<voxel_pair.first<<endl;
            	cout<<coefficients_of_plane_of_voxel->values[3]<<endl;
		cout<<coefficients_of_plane_of_voxel->values[0]<<endl;
		cout<<coefficients_of_plane_of_voxel->values[1]<<endl;
		cout<<coefficients_of_plane_of_voxel->values[2]<<endl;
            
		Eigen::Vector3d point(next_frame_cloud->points[0].x, next_frame_cloud->points[0].y, next_frame_cloud->points[0].z);

		int x = static_cast<int>(point.x() / VOXEL_SIZE);
		int y = static_cast<int>(point.y() / VOXEL_SIZE);
		int z = static_cast<int>(point.z() / VOXEL_SIZE);

		int voxel_index = getVoxelIndex(x, y, z);

            	cout<<"====================================================voxel_plane_coefficient[voxel_pair.first]->values[2]============================================================"<<endl;
            	cout<<"voxel_index="<<voxel_index<<endl;
            	cout<<voxel_plane_coefficient[voxel_index]->values[3]<<endl;
		cout<<voxel_plane_coefficient[voxel_index]->values[0]<<endl;
		cout<<voxel_plane_coefficient[voxel_index]->values[1]<<endl;
		cout<<voxel_plane_coefficient[voxel_index]->values[2]<<endl;

            
            	
            }
            else
            {
            	cout<<"====================================================NULL============================================================"<<endl;
            	voxel_plane_coefficient[voxel_pair.first] = NULL;
            }
            

            //cout<<voxel_plane_coefficient[voxel_pair.first]<<endl;
        }
    }
    //*/

/*
    // Process the next frame
    for (size_t i = 0; i < next_frame_cloud->points.size(); ++i) {
        Eigen::Vector3d point(next_frame_cloud->points[i].x, next_frame_cloud->points[i].y, next_frame_cloud->points[i].z);

        int x = static_cast<int>(point.x() / VOXEL_SIZE);
        int y = static_cast<int>(point.y() / VOXEL_SIZE);
        int z = static_cast<int>(point.z() / VOXEL_SIZE);

        int voxel_index = getVoxelIndex(x, y, z);

        //if (voxel_planes.find(voxel_index) != voxel_planes.end()) 
        //{
	const auto& plane_normal = voxel_planes[voxel_index].first;
	const auto& plane_center = voxel_planes[voxel_index].second;

	double distance = pointToPlaneDistance(point, plane_normal, plane_center);
	cout<<"========================================distance================================================="<<endl;
	std::cout << "Point " << i << " distance to plane: " << distance << std::endl;
	cout<<"x="<<x<<","<<point.x()<<endl;
	cout<<"y="<<y<<","<<point.y()<<endl;
	cout<<"z="<<z<<","<<point.z()<<endl;
	cout<<"voxel_index="<<voxel_index<<endl;
	cout<<"plane_normal="<<plane_normal<<endl;
	cout<<"plane_center="<<plane_center<<endl;
	
	if( coefficients_of_plane != NULL )
	{
		cout<<coefficients_of_plane->values[3]<<endl;
		cout<<coefficients_of_plane->values[0]<<endl;
		cout<<coefficients_of_plane->values[1]<<endl;
		cout<<coefficients_of_plane->values[2]<<endl;
	}
	
	
        //}
    }
    //*/
    
    //return coefficients_of_plane_of_voxel;
}


pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_store(new pcl::PointCloud<pcl::PointXYZ>);


void publish_cloud_store(const ros::Publisher & pub_cloud_store)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(false ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI laserCloudWorld;
        
        cloud_store_intensity->clear();

        
        for (int i = 0; i < size; i++)
        {
            PointType const * const p = &laserCloudFullRes->points[i];
            if(p->intensity < 80 ){
                continue;
            }
            
//            if (p->x < 0 and p->x > -4
//                    and p->y < 1.5 and p->y > -1.5
//                            and p->z < 2 and p->z > -1) {
//                continue;
//            }
            PointType p_world;

            RGBpointBodyToWorld(p, &p_world);
            

            pcl::PointXYZ pointXYZ;
            // Copy the XYZ coordinates from pointXYZI to pointXYZ
            pointXYZ.x = p_world.x;
            pointXYZ.y = p_world.y;
            pointXYZ.z = p_world.z;
            // Add the point to the new cloud
            cloud_store->points.push_back(pointXYZ);
            
            cloud_store_intensity->points.push_back(pointXYZ);

        }

	// Compute smoothness values
	//int k_neighbors = 10; // Number of neighbors to consider
	//std::vector<float> smoothness = computeSmoothness( cloud_store, k_neighbors, cloud_store_edge );
        
	// Get corner features from point cloud
	// double threshold = 0.5;
	// std::vector<CornerFeature> cornerFeatures = dete ctCornerFeatures(cloud_store, threshold, cloud_store_edge);

	//double radius = 3.0;
	//coefficients_of_plane = findPlaneInPointCloud(cloud_store, radius);

        
        //cout<<"====================cloud_store=========================="<<endl;
        //cout<< cloud_store->size()<<endl;
        
	// make the point cloud less point, speed up the CPU
	//if( cloud_store->size() > -1 )
	{
		//cloud_store->clear();
	}
        
        //cout<<"====================cloud_store[1]=========================="<<endl;

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*map_final, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pub_cloud_store.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
        
        //cloud_store->clear();
        //cloud_store_intensity->clear();
        
        //cout<<"====================cloud_store[2]=========================="<<endl;
    }

}


void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
//    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));
    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&laserCloudFullRes->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;

}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);// ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body" ) );

    static tf::TransformBroadcaster br_world;
    transform.setOrigin(tf::Vector3(0, 0, 0));
    q.setValue(p_imu->Initial_R_wrt_G.x(), p_imu->Initial_R_wrt_G.y(), p_imu->Initial_R_wrt_G.z(), p_imu->Initial_R_wrt_G.w());
    transform.setRotation(q);
    br_world.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "world", "camera_init"));
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 1 == 0)
    {
        path.header.stamp = msg_body_pose.header.stamp;
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

void transformLidar(const state_ikfom &state_point, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud)
{
    trans_cloud->clear();
    for (size_t i = 0; i < input_cloud->size(); i++) {
        pcl::PointXYZINormal p_c = input_cloud->points[i];
        Eigen::Vector3d p_lidar(p_c.x, p_c.y, p_c.z);
        // HACK we need to specify p_body as a V3D type!!!
        V3D p_body = state_point.rot * (state_point.offset_R_L_I * p_lidar + state_point.offset_T_L_I) + state_point.pos;
        PointType pi;
        pi.x = p_body(0);
        pi.y = p_body(1);
        pi.z = p_body(2);
        pi.intensity = p_c.intensity;
        trans_cloud->points.push_back(pi);
    }
}

//M3D transformLiDARCovToWorld(Eigen::Vector3d &p_lidar, const esekfom::esekf<state_ikfom, 12, input_ikfom>& kf, const Eigen::Matrix3d& COV_lidar)
//{
//    double match_start = omp_get_wtime();
//    // FIXME 这里首先假定LiDAR系和body是重叠的 没有外参
//    M3D point_crossmat;
//    point_crossmat << SKEW_SYM_MATRX(p_lidar);
//    // 注意这里Rt的cov顺序
//    M3D rot_var = kf.get_P().block<3, 3>(3, 3);
//    M3D t_var = kf.get_P().block<3, 3>(0, 0);
//    auto state = kf.get_x();
//
//    // Eq. (3)
//    M3D COV_world =
//            state.rot * COV_lidar * state.rot.conjugate()
//            + state.rot * (-point_crossmat) * rot_var * (-point_crossmat).transpose()  * state.rot.conjugate()
//            + t_var;
//    return COV_world;
//    // Voxel map 真实实现
////    M3D cov_world = R_body * COV_lidar * R_body.conjugate() +
////          (-point_crossmat) * rot_var * (-point_crossmat).transpose() + t_var;
//
//}

M3D transformLiDARCovToWorld(Eigen::Vector3d &p_lidar, const esekfom::esekf<state_ikfom, 12, input_ikfom>& kf, const Eigen::Matrix3d& COV_lidar)
{
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(p_lidar);
    auto state = kf.get_x();

    // lidar到body的方差传播
    // 注意外参的var是先rot 后pos
    M3D il_rot_var = kf.get_P().block<3, 3>(6, 6);
    M3D il_t_var = kf.get_P().block<3, 3>(9, 9);

    M3D COV_body =
            state.offset_R_L_I * COV_lidar * state.offset_R_L_I.conjugate()
            + state.offset_R_L_I * (-point_crossmat) * il_rot_var * (-point_crossmat).transpose() * state.offset_R_L_I.conjugate()
            + il_t_var;

    // body的坐标
    V3D p_body = state.offset_R_L_I * p_lidar + state.offset_T_L_I;

    // body到world的方差传播
    // 注意pose的var是先pos 后rot
    point_crossmat << SKEW_SYM_MATRX(p_body);
    M3D rot_var = kf.get_P().block<3, 3>(3, 3);
    M3D t_var = kf.get_P().block<3, 3>(0, 0);

    // Eq. (3)
    M3D COV_world =
        state.rot * COV_body * state.rot.conjugate()
        + state.rot * (-point_crossmat) * rot_var * (-point_crossmat).transpose()  * state.rot.conjugate()
        + t_var;

    return COV_world;
    // Voxel map 真实实现
//    M3D cov_world = R_body * COV_lidar * R_body.conjugate() +
//          (-point_crossmat) * rot_var * (-point_crossmat).transpose() + t_var;

}

void observation_model_share(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
//    laserCloudOri->clear();
//    corr_normvect->clear();
    feats_with_correspondence->clear();
    total_residual = 0.0;

    // =================================================================================================================
    // 用当前迭代轮最新的位姿估计值 将点云转换到world地图系
    vector<pointWithCov> pv_list;
    PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI);
    // FIXME stupid mistake 这里应该用迭代的最新线性化点
    // FIXME stupid mistake 这里应该用迭代的最新线性化点
//    transformLidar(state_point, feats_down_body, world_lidar);
    transformLidar(s, feats_down_body, world_lidar);
    pv_list.resize(feats_down_body->size());
    for (size_t i = 0; i < feats_down_body->size(); i++) {
        // 保存body系和world系坐标
        pointWithCov pv;
        pv.point << feats_down_body->points[i].x, feats_down_body->points[i].y, feats_down_body->points[i].z;
        pv.point_world << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
        // 计算lidar点的cov
        // 注意这个在每次迭代时是存在重复计算的 因为lidar系的点云covariance是不变的
        // M3D cov_lidar = calcBodyCov(pv.point, ranging_cov, angle_cov);
        M3D cov_lidar = var_down_body[i];
        // 将body系的var转换到world系
        M3D cov_world = transformLiDARCovToWorld(pv.point, kf, cov_lidar);
        pv.cov = cov_world;
        pv.cov_lidar = cov_lidar;
        pv_list[i] = pv;
    }

    // ===============================================================================================================
    // 查找最近点 并构建residual
    double match_start = omp_get_wtime();
    std::vector<ptpl> ptpl_list;
    std::vector<V3D> non_match_list;
    BuildResidualListOMP(voxel_map, max_voxel_size, 30.0, max_layer, pv_list,
                         ptpl_list, non_match_list);
    double match_end = omp_get_wtime();
    // std::printf("Match Time: %f\n", match_end - match_start);

    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    // 根据匹配结果 设置H和R的维度
    // h_x是观测值对状态量的导数 TODO 为什么不加上状态量对状态量误差的导数？？？？像quaternion那本书？
    
    //cloud_store_intensity->clear();
    
    int other_least_square_size = cloud_store_intensity->size();
    
    //cout<<"===============cloud_store_intensity->size()================"<<endl;
    //cout<<cloud_store_intensity->size()<<endl;
    
    int counter_effective_point = 0;
    
    vector<Eigen::Vector3d> point_in_voxel;
    
    if ( cloud_store->size() > 5 && false )
    { 
    
    	processPointCloud(cloud_store);
    
    	for( int i = 0; i < cloud_store_intensity->size() ; i++ )
    	{
    		Eigen::Vector3d point(cloud_store_intensity->points[0].x, cloud_store_intensity->points[0].y, cloud_store_intensity->points[0].z);

		int x = static_cast<int>(point.x() / VOXEL_SIZE);
		int y = static_cast<int>(point.y() / VOXEL_SIZE);
		int z = static_cast<int>(point.z() / VOXEL_SIZE);

		cout<<point.x()<<endl;
		cout<<point.y()<<endl;
		cout<<point.z()<<endl;


		int voxel_index = getVoxelIndex(x, y, z);

            	cout<<"====================================================voxel_plane_coefficient_after_getVoxelIndex(x, y, z);============================================================"<<endl;
            	cout<<"voxel_index="<<voxel_index<<endl;
            	if( voxel_plane_coefficient[voxel_index] != NULL )
            	{
            		if( voxel_plane_coefficient[voxel_index]->values[3] != -100 && abs( voxel_plane_coefficient[voxel_index]->values[2] ) < 0.3 )
            		{
            			point_in_voxel.push_back(point);
            			cout<<point_in_voxel[counter_effective_point].x()<<endl;
            			cout<<point_in_voxel[counter_effective_point].y()<<endl;
            			cout<<point_in_voxel[counter_effective_point].z()<<endl;
            			
		    		counter_effective_point++;
			    	cout<<voxel_plane_coefficient[voxel_index]->values[3]<<endl;
				cout<<voxel_plane_coefficient[voxel_index]->values[0]<<endl;
				cout<<voxel_plane_coefficient[voxel_index]->values[1]<<endl;
				cout<<voxel_plane_coefficient[voxel_index]->values[2]<<endl;
				
			}
    		}
    	}
    }
    
    //int other_least_square_size = 100;
    
    
    other_least_square_size = point_in_voxel.size();
    
    //cout<<"===========point_in_voxel.size()============"<<endl;
    //cout<<point_in_voxel.size()<<endl;
    
    effct_feat_num = ptpl_list.size() + other_least_square_size;
    
    //effct_feat_num = 100 + other_least_square_size;
    
    //effct_feat_num = 1000;
    	
    //cout<<"===============ptpl_list.size()================"<<endl;
    //cout<<ptpl_list.size()<<endl;    
    
    if ( ptpl_list.size() < 20){
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23 因为点面距离只和位姿 外参有关 对其他状态量的导数都是0
    ekfom_data.h.resize(effct_feat_num);
    ekfom_data.R.resize(effct_feat_num, 1); // 把R作为向量 用的时候转换成diag
//    ekfom_data.R.setZero();
//    printf("isDiag: %d  R norm: %f\n", ekfom_data.R.isDiagonal(1e-10), ekfom_data.R.norm());

//    // 求每个匹配点到平面的距离
//    for (int i = 0; i < ptpl_list.size(); i++) {
//        // 取出匹配到的world系norm
//        PointType pl;
//        pl.x = ptpl_list[i].normal(0);
//        pl.y = ptpl_list[i].normal(1);
//        pl.z = ptpl_list[i].normal(2);
//
//        // 将原始点云转换到world系
//        V3D pi_world(s.rot * (s.offset_R_L_I * ptpl_list[i].point + s.offset_T_L_I) + s.pos);
//
//        // 求点面距离
//        float dis = pi_world.x() * pl.x + pi_world.y() * pl.y + pi_world.z() * pl.z + ptpl_list[i].d;
//        pl.intensity = dis;
////        std::printf("%.5f   %.5f\n", dis, ptpl_list[i].pd2);
////        std::printf("%.5f  %.5f\n", pi_world.x(), ptpl_list[i].point_world.x());
////        std::printf("%.5f  %.5f\n", pi_world.y(), ptpl_list[i].point_world.y());
//
//        PointType pi_body;
//        pi_body.x = ptpl_list[i].point(0);
//        pi_body.y = ptpl_list[i].point(1);
//        pi_body.z = ptpl_list[i].point(2);
//        laserCloudOri->push_back(pi_body);
//        corr_normvect->push_back(pl);
//        // for visualization
//        feats_with_correspondence->push_back(pi_body);
//
//        total_residual += fabs(dis);
//    }
//    assert(laserCloudOri->size() == effct_feat_num && corr_normvect->size() == effct_feat_num);
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < effct_feat_num - other_least_square_size; i++)
    {

//        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(ptpl_list[i].point);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
//        const PointType &norm_p = corr_normvect->points[i];
//        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);
        V3D norm_vec(ptpl_list[i].normal);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            // ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_vec.x(), norm_vec.y(), norm_vec.z(), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            // ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            ekfom_data.h_x.block<1, 12>(i,0) << norm_vec.x(), norm_vec.y(), norm_vec.z(), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }
        
        //cout<<"=============h_x===================="<<endl;
        //cout<<ekfom_data.h_x.block<1, 12>(i,0)<<endl;
        

/*        
        if (  ptpl_list[i].d < 0 )
        {
		norm_vec.x() = -1 * norm_vec.x();
		norm_vec.y() = -1 * norm_vec.y();
		norm_vec.z() = -1 * norm_vec.z();
		ptpl_list[i].d = -1 * ptpl_list[i].d;
        } 
//*/        

        /*** Measuremnt: distance to the closest surface/corner ***/
//        ekfom_data.h(i) = -norm_p.intensity;
        float pd2 = norm_vec.x() * ptpl_list[i].point_world.x()
                + norm_vec.y() * ptpl_list[i].point_world.y()
                + norm_vec.z() * ptpl_list[i].point_world.z()
                + ptpl_list[i].d;
        ekfom_data.h(i) = -pd2;
        
        
        //cout<<"=============h===================="<<endl;
        //cout<<ekfom_data.h(i)<<endl;

        /*** Covariance ***/
//        // norm_p中存了匹配的平面法向 还有点面距离
//        V3D point_world = s.rot * (s.offset_R_L_I * ptpl_list[i].point + s.offset_T_L_I) + s.pos;
//        // /*** get the normal vector of closest surface/corner ***/
//        Eigen::Matrix<double, 1, 6> J_nq;
//        J_nq.block<1, 3>(0, 0) = point_world - ptpl_list[i].center;
//        J_nq.block<1, 3>(0, 3) = -ptpl_list[i].normal;
//        double sigma_l = J_nq * ptpl_list[i].plane_cov * J_nq.transpose();
//
//        M3D cov_lidar = calcBodyCov(ptpl_list[i].point, ranging_cov, angle_cov);
//        M3D R_cov_Rt = s.rot * cov_lidar * s.rot.conjugate();
//        // HACK 1. 因为是标量 所以求逆直接用1除
//        // HACK 2. 不同分量的方差用加法来合成 因为公式(12)中的Sigma是对角阵，逐元素运算之后就是对角线上的项目相加
//        double R_inv = 1.0 / (sigma_l + norm_vec.transpose() * R_cov_Rt * norm_vec);

        // norm_p中存了匹配的平面法向 还有点面距离
        // V3D point_world = s.rot * (s.offset_R_L_I * ptpl_list[i].point + s.offset_T_L_I) + s.pos;
        V3D point_world = ptpl_list[i].point_world;
        // /*** get the normal vector of closest surface/corner ***/
        Eigen::Matrix<double, 1, 6> J_nq;
        J_nq.block<1, 3>(0, 0) = point_world - ptpl_list[i].center;
        J_nq.block<1, 3>(0, 3) = -ptpl_list[i].normal;
        double sigma_l = J_nq * ptpl_list[i].plane_cov * J_nq.transpose();


	float dis_to_plane = fabs(ptpl_list[i].normal(0) * ptpl_list[i].point_world(0) + ptpl_list[i].normal(1) * ptpl_list[i].point_world(1) + ptpl_list[i].normal(2) * ptpl_list[i].point_world(2) + ptpl_list[i].d);


        // M3D cov_lidar = calcBodyCov(ptpl_list[i].point, ranging_cov, angle_cov);
        M3D cov_lidar = ptpl_list[i].cov_lidar;
        M3D R_cov_Rt = s.rot * s.offset_R_L_I * cov_lidar * s.offset_R_L_I.conjugate() * s.rot.conjugate();
        // HACK 1. 因为是标量 所以求逆直接用1除
        // HACK 2. 不同分量的方差用加法来合成 因为公式(12)中的Sigma是对角阵，逐元素运算之后就是对角线上的项目相加
        // 1  		1						[NOT], LOW SPEED, FLY AWAY, SMALL FLY, and NOT ACCURATE
        // 0.1		9.87144					[LITTLE] STABLE and SPEED UP, FLY AWAY
        // 0.01	85.8414					[MIDDLE] STABLE, GOOD PERFORMANCE, BIG MOVEMENT, SPEED UP FLY AWAY				
        // 0.001	211.672	185.779 	328.87		[MIDDLE LITTLE] STABLE and SPEED
        // 0.0001	832.246         376.594	255.439	[MIDDLE] STABLE and SPEED

	double R_inv = 1.0 / ( 0.9 * sigma_l + 0.1 * dis_to_plane * dis_to_plane  + norm_vec.transpose() * R_cov_Rt * norm_vec + 0.001 ) ;
        //double R_inv = 1.0 / (sigma_l + norm_vec.transpose() * R_cov_Rt * norm_vec + 0.001 ) ;
        //double R_inv = 1.0 / (sigma_l + norm_vec.transpose() * R_cov_Rt * norm_vec  ) ;

        // 计算测量方差R并赋值 目前暂时使用固定值
        // ekfom_data.R(i) = 1.0 / LASER_POINT_COV;
        
        if( abs( norm_vec.x() ) > 0.8  || abs( norm_vec.y() ) > 0.8 || abs( norm_vec.z() ) > 0.8  )
        {
        	R_inv = 100;  //* R_inv;
        	ekfom_data.R(i) = R_inv;
        }
        else
        {
        	R_inv = 0.0 * R_inv;
        	ekfom_data.R(i) = R_inv;
        }
        //cout<<"=============R===================="<<endl;
        //cout<<ekfom_data.R(i)<<endl;
        
    }
    
    
    //double radius = 1.0;
    //coefficients_of_plane = findPlaneInPointCloud(cloud_store, radius, cloud_plane);
    
    //if( coefficients_of_plane != NULL )
    //{
    //	cout<<"==============coefficients_of_plane[0]=================="<<endl;
    //	cout<<"coefficients:"<<coefficients_of_plane->values[0]<<","<<coefficients_of_plane->values[1]<<","<<coefficients_of_plane->values[2]<<","<<coefficients_of_plane->values[3]<<endl;

	    
	

	if ( cloud_store->size() > 5 && point_in_voxel.size() > 0 )
	{

		//coefficients_of_plane = findPlaneInPointCloud(cloud_plane, radius, cloud_plane);
		//cout<<"==============coefficients_of_plane[1]=================="<<endl;
		//cout<<"coefficients:"<<coefficients_of_plane->values[0]<<","<<coefficients_of_plane->values[1]<<","<<coefficients_of_plane->values[2]<<","<<coefficients_of_plane->values[3]<<endl;
		
		//pcl::ModelCoefficients::Ptr coefficients_of_plane_2(new pcl::ModelCoefficients);
		//processPointCloud(cloud_store);
		//cout<<"==============coefficients_of_plane[2]=================="<<endl;
		//cout<<"coefficients:"<<coefficients_of_plane_2->values[0]<<","<<coefficients_of_plane_2->values[1]<<","<<coefficients_of_plane_2->values[2]<<","<<coefficients_of_plane_2->values[3]<<endl;

		//coefficients_of_plane = coefficients_of_plane_2;

		//if( coefficients_of_plane->values[3] < 0 )
		//{
		//	coefficients_of_plane->values[3] = -1 * coefficients_of_plane->values[3];
		//	coefficients_of_plane->values[0] = -1 * coefficients_of_plane->values[0];
		//	coefficients_of_plane->values[1] = -1 * coefficients_of_plane->values[1];
		//	coefficients_of_plane->values[2] = -1 * coefficients_of_plane->values[2];
		//	cout<<"coefficients:"<<coefficients_of_plane->values[0]<<","<<coefficients_of_plane->values[1]<<","<<coefficients_of_plane->values[2]<<","<<coefficients_of_plane->values[3]<<endl;
		//}
		
		
		for (int i = 0; i < point_in_voxel.size(); i++)
		{
			//        const PointType &norm_p = corr_normvect->points[i];
			//        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);
			
			
			//	input is 
			//	1. cloud_store_intensity->points[i]	the point
			//	2. coefficients_of_plane->values, 	the plane
			
			
			
			V3D point_this( point_in_voxel[i].x(), point_in_voxel[i].y(), point_in_voxel[i].z()  );

			//Eigen::Vector3d point(cloud_store_intensity->points[0].x, cloud_store_intensity->points[0].y, cloud_store_intensity->points[0].z);

			int x = static_cast<int>(point_this.x() / VOXEL_SIZE);
			int y = static_cast<int>(point_this.y() / VOXEL_SIZE);
			int z = static_cast<int>(point_this.z() / VOXEL_SIZE);

			cout<<point_this.x()<<endl;
			cout<<point_this.y()<<endl;
			cout<<point_this.z()<<endl;


			int voxel_index = getVoxelIndex(x, y, z);

			cout<<"====================================================coefficients_of_plane_getVoxelIndex(x, y, z);============================================================"<<endl;
			cout<<"voxel_index="<<voxel_index<<endl;
			
			cout<<voxel_plane_coefficient[voxel_index]->values[3]<<endl;
			cout<<voxel_plane_coefficient[voxel_index]->values[0]<<endl;
			cout<<voxel_plane_coefficient[voxel_index]->values[1]<<endl;
			cout<<voxel_plane_coefficient[voxel_index]->values[2]<<endl;
			
			coefficients_of_plane->values.resize(4);

			coefficients_of_plane->values[3] = voxel_plane_coefficient[voxel_index]->values[3];
			coefficients_of_plane->values[0] = voxel_plane_coefficient[voxel_index]->values[0];
			coefficients_of_plane->values[1] = voxel_plane_coefficient[voxel_index]->values[1];
			coefficients_of_plane->values[2] = voxel_plane_coefficient[voxel_index]->values[2];
			
			if( coefficients_of_plane->values[3] < 0 )
			{
				coefficients_of_plane->values[3] = -1 * coefficients_of_plane->values[3];
				coefficients_of_plane->values[0] = -1 * coefficients_of_plane->values[0];
				coefficients_of_plane->values[1] = -1 * coefficients_of_plane->values[1];
				coefficients_of_plane->values[2] = -1 * coefficients_of_plane->values[2];
				cout<<"coefficients:"<<coefficients_of_plane->values[0]<<","<<coefficients_of_plane->values[1]<<","<<coefficients_of_plane->values[2]<<","<<coefficients_of_plane->values[3]<<endl;
			}
			
			cout<<"=======================point_this==========================="<<i<<endl;
			cout<<point_this<<endl;

			M3D point_crossmat;
			point_crossmat<<SKEW_SYM_MATRX(point_this);


			cout<<"=======================point_this[1]==========================="<<effct_feat_num - other_least_square_size + i <<endl;
			cout<<"coefficients:"<<coefficients_of_plane->values[0]<<","<<coefficients_of_plane->values[1]<<","<<coefficients_of_plane->values[2]<<","<<coefficients_of_plane->values[3]<<endl;

			V3D norm_vec( coefficients_of_plane->values[0], coefficients_of_plane->values[1], coefficients_of_plane->values[2] );

			cout<<"=======================norm_vec==========================="<<i<<endl;
			cout<<norm_vec<<endl;

			V3D C(s.rot.conjugate() * norm_vec);
			V3D A(point_crossmat * C);

			//if (extrinsic_est_en)
			//{
			//    V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
			// ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
			//    ekfom_data.h_x.block<1, 12>(i,0) << norm_vec.x(), norm_vec.y(), norm_vec.z(), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
			//}
			//else
			{
			// ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
				ekfom_data.h_x.block<1, 12>( effct_feat_num - other_least_square_size + i, 0 ) << norm_vec.x(), norm_vec.y(), norm_vec.z(), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
			}

			float pd2 = 	norm_vec.x() * point_this.x()
			+ 		norm_vec.y() * point_this.y()
			+ 		norm_vec.z() * point_this.z()
			+ 		coefficients_of_plane->values[3] ;

			cout<<"=======================-pd2==========================="<<i<<endl;
			cout<<-pd2<<endl;

			
			ekfom_data.h(effct_feat_num - other_least_square_size + i) = -pd2;

			
			
						
			if( abs(-pd2) > 1.0 )
			{
				ekfom_data.R(effct_feat_num - other_least_square_size + i) = 1; 
			}
			else
			{
				ekfom_data.R(effct_feat_num - other_least_square_size + i) = 100;
			}
			
			

			//ekfom_data.h_x.block<1, 12>(i,0) << -0.211307, 0.0581584,  0.975688,  -1.50686,  -8.75256,  0.175648,  -1.50689,  -8.75258,   0.17432, -0.210924, 0.0558978,  0.975903;
			//ekfom_data.h(i) = -1.00109248;
			//ekfom_data.R(i) = 0.027647;
		}

	}
	else
	{
		
		for (int i = 0; i < point_in_voxel.size(); i++)
		{
			ekfom_data.h_x.block<1, 12>(effct_feat_num - other_least_square_size + i,0) << -0.211307, 0.0581584,  0.975688,  -1.50686,  -8.75256,  0.175648,  -1.50689,  -8.75258,   0.17432, -0.210924, 0.0558978,  0.975903;
			ekfom_data.h(effct_feat_num - other_least_square_size + i) = -1.00109248;
			ekfom_data.R(effct_feat_num - other_least_square_size + i) = 0.027647;
	    
	    	}
	}	
	
    //}
    //else
    //{
    //		ekfom_data.h_x.block<1, 12>(effct_feat_num - other_least_square_size + i,0) << -0.211307, 0.0581584,  0.975688,  -1.50686,  -8.75256,  0.175648,  -1.50689,  -8.75258,   0.17432, -0.210924, 0.0558978,  0.975903;
    //		ekfom_data.h(effct_feat_num - other_least_square_size + i) = -1.00109248;
    //		ekfom_data.R(effct_feat_num - other_least_square_size + i) = 0.027647;
    
    //    	}
    
    //}
    //*/
    
    

	//ekfom_data.h_x.block<1, 12>(i,0) << -0.211307, 0.0581584,  0.975688,  -1.50686,  -8.75256,  0.175648,  -1.50689,  -8.75258,   0.17432, -0.210924, 0.0558978,  0.975903;
	//ekfom_data.h(i) = -1.00109248;
	//ekfom_data.R(i) = 0.027647;

    // std::printf("Effective Points: %d\n", effct_feat_num);
    res_mean_last = total_residual / effct_feat_num;
    // std::printf("res_mean: %f\n", res_mean_last);
    // std::printf("ef_num: %d\n", effct_feat_num);
}





void extractNARFKeypoints(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<int>::Ptr& keypoints) {
    // Create a range image from the point cloud
    float angular_resolution = pcl::deg2rad(0.5f);
    Eigen::Affine3f sensor_pose = Eigen::Affine3f::Identity();  // Assume the sensor is at the origin

    pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
    float noise_level = 0.0;
    float min_range = 0.0f;
    int border_size = 1;

    pcl::RangeImage::Ptr range_image_ptr(new pcl::RangeImage);
    pcl::RangeImage& range_image = *range_image_ptr;   
    range_image.createFromPointCloud(*cloud, angular_resolution, pcl::deg2rad(360.0f), pcl::deg2rad(180.0f),
                                     sensor_pose, coordinate_frame, noise_level, min_range, border_size);

    // Detect NARF keypoints
    pcl::NarfKeypoint narf_keypoint_detector;
    narf_keypoint_detector.setRangeImage(&range_image);
    narf_keypoint_detector.getParameters().support_size = 0.2f;  // Adjust based on your data
    narf_keypoint_detector.compute(*keypoints);
}

void segmentPlanes(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<int>::Ptr& keypoints) {
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());

    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    // Extract inliers
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    extract.filter(*plane_cloud);

    // Combine keypoints and plane points
    pcl::PointCloud<int>::Ptr combined_indices(new pcl::PointCloud<int>(*keypoints));
    for (const auto& idx : inliers->indices) {
        combined_indices->points.push_back(idx);
    }

    *keypoints = *combined_indices;
}

void alignPointCloudsGICP(pcl::PointCloud<pcl::PointXYZ>::Ptr source,
                          pcl::PointCloud<pcl::PointXYZ>::Ptr target,
                          Eigen::Matrix4f& transformation_matrix) {

    pcl::PointCloud<pcl::PointXYZ> aligned;

		// 1st icp for cloud_in [now] and cloud_pre [last]
		pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
		icp.setInputSource(source);
		icp.setInputTarget(target);
		icp.align(aligned);
			


    if (icp.hasConverged()) {
        transformation_matrix = icp.getFinalTransformation();
        std::cout << "GICP has converged, score: " << icp.getFitnessScore() << std::endl;
    } else {
        std::cerr << "GICP did not converge." << std::endl;
    }
}








Eigen::Vector3f getLongestDirection(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    // Compute the centroid of the point cloud
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);

    // Subtract the centroid from the point cloud
    Eigen::MatrixXf centered_points(cloud->size(), 3);
    for (size_t i = 0; i < cloud->size(); ++i) {
        centered_points(i, 0) = cloud->points[i].x - centroid(0);
        centered_points(i, 1) = cloud->points[i].y - centroid(1);
        centered_points(i, 2) = cloud->points[i].z - centroid(2);
    }

    // Compute the covariance matrix
    Eigen::Matrix3f covariance_matrix = (centered_points.transpose() * centered_points) / (cloud->size() - 1);

    // Perform eigenvalue decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance_matrix);
    Eigen::Matrix3f eigen_vectors = eigen_solver.eigenvectors();
    Eigen::Vector3f eigen_values = eigen_solver.eigenvalues();

    // The eigenvector corresponding to the largest eigenvalue is the direction of the longest axis
    Eigen::Vector3f longest_direction = eigen_vectors.col(2);  // Column 2 has the largest eigenvalue

    return longest_direction;
}




// Function to compute the Hessian matrix
Eigen::Matrix<double, 6, 6> computeHessian(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {

    cout<<"size="<<cloud->size()<<endl;
    Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Zero();

   // Create the normal estimation class, and pass the input dataset to it
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree);

    // Output datasets
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);

    // Use all neighbors in a sphere of radius 3cm
    ne.setRadiusSearch(0.5);

    // Compute the features
    ne.compute(*cloud_normals);


    for (size_t i = 0; i < cloud->points.size(); ++i) {

	Eigen::Vector3d p(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
	Eigen::Vector3d n(cloud_normals->points[i].normal_x, cloud_normals->points[i].normal_y, cloud_normals->points[i].normal_z);

        
        if (!std::isnan(n.x()) && !std::isnan(n.y()) && !std::isnan(n.z())) 
        {

		Eigen::Matrix<double, 6, 1> Hi;
		Hi.block<3, 1>(0, 0) = -p.cross(n);
		Hi.block<3, 1>(3, 0) = -n;

		A += Hi * Hi.transpose();
        }
    }
    
    cout<<"x_normal="<<cloud_normals->points[100].normal_x<<"y_normal="<<cloud_normals->points[100].normal_y<<"z_normal="<<cloud_normals->points[100].normal_z<<endl;
    cout<<A<<endl;

    return 2.0 * A;
}

// Function to compute the covariance matrix
Eigen::Matrix<double, 6, 6> computeCovarianceMatrix(const Eigen::Matrix<double, 6, 6>& H) {
    return H;
}

// Function to perform eigen decomposition and return eigenvalues and eigenvectors
void computeEigenDecomposition(
    const Eigen::Matrix<double, 3, 3>& covariance_matrix,
    Eigen::VectorXd& eigenvalues,
    Eigen::MatrixXd& eigenvectors)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> eigensolver(covariance_matrix);

    if (eigensolver.info() != Eigen::Success) {
        throw std::runtime_error("Eigenvalue decomposition failed!");
    }

    eigenvalues = eigensolver.eigenvalues();
    eigenvectors = eigensolver.eigenvectors();
}

// Function to create and publish visualization markers
void publishEllipsoidMarker(
    ros::Publisher& marker_pub,
    const Eigen::Vector3d& position,
    const Eigen::Vector3d& scales,
    const Eigen::Matrix3d& rotation_matrix,
    const std::string& frame_id)
{
    visualization_msgs::Marker ellipsoid_marker;
    ellipsoid_marker.header.frame_id = frame_id;
    ellipsoid_marker.header.stamp = ros::Time::now();
    ellipsoid_marker.ns = "covariance_ellipsoid";
    ellipsoid_marker.id = 0;
    ellipsoid_marker.type = visualization_msgs::Marker::SPHERE;
    ellipsoid_marker.action = visualization_msgs::Marker::ADD;

    // Set position
    ellipsoid_marker.pose.position.x = position.x();
    ellipsoid_marker.pose.position.y = position.y();
    ellipsoid_marker.pose.position.z = position.z();

    // Set orientation from rotation matrix
    Eigen::Quaterniond quat(rotation_matrix);
    ellipsoid_marker.pose.orientation.x = quat.x();
    ellipsoid_marker.pose.orientation.y = quat.y();
    ellipsoid_marker.pose.orientation.z = quat.z();
    ellipsoid_marker.pose.orientation.w = quat.w();

    // Set scale (standard deviations)
    ellipsoid_marker.scale.x = scales.x() * 2; // multiply by 2 for full length
    ellipsoid_marker.scale.y = scales.y() * 2;
    ellipsoid_marker.scale.z = scales.z() * 2;

    // Set color
    ellipsoid_marker.color.r = 0.0f;
    ellipsoid_marker.color.g = 1.0f;
    ellipsoid_marker.color.b = 0.0f;
    ellipsoid_marker.color.a = 0.5f; // semi-transparent

    // Set lifetime
    ellipsoid_marker.lifetime = ros::Duration();

    // Publish marker
    marker_pub.publish(ellipsoid_marker);
}



// Include OpenMP for parallelization
#ifdef _OPENMP
#include <omp.h>
#endif

// Function to downsample point clouds using a voxel grid filter
void downsamplePointCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud,
    float leaf_size)
{
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud(input_cloud);
    voxel_grid.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel_grid.filter(*output_cloud);
}



// Function to perform ICP alignment
void run_ICP(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud,
    const Eigen::Matrix4f& initial_guess,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& aligned_cloud,
    Eigen::Matrix4f& final_transformation)
{
    // Parameters
    const int max_iterations = 5; // Reduced iterations
    const double convergence_threshold = 1e-3;
    const double max_correspondence_distance = 0.07; // Adjust based on your data

    // Initialize
    final_transformation = initial_guess;
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_source(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*source_cloud, *transformed_source, initial_guess);

    // Build Kd-tree for the target cloud
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(target_cloud);

    for (int iter = 0; iter < max_iterations; ++iter)
    {
        // Step 1: Find Correspondences
        std::vector<int> correspondences(transformed_source->size(), -1);

        // Parallelize the correspondence search
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(transformed_source->size()); ++i)
        {
            const pcl::PointXYZ& src_point = transformed_source->points[i];
            std::vector<int> pointIdxNKNSearch(1);
            std::vector<float> pointNKNSquaredDistance(1);

            if (kdtree.nearestKSearch(src_point, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
            {
                if (pointNKNSquaredDistance[0] < max_correspondence_distance * max_correspondence_distance)
                {
                    correspondences[i] = pointIdxNKNSearch[0];
                }
            }
        }

        // Collect valid correspondences
        std::vector<Eigen::Vector3f> src_points;
        std::vector<Eigen::Vector3f> tgt_points;
        src_points.reserve(transformed_source->size());
        tgt_points.reserve(transformed_source->size());

        for (size_t i = 0; i < correspondences.size(); ++i)
        {
            if (correspondences[i] != -1)
            {
                src_points.push_back(transformed_source->points[i].getVector3fMap());
                tgt_points.push_back(target_cloud->points[correspondences[i]].getVector3fMap());
            }
        }

        // Check if there are enough correspondences
        if (src_points.size() < 3)
        {
            std::cout << "Not enough correspondences found." << std::endl;
            break;
        }

        // Step 2: Compute Centroids
        Eigen::Vector3f src_centroid = Eigen::Vector3f::Zero();
        Eigen::Vector3f tgt_centroid = Eigen::Vector3f::Zero();
        for (size_t i = 0; i < src_points.size(); ++i)
        {
            src_centroid += src_points[i];
            tgt_centroid += tgt_points[i];
        }
        src_centroid /= static_cast<float>(src_points.size());
        tgt_centroid /= static_cast<float>(tgt_points.size());

	//cout<<" [src_centroid] = "<<src_centroid<<endl;
	//cout<<" [tgt_centroid] = "<<tgt_centroid<<endl;

        // Step 3: Compute Covariance Matrix
        Eigen::Matrix3f W = Eigen::Matrix3f::Zero();
        for (size_t i = 0; i < src_points.size(); ++i)
        {
            W += (src_points[i] - src_centroid) * (tgt_points[i] - tgt_centroid).transpose();
        }

        // Step 4: Singular Value Decomposition
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3f U = svd.matrixU();
        Eigen::Matrix3f V = svd.matrixV();

        Eigen::Matrix3f R = V * U.transpose();

        // Ensure a proper rotation (determinant = 1)
        if (R.determinant() < 0)
        {
            V.col(2) *= -1;
            R = V * U.transpose();
        }

        Eigen::Vector3f t = tgt_centroid - R * src_centroid;

        // Step 5: Build Transformation Matrix
        Eigen::Matrix4f delta_transform = Eigen::Matrix4f::Identity();
        delta_transform.block<3,3>(0,0) = R;
        delta_transform.block<3,1>(0,3) = t;

        // Update the overall transformation
        final_transformation = delta_transform * final_transformation;

        // Step 6: Transform the source cloud
        pcl::transformPointCloud(*source_cloud, *transformed_source, final_transformation);

        // Step 7: Check for convergence
        double delta = (delta_transform - Eigen::Matrix4f::Identity()).norm();
        if (delta < convergence_threshold)
        {
            std::cout << "Converged after " << iter + 1 << " iterations." << std::endl;
            break;
        }
    }

    // Output the aligned cloud
    *aligned_cloud = *transformed_source;
}

// Define the bounds of the 3x3 field (assuming it is centered at the origin)
const float field_min = -2.0; // The minimum value for x, y, z (centered at origin)
const float field_max = 2.0;  // The maximum value for x, y, z (centered at origin)

bool is_50_percent_in_field(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in_filtered) {
    // Count the number of points within the 3x3 field
    int points_in_field = 0;
    int total_points = cloud_in_filtered->points.size();

    for (const auto& point : cloud_in_filtered->points) {
        // Check if the point is within the 3x3 cube
        if (point.x >= field_min && point.x <= field_max &&
            point.y >= field_min && point.y <= field_max &&
            point.z >= field_min && point.z <= field_max) {
            points_in_field++;
        }
    }

    // Calculate the percentage of points inside the 3x3 field
    float percentage_in_field = static_cast<float>(points_in_field) / total_points * 100.0f;

    // Check if 70% or more of the points are inside the field
    return percentage_in_field >= 50.0f;
}




vector<Eigen::Vector3f> longest_direction_vector;

// Function to be executed in the new thread
void processPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr world_lidar_point, pcl::PointCloud<pcl::PointXYZ>::Ptr feats_down_body) {

	    //--------------------------------------------------------------------------------------------backup_system[start]--------------------------------------------------------------------------------------//
            
            double t_update_start_gicp = omp_get_wtime();
            
            int feats_down_size = feats_down_body->points.size();

            cout<<"==================feats_down_size[1]====================="<<endl;
            cout<<feats_down_size<<endl;
            
            pcl::PointCloud<pcl::PointXYZ>::Ptr Final_cloud_translate (new pcl::PointCloud<pcl::PointXYZ>(5,1));
            
	    if( counter_loop == 0 )
	    {
		// Initialize the map
		*map_final = *world_lidar_point;
		*cloud_pre = *feats_down_body;
		*cloud_in_filtered = *feats_down_body;
		counter_loop++;
	    }
	    else
	    {

		// 1.0 remove the Nan points
		// remove the NaN points and remove the points out of distance
		std::vector<int> indices;
		pcl::removeNaNFromPointCloud(*feats_down_body, *feats_down_body, indices);

		bool in_field = is_50_percent_in_field(feats_down_body);

		if(  in_field == true  )
		{
			//return;
		}

		// 1.1 Voxel the cloud

		// Create the filtering object
		pcl::VoxelGrid<pcl::PointXYZ> sor_input;
		sor_input.setInputCloud (feats_down_body);
		sor_input.setLeafSize (0.5f, 0.5f, 0.5f);
		sor_input.filter (*cloud_in_filtered);
		
		// Use of Hybrid ICP Algorithms
		//pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
		//icp.setMaxCorrespondenceDistance(0.5);
		//icp.setTransformationEpsilon(1e-6);
		//icp.setEuclideanFitnessEpsilon(1e-6);

		//icp.align(*aligned_cloud_initial);

		// Use initial alignment for GICP
		//gicp.setInputSource(aligned_cloud_initial);
		//gicp.align(*aligned_cloud_refined);
		
		/*
		
		// Optional: Apply Statistical Outlier Removal filter
		pcl::StatisticalOutlierRemoval<pcl::PointXYZ> statistical_sor;
		statistical_sor.setInputCloud(cloud_in_filtered);
		statistical_sor.setMeanK(50);
		statistical_sor.setStddevMulThresh(1.0);
		statistical_sor.filter(*cloud_in_filtered);
		
		//*/
		
		int cloud_in_filtered_size = cloud_in_filtered->points.size();

		cout<<"==================cloud_in_filtered_size[1]====================="<<endl;
		cout<<cloud_in_filtered_size<<endl;
		
		// Check if point clouds are populated
		if ( cloud_in_filtered->points.size() < 30 )
		{
			std::cerr << "cloud_in_filtered Source point cloud is empty!" << std::endl;
			return ;
		}

		if ( cloud_pre->points.size() < 30 )
		{
			std::cerr << "cloud_pre Target point cloud is empty!" << std::endl;
			return ;
		}   
		
		
		// 1.2 boxed the cloud
		*cloud_in_boxed_for_local = *cloud_in_filtered;

		pcl::CropBox<pcl::PointXYZ> boxFilter_for_in;
		float x_min_for_in = - 50, y_min_for_in = - 50, z_min_for_in = - 0 ;
		float x_max_for_in = + 50, y_max_for_in = + 50, z_max_for_in = + 50;

		boxFilter_for_in.setMin(Eigen::Vector4f(x_min_for_in, y_min_for_in, z_min_for_in, 1.0));
		boxFilter_for_in.setMax(Eigen::Vector4f(x_max_for_in, y_max_for_in, z_max_for_in, 1.0));

		boxFilter_for_in.setInputCloud(cloud_in_boxed_for_local);
		boxFilter_for_in.filter(*cloud_in_boxed_for_local);
	
		//pcl::PointCloud<pcl::PointXYZ> Final;	
		
		if( counter_loop < 2 )
		{
			// 1st icp for cloud_in [now] and cloud_pre [last]
			pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
			icp.setInputSource(cloud_in_filtered);
			icp.setInputTarget(cloud_pre);
			icp.align(*Final);
				
			// 2.2 get the x,y,z of the odometry of the trajectory
			Ti = icp.getFinalTransformation () * Ti;

			
		}
		else
		{
		
		
			// ------------------------------------------5_line[start] ---------------------------------------------------// 
			auto t1_multi = std::chrono::high_resolution_clock::now();
			//clock_t start = clock(); // Get the current time in clock ticks
			

			int num_threads = 5; // Adjust based on your CPU and desired usage
			
			// Varying parameters for each thread
			std::vector<float> max_correspondence_dists = {0.05, 0.2, 0.4, 0.1, 0.3};
			std::vector<int> max_iterations = {30, 50, 70, 40, 60};
			std::vector<Eigen::Matrix4f> initial_guesses(num_threads, Eigen::Matrix4f::Identity());

			// Adding some small perturbations to initial guesses to make them different
			for (int i = 0; i < num_threads; ++i) {
				initial_guesses[i](0, 3) += 0.01f * i;  // Slight translation perturbation
				initial_guesses[i](1, 3) += 0.01f * i;
			}


			std::vector<std::thread> threads;
			std::vector<GICPResult> results(num_threads);


			/*
			// Create a new PointXYZ cloud to store the converted points
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_filtered_XYZ(new pcl::PointCloud<pcl::PointXYZ>);

			// Convert PointXYZI to PointXYZ
			for (const auto& point : cloud_in_filtered->points) {
				pcl::PointXYZ new_point;
				new_point.x = point.x;
				new_point.y = point.y;
				new_point.z = point.z;
				cloud_in_filtered_XYZ->points.push_back(new_point);
			}

			// Create a new PointXYZ cloud to store the converted points
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_XYZ_pre(new pcl::PointCloud<pcl::PointXYZ>);

			// Convert PointXYZI to PointXYZ
			for (const auto& point : cloud_pre->points) {
				pcl::PointXYZ new_point;
				new_point.x = point.x;
				new_point.y = point.y;
				new_point.z = point.z;
				cloud_XYZ_pre->points.push_back(new_point);
			}
			//*/
		        
			// Launch multiple GICP processes
			#pragma omp simd
			for (int i = 0; i < num_threads; ++i) {
				threads.emplace_back(runGICP, cloud_in_filtered, cloud_pre, std::ref(results[i]), max_correspondence_dists[i], max_iterations[i], initial_guesses[i]);

			}

			// Join all threads
			for (auto& thread : threads) {
				thread.join();
			}

			// Select the best GICP result
			GICPResult best_result = selectBestResult(results);

			// Output the best transformation
			std::cout << "Best Transformation Matrix:\n" << best_result.transformation << std::endl;
			std::cout << "Best Alignment Error: " << best_result.alignment_error << std::endl;

			
			//if( best_result.transformation(0,3) > 0.3 || best_result.transformation(1,3) > 0.3 || best_result.transformation(2,3) > 0.3 )
			//{
			//	return;
			//}

			//if(  best_result.alignment_error == 0.0 || in_field == true  )
			if(  best_result.alignment_error == 0.0 )
			{
				return;
			}
			
			Ti = best_result.transformation * Ti;



			//clock_t end = clock(); // Get the current time in clock ticks

			//double runningTime = (double)(end - start) / CLOCKS_PER_SEC; // Convert clock ticks to seconds

			//std::cout << "Running time: " << runningTime << " seconds" << std::endl;
	//*/

			auto t2_multi = std::chrono::high_resolution_clock::now();
			double single_multi = std::chrono::duration_cast<std::chrono::nanoseconds>(t2_multi - t1_multi).count() / 1e6;
			std::cout << "single_multi:" << single_multi << "[msec] " << std::endl;

			// ------------------------------------------5_line[end] ---------------------------------------------------// 
		
		
		
		
	/*	
		
		
			// 2.0 gicp of the cloud_in voxeled and pre cloud
			pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
			gicp.setMaxCorrespondenceDistance(1.0);
			gicp.setTransformationEpsilon(0.001);
			gicp.setMaximumIterations(1000);
			
			gicp.setInputSource(cloud_in_filtered);
			gicp.setInputTarget(cloud_pre);
			gicp.align(Final);
			Ti = gicp.getFinalTransformation () * Ti;
			//*/
		
		}
		//*/


		//pcl::PointCloud<pcl::PointXYZ> Final;

               /*
               std::cout << "================================[0]================================[" << std::endl;
		// 2.0 gicp of the cloud_in voxeled and pre cloud
		pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
		//pcl::registration::GICPWithRobustKernel<pcl::PointXYZ> gicp;
		gicp.setMaxCorrespondenceDistance(1.0);
		gicp.setTransformationEpsilon(0.02);
		gicp.setMaximumIterations(500);

		gicp.setInputSource(cloud_in_filtered);
		gicp.setInputTarget(cloud_pre);
		gicp.align(*Final);
		Ti = gicp.getFinalTransformation () * Ti;
		
		cout<<"==============================Ti========================="<<endl;
		cout<<Ti<<endl;            
               //*/
               
		// 3.0 voxed the map_final
		// Create the filtering object
		pcl::VoxelGrid<pcl::PointXYZ> sor;
		sor.setInputCloud (map_final);
		sor.setLeafSize (0.5f, 0.5f, 0.5f);
		sor.filter (*map_final);
	
		if( counter_stable_map % 20 == 0 )
		{
			// 3.1 boxed the map_final
			pcl::CropBox<pcl::PointXYZ> boxFilter;
			float x_min = Ti(0,3) - 50, y_min = Ti(1,3) - 50, z_min = Ti(2,3) - 0;
			float x_max = Ti(0,3) + 50, y_max = Ti(1,3) + 50, z_max = Ti(2,3) + 50;

			boxFilter.setMin(Eigen::Vector4f(x_min, y_min, z_min, 1.0));
			boxFilter.setMax(Eigen::Vector4f(x_max, y_max, z_max, 1.0));

			boxFilter.setInputCloud(map_final);
			boxFilter.filter(*map_final_boxed);
		}
		
		counter_stable_map++;
               
   
               
               
		//3.3 get the gicp of the cloud in boxed and the map boxed

		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_boxed_translate_to_near_mapboxed (new pcl::PointCloud<pcl::PointXYZ>);
		pcl::transformPointCloud (*cloud_in_boxed_for_local, *cloud_in_boxed_translate_to_near_mapboxed, Ti); 
      
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_boxed_translate_to_near_mapboxed_far_icp (new pcl::PointCloud<pcl::PointXYZ>(5,1));
		pcl::transformPointCloud (*cloud_in_boxed_for_local, *cloud_in_boxed_translate_to_near_mapboxed_far_icp, Ti);  
      
                // Check if point clouds are populated
		if (cloud_in_boxed_translate_to_near_mapboxed->empty() )
		{
			std::cerr << "Source point cloud is empty!" << std::endl;
			return ;
		}

		if (map_final_boxed->empty() )
		{
			std::cerr << "Target point cloud is empty!" << std::endl;
			return ;
		}        

		std::cout << "[ map for fly ] Ti Transformation matrix:\n" << Ti << std::endl;

		auto t1 = std::chrono::high_resolution_clock::now();

		pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		Eigen::Matrix4f final_transformation = Eigen::Matrix4f::Identity();
		

		
		run_ICP( cloud_in_boxed_translate_to_near_mapboxed_far_icp, map_final_boxed,  initial_guess, aligned_cloud, final_transformation );
		std::cout << "[ map for fly ] run_ICP:\n" << final_transformation << std::endl;

		auto t2 = std::chrono::high_resolution_clock::now();
		double single = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
		std::cout << "single:" << single << "[msec] " << std::endl;

               
               //std::cout << "================================[0.1]================================" << std::endl;
               
               auto t1_ndt = std::chrono::high_resolution_clock::now();
               
		// Step 1: Initial alignment using NDT
		pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
		ndt.setResolution(2.0);
		ndt.setInputSource(cloud_in_boxed_translate_to_near_mapboxed_far_icp);
		ndt.setInputTarget(map_final_boxed);

		// NDT parameters
		ndt.setMaximumIterations(20);
		ndt.setStepSize(0.2);
		ndt.setTransformationEpsilon(0.05);

		// Initial guess for the transformation (identity matrix)
		Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();

		// Perform the initial alignment
		ndt.align(*Final, final_transformation);
		//std::cout << "================================[0.2]================================" << std::endl;
		// Check if NDT has converged
		if (ndt.hasConverged()) {
			std::cout << "NDT converged with score: " << ndt.getFitnessScore() << std::endl;
			std::cout << "NDT Transformation matrix:\n" << ndt.getFinalTransformation() << std::endl;
		} else {
			std::cout << "NDT did not converge." << std::endl;
			//return -1;
		}

               

		auto t2_ndt = std::chrono::high_resolution_clock::now();
		double single_ndt = std::chrono::duration_cast<std::chrono::nanoseconds>(t2_ndt - t1_ndt).count() / 1e6;
		std::cout << "single_ndt:" << single_ndt << "[msec] " << std::endl;

               
		cout<<"=================== cloud_in_filtered->size() ==================" <<endl;
		cout<<cloud_in_filtered->size() <<endl;
               
               
               //std::cout << "================================[1]================================" << std::endl;
               auto t1_gicp = std::chrono::high_resolution_clock::now();
               
		pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp_for_map;
		gicp_for_map.setMaxCorrespondenceDistance(10.0);
		gicp_for_map.setTransformationEpsilon(0.01);
		gicp_for_map.setRotationEpsilon(0.01);
		gicp_for_map.setMaximumIterations(1000);

		gicp_for_map.setInputSource(cloud_in_boxed_translate_to_near_mapboxed_far_icp);
		gicp_for_map.setInputTarget(map_final_boxed);
		//gicp_for_map.align(*Final);
		gicp_for_map.align(*Final, ndt.getFinalTransformation());

		Ti_of_map = gicp_for_map.getFinalTransformation (); // * Ti_of_map;
               
               Ti_real = Ti_of_map * Ti;
		       
		std::cout << "Ti_of_map Transformation matrix:\n" << Ti_of_map << std::endl;

		auto t2_gicp = std::chrono::high_resolution_clock::now();
		double single_gicp = std::chrono::duration_cast<std::chrono::nanoseconds>(t2_gicp - t1_gicp).count() / 1e6;
		std::cout << "single_gicp:" << single_gicp << "[msec] " << std::endl;




		
		
		Eigen::Matrix4f rotation_matrix = Ti_of_map;
		double yaw_of_cloud_ti_to_map = atan2( rotation_matrix(1,0),rotation_matrix(0,0) )/3.1415926*180;
		
		if( abs( Ti_of_map(0,3) ) > 0.2 || abs( Ti_of_map(1,3) ) > 0.2  || abs( Ti_of_map(2,3) ) > 0.2 ||  abs( yaw_of_cloud_ti_to_map ) > 1)
		{
			//cout<<"===========Ti_real=============="<<endl;
			//cout<<Ti_of_map<<endl;
			//cout<<Ti<<endl;
			//cout<<Ti_real<<endl;
			Ti = Ti_real;


		}
		

		//cout<<"========================Eigen::Matrix4d_T[1.1]======================"<<endl;  
		//cout<<T<<endl;
		//cout<< T(0,3) - Ti_real(0,3) <<","<<T(1,3) - Ti_real(1,3)<<","<<T(2,3) - Ti_real(2,3)<<endl;


		//if( abs( T(0,3) - Ti_real(0,3) ) > 0.2 || abs( T(1,3) - Ti_real(1,3) ) > 0.2  || abs( T(2,3) - Ti_real(2,3) ) > 0.2 )
		//{

			//Ti_real(0,3)  = T.cast<float>()(0,3) ;
			//Ti_real(1,3)  = T.cast<float>()(1,3) ;
			//Ti_real(2,3)  = T.cast<float>()(2,3) ;

		//}
		
		//cout<<"========================Eigen::Matrix4d_Ti[1.2]======================"<<endl;  
		//cout<<Ti_real<<endl;
		//cout<<Ti<<endl;

		
		
		
		pcl::transformPointCloud (*cloud_in_filtered, *Final_cloud_translate, Ti_real); 

		if( counter_loop % 10 == 0 )
		{
			//*map_final += *world_lidar_point; 
			

			// 3.1 boxed the map_final
			pcl::CropBox<pcl::PointXYZ> box_filter;
			float x_min = Ti_real(0,3) - 50, y_min = Ti_real(1,3) - 50, z_min = Ti_real(2,3) - 50;
			float x_max = Ti_real(0,3) + 50, y_max = Ti_real(1,3) + 50, z_max = Ti_real(2,3) + 50;

			box_filter.setMin(Eigen::Vector4f(x_min, y_min, z_min, 1.0));
			box_filter.setMax(Eigen::Vector4f(x_max, y_max, z_max, 1.0));

			box_filter.setInputCloud(map_final);
			box_filter.filter(*map_final_boxed_2);
			

			pcl::PointCloud<pcl::PointXYZ> Final_for_add_to_map;

			//std::cout << "================================[3]================================[" << std::endl;
			pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp_for_add_to_map_final;
			gicp_for_add_to_map_final.setMaxCorrespondenceDistance(5.0);
			gicp_for_add_to_map_final.setTransformationEpsilon(0.001);
			gicp_for_add_to_map_final.setMaximumIterations(1000);

			gicp_for_add_to_map_final.setInputSource(Final_cloud_translate);
			gicp_for_add_to_map_final.setInputTarget(map_final_boxed_2);
			gicp_for_add_to_map_final.align(Final_for_add_to_map);



			if( abs( gicp_for_add_to_map_final.getFinalTransformation ()(0,3) ) < 1 && abs( gicp_for_add_to_map_final.getFinalTransformation ()(1,3) ) < 1 ) 
			{
				Ti = gicp_for_add_to_map_final.getFinalTransformation () * Ti;
				*map_final += Final_for_add_to_map;
			} 
			

			
			
			
		}
               
	    	counter_loop++;
            }
            
            *cloud_pre = *cloud_in_filtered;
            
            
            
            // Get the longest direction of the tunnel
            Eigen::Vector3f longest_direction = getLongestDirection(cloud_pre);

	    
	    longest_direction_vector.push_back(longest_direction);

            // Convert Eigen::Vector3f to geometry_msgs::Vector3
            geometry_msgs::Vector3 direction_msg;
            direction_msg.x = longest_direction(0);
            direction_msg.y = longest_direction(1);
            direction_msg.z = longest_direction(2);

            direction_pub.publish(direction_msg);
	    cout<<"longest_direction="<<longest_direction<<endl;
	    
	    // Initialize mean and standard deviation vectors
	    Eigen::Vector3f mean_longest_direction = Eigen::Vector3f::Zero();
	    Eigen::Vector3f std_longest_direction = Eigen::Vector3f::Zero();

	    // Compute the mean of the longest directions
	    for (int i = 0; i < longest_direction_vector.size(); i++) {
	        mean_longest_direction += longest_direction_vector[i];
	    }

	    mean_longest_direction /= longest_direction_vector.size();

	    // Compute the standard deviation
	    for (int i = 0; i < longest_direction_vector.size(); i++) {
	        std_longest_direction += (longest_direction_vector[i] - mean_longest_direction).array().square().matrix();
	    }

	    std_longest_direction = (std_longest_direction / longest_direction_vector.size()).array().sqrt().matrix();

	    cout << "mean_longest_direction=" << mean_longest_direction.transpose() << endl;
	    cout << "std_longest_direction=" << std_longest_direction.transpose() << endl;
	    
	    
	    geometry_msgs::Vector3 mean_msg;
	    geometry_msgs::Vector3 std_msg;

	    mean_msg.x = mean_longest_direction.x();
	    mean_msg.y = mean_longest_direction.y();
	    mean_msg.z = mean_longest_direction.z();

	    std_msg.x = std_longest_direction.x();
	    std_msg.y = std_longest_direction.y();
	    std_msg.z = std_longest_direction.z();
	    
	    mean_dir_pub.publish(mean_msg);
	    std_dir_pub.publish(std_msg);
	    
	    

	    // Create a marker for RViz
	    visualization_msgs::Marker marker;
	    marker.header.frame_id = "world";  // Use the appropriate frame
	    marker.header.stamp = ros::Time::now();
	    marker.ns = "tunnel_direction";
	    marker.id = 0;
	    marker.type = visualization_msgs::Marker::ARROW;
	    marker.action = visualization_msgs::Marker::ADD;

	    // Set the start and end points of the arrow
	    marker.points.resize(2);
	    marker.points[0].x = 0;  // Starting point (assume origin)
	    marker.points[0].y = 0;
	    marker.points[0].z = 0;
	    marker.points[1].x = direction_msg.x;  // End point (direction vector)
	    marker.points[1].y = direction_msg.y;
	    marker.points[1].z = direction_msg.z;

	    // Set the color and scale of the arrow
	    marker.scale.x = 0.1;  // Shaft diameter
	    marker.scale.y = 0.2;  // Head diameter
	    marker.scale.z = 0.2;  // Head length
	    marker.color.r = 1.0f;
	    marker.color.g = 0.0f;
	    marker.color.b = 0.0f;
	    marker.color.a = 1.0f;

	    marker_pub.publish(marker);
            
            cout<<"longest_direction[1]="<<longest_direction<<endl;
            
            
            
            // 3.1 boxed the map_final
            pcl::CropBox<pcl::PointXYZ> box_filter_cov;
            float x_min = Ti_real(0,3) - 15, y_min = Ti_real(1,3) - 15, z_min = Ti_real(2,3) - 15;
            float x_max = Ti_real(0,3) + 15, y_max = Ti_real(1,3) + 15, z_max = Ti_real(2,3) + 15;

            box_filter_cov.setMin(Eigen::Vector4f(x_min, y_min, z_min, 1.0));
            box_filter_cov.setMax(Eigen::Vector4f(x_max, y_max, z_max, 1.0));

            box_filter_cov.setInputCloud(map_final);
            box_filter_cov.filter(*map_final_boxed_cov);
            
            

            if( map_final_boxed_cov->size() > 0 )
            {
		// Step 1: Compute Hessian
		Eigen::Matrix<double, 6, 6> H = computeHessian(map_final_boxed_cov);

		cout<<"computeHessian[1]="<<H<<endl;


		if(!H.isZero())
		{

		    // Step 2: Compute Covariance Matrix
		    Eigen::Matrix<double, 6, 6> covariance_matrix = computeCovarianceMatrix(H);

		   Eigen::Matrix<double, 3, 3> covariance_matrix_tt = covariance_matrix.bottomRightCorner<3, 3>();
		   
		   cout<<"covariance_matrix_tt[1]="<<covariance_matrix_tt<<endl;
		   
		    // Step 3: Compute Eigenvalues and Eigenvectors
		    Eigen::VectorXd eigenvalues;
		    Eigen::MatrixXd eigenvectors;
		    computeEigenDecomposition(covariance_matrix_tt, eigenvalues, eigenvectors);

		    // Prepare eigenvalues message
		    std_msgs::Float64MultiArray eigenvalues_msg;
		    for (int i = 0; i < eigenvalues.size(); ++i) {
		    	eigenvalues_msg.data.push_back(eigenvalues[i]);
		    }

		    // Compute centroid of the point cloud for marker position
		    Eigen::Vector4d centroid;
		    pcl::compute3DCentroid(*map_final_boxed_cov, centroid);

		    // Extract rotational part (first 3 components)
		    Eigen::Vector3d rotation_eigenvalues = eigenvalues.head(3);
		    Eigen::Matrix3d rotation_eigenvectors = eigenvectors.block<3,3>(0,0);

		    // For visualization, take square roots of eigenvalues (standard deviations)
		    
		    for(int i = 0; i < 3; ++i) {
		    	scales[i] = std::sqrt(rotation_eigenvalues[i]);
		    }

		    cout<<"===================scales===================="<<endl;
		    cout<<scales<<endl;

		    eigenvalue_pub.publish(eigenvalues_msg);

		    // Publish ellipsoid marker
		    publishEllipsoidMarker(
			    marker_pub_Hessien,
			    centroid.head<3>(),
			    scales,
			    rotation_eigenvectors,
			    "world" // Replace with appropriate frame_id
		    );
		}
	    }

            cout<<"longest_direction[2]="<<longest_direction<<endl;
            
            double time_laser_odom_;
            nav_msgs::OdometryPtr msg_2(new nav_msgs::Odometry);
            msg_2->header.stamp.fromSec(time_laser_odom_);
            msg_2->header.frame_id = "world";
            msg_2->child_frame_id = "/laser";
            msg_2->pose.pose.position.x = Ti_real(0,3);
            msg_2->pose.pose.position.y = Ti_real(1,3);
            msg_2->pose.pose.position.z = Ti_real(2,3);
            msg_2->pose.pose.orientation.w = 1;
            msg_2->pose.pose.orientation.x = 0;
            msg_2->pose.pose.orientation.y = 0;
            msg_2->pose.pose.orientation.z = 0;
            pub_odom_aft_mapped_2.publish(msg_2);
            
            
            double t_update_end_gicp = omp_get_wtime();
            double optimize_time_gicp = t_update_start_gicp - t_update_end_gicp;
            
            cout<<"optimize_time_gicp="<<optimize_time_gicp<<endl;
            //std::this_thread::sleep_for(std::chrono::milliseconds(30));
            //*/
            //--------------------------------------------------------------------------------------------backup_system[end]--------------------------------------------------------------------------------------//
            
}


int counter_gicp = 0;
  
//process the input cloud
void main_callback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
  
  
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>(5,1));
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_boxed_for_local (new pcl::PointCloud<pcl::PointXYZ>(5,1));
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_filtered (new pcl::PointCloud<pcl::PointXYZ>(5,1));
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_cut (new pcl::PointCloud<pcl::PointXYZ>(5,1));
    
    
    pcl::fromROSMsg(*msg, *cloud_in);

    if( counter_gicp == 0 )
    {
        // Initialize the map
    	*map_final = *cloud_in;
    	
    }
    else
    {
        // 1.0 remove the Nan points
	// remove the NaN points and remove the points out of distance
	std::vector<int> indices;
	pcl::removeNaNFromPointCloud(*cloud_in, *cloud_in, indices);
	

	bool in_field = is_50_percent_in_field(cloud_in);
	
	if(  in_field == true  )
	{
		//return;
	}

	
	
	cout<<"=============50_percent================"<<endl;
	cout<<in_field<<endl;

	float x_min = -5.0f;
	float x_max = 5.0f;
	float y_min = -2.0f;
	float y_max = 2.0f;

	//filterPointCloud(cloud_in, cloud_in_cut, x_min, x_max, y_min, y_max);
	//*cloud_in = *cloud_in_cut;

	// 1.1 Voxel the cloud
	
	// Create the filtering object
	pcl::VoxelGrid<pcl::PointXYZ> sor_input;
	sor_input.setInputCloud (cloud_in);
	sor_input.setLeafSize (0.2f, 0.2f, 0.2f);
	sor_input.filter (*cloud_in_filtered);
	
	// 1.2 boxed the cloud
	*cloud_in_boxed_for_local = *cloud_in_filtered;
	
        pcl::CropBox<pcl::PointXYZ> boxFilter_for_in;
	float x_min_for_in = - 50, y_min_for_in = - 50, z_min_for_in = - 20 ;
	float x_max_for_in = + 50, y_max_for_in = + 50, z_max_for_in = + 50;
	
	boxFilter_for_in.setMin(Eigen::Vector4f(x_min_for_in, y_min_for_in, z_min_for_in, 1.0));
	boxFilter_for_in.setMax(Eigen::Vector4f(x_max_for_in, y_max_for_in, z_max_for_in, 1.0));

	boxFilter_for_in.setInputCloud(cloud_in_boxed_for_local);
	boxFilter_for_in.filter(*cloud_in_boxed_for_local);
	

	pcl::PointCloud<pcl::PointXYZ> Final;
	
	
	if( cloud_in_filtered->size() < 30 )
	{
		return;
	}

	
	cout<<"counter_gicp[0]="<<counter_gicp<<endl;
	
	if( counter_gicp < 2 )
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
	
		cout<<"counter_gicp[1]="<<counter_gicp<<endl;
		// ------------------------------------------5_line[start] ---------------------------------------------------// 
		auto t1_multi = std::chrono::high_resolution_clock::now();
		//clock_t start = clock(); // Get the current time in clock ticks
		

		int num_threads = 5; // Adjust based on your CPU and desired usage
		
		// Varying parameters for each thread
		std::vector<float> max_correspondence_dists = {0.05, 0.2, 0.4, 0.1, 0.3};
		std::vector<int> max_iterations = {30, 50, 70, 40, 60};
		std::vector<Eigen::Matrix4f> initial_guesses(num_threads, Eigen::Matrix4f::Identity());

		// Adding some small perturbations to initial guesses to make them different
		for (int i = 0; i < num_threads; ++i) {
			initial_guesses[i](0, 3) += 0.01f * i;  // Slight translation perturbation
			initial_guesses[i](1, 3) += 0.01f * i;
		}


		std::vector<std::thread> threads;
		std::vector<GICPResult> results(num_threads);

                cout<<"===============================runGICP=================================="<<endl;
		// Launch multiple GICP processes
		#pragma omp simd
		for (int i = 0; i < num_threads; ++i) {
			threads.emplace_back(runGICP, cloud_in_filtered, cloud_pre, std::ref(results[i]), max_correspondence_dists[i], max_iterations[i], initial_guesses[i]);

		}

		// Join all threads
		for (auto& thread : threads) {
			thread.join();
		}

		// Select the best GICP result
		GICPResult best_result = selectBestResult(results);

		// Output the best transformation
		std::cout << "Best Transformation Matrix:\n" << best_result.transformation << std::endl;
		std::cout << "Best Alignment Error: " << best_result.alignment_error << std::endl;

		
		//if( best_result.transformation(0,3) > 0.3 || best_result.transformation(1,3) > 0.3 || best_result.transformation(2,3) > 0.3 )
		//{
		//	return;
		//}

		//if(  best_result.alignment_error == 0.0 || in_field == true  )
		if(  best_result.alignment_error == 0.0 )
		{
			return;
		}
		
		Ti = best_result.transformation * Ti;



		//clock_t end = clock(); // Get the current time in clock ticks

		//double runningTime = (double)(end - start) / CLOCKS_PER_SEC; // Convert clock ticks to seconds

		//std::cout << "Running time: " << runningTime << " seconds" << std::endl;
//*/

		auto t2_multi = std::chrono::high_resolution_clock::now();
		double single_multi = std::chrono::duration_cast<std::chrono::nanoseconds>(t2_multi - t1_multi).count() / 1e6;
		std::cout << "single_multi:" << single_multi << "[msec] " << std::endl;

		// ------------------------------------------5_line[end] ---------------------------------------------------// 
	
	
	
	
	}
	//*/
	
	
	// 2.1 output the cloud_in_boxed_for_local and cloud_pre
	//brown is the input cloud of right now frame
	sensor_msgs::PointCloud2Ptr msg_second(new sensor_msgs::PointCloud2);
	//cout<<"==================cloud_in====================="<<endl;
	pcl::toROSMsg(*cloud_in_boxed_for_local, *msg_second);
	msg_second->header.stamp.fromSec(0);
	msg_second->header.frame_id = "world";
	pub_cloud_surround_.publish(msg_second);
	//*/

  
	
	
	
	// 3.0 voxed the map_final
	// Create the filtering object
	pcl::VoxelGrid<pcl::PointXYZ> sor;
	sor.setInputCloud (map_final);
	sor.setLeafSize (0.2f, 0.2f, 0.2f);
	sor.filter (*map_final);
	
	
	if( counter_stable_map % 20 == 0 )
	{
		// 3.1 boxed the map_final
		pcl::CropBox<pcl::PointXYZ> boxFilter;
		float x_min = Ti(0,3) - 50, y_min = Ti(1,3) - 50, z_min = Ti(2,3) - 20;
		float x_max = Ti(0,3) + 50, y_max = Ti(1,3) + 50, z_max = Ti(2,3) + 50;

		boxFilter.setMin(Eigen::Vector4f(x_min, y_min, z_min, 1.0));
		boxFilter.setMax(Eigen::Vector4f(x_max, y_max, z_max, 1.0));

		boxFilter.setInputCloud(map_final);
		boxFilter.filter(*map_final_boxed);
	}
	counter_stable_map++;
        //3.2 output the map_final boxed
	//red is the map_final
	pcl::toROSMsg(*map_final, *msg_second);
	msg_second->header.stamp.fromSec(0);
	msg_second->header.frame_id = "world";
	pub_icp_keyframes_.publish(msg_second);  
	
	//3.3 get the gicp of the cloud in boxed and the map boxed
	
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_boxed_translate_to_near_mapboxed (new pcl::PointCloud<pcl::PointXYZ>(5,1));
	pcl::transformPointCloud (*cloud_in_boxed_for_local, *cloud_in_boxed_translate_to_near_mapboxed, Ti);   

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_boxed_translate_to_near_mapboxed_far_icp (new pcl::PointCloud<pcl::PointXYZ>(5,1));
	pcl::transformPointCloud (*cloud_in_boxed_for_local, *cloud_in_boxed_translate_to_near_mapboxed_far_icp, Ti);   

	//blue is cloud_pre, the last frame
	pcl::toROSMsg(*cloud_in_boxed_translate_to_near_mapboxed, *msg_second);
	msg_second->header.stamp.fromSec(0);
	msg_second->header.frame_id = "world";
	pub_recent_keyframes_.publish(msg_second); 

	
	
	// Check if point clouds are populated
	if (cloud_in_boxed_translate_to_near_mapboxed_far_icp->empty())
	{
		std::cerr << "Source point cloud is empty!" << std::endl;
		return ;
	}

	if (map_final_boxed->empty())
	{
		std::cerr << "Target point cloud is empty!" << std::endl;
		return ;
	}


	std::cout << "[ map for fly ] Ti Transformation matrix:\n" << Ti << std::endl;
	
	auto t1 = std::chrono::high_resolution_clock::now();
	
	pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	Eigen::Matrix4f final_transformation = Eigen::Matrix4f::Identity();
	run_ICP( cloud_in_boxed_translate_to_near_mapboxed_far_icp, map_final_boxed,  initial_guess, aligned_cloud, final_transformation );
	std::cout << "[ map for fly ] run_ICP:\n" << final_transformation << std::endl;
	
	auto t2 = std::chrono::high_resolution_clock::now();
	double single = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
	std::cout << "single:" << single << "[msec] " << std::endl;
      
      

        auto t1_ndt = std::chrono::high_resolution_clock::now();
       
	//clock_t start_3 = clock(); // Get the current time in clock ticks

	// Step 1: Initial alignment using NDT
	pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
	ndt.setResolution(2.0);
	ndt.setInputSource(cloud_in_boxed_translate_to_near_mapboxed);
	ndt.setInputTarget(map_final_boxed);

	// NDT parameters
	ndt.setMaximumIterations(20);
	ndt.setStepSize(0.2);
	ndt.setTransformationEpsilon(0.05);

	// Initial guess for the transformation (identity matrix)
	Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();

	ndt.align(Final, final_transformation);
	
	
	// Perform the initial alignment
	//ndt.align(Final, icp.getFinalTransformation ());

	std::cout << "===============================ndt_and_icp===============================" << std::endl;
	// Check if NDT has converged
	if (ndt.hasConverged()) {
		std::cout << "NDT converged with score: " << ndt.getFitnessScore() << std::endl;
		std::cout << "NDT Transformation matrix:\n" << ndt.getFinalTransformation() << std::endl;
	} else {
		std::cout << "NDT did not converge." << std::endl;
		//return -1;
	}


	
	
	//clock_t end_3 = clock(); // Get the current time in clock ticks

	//double runningTime_3 = (double)(end_3 - start_3) / CLOCKS_PER_SEC; // Convert clock ticks to seconds

	//std::cout << "Running time: " << runningTime_3 << " seconds" << std::endl;

	auto t2_ndt = std::chrono::high_resolution_clock::now();
	double single_ndt = std::chrono::duration_cast<std::chrono::nanoseconds>(t2_ndt - t1_ndt).count() / 1e6;
	std::cout << "single_ndt:" << single_ndt << "[msec] " << std::endl;


	//cout<< ndt.getFinalTransformation()<<endl;
  //*/  


	cout<<"=================== cloud_in_filtered->size() ==================" <<endl;
	cout<<cloud_in_filtered->size() <<endl;



	auto t1_gicp = std::chrono::high_resolution_clock::now();
	
	pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp_for_map;
	gicp_for_map.setMaxCorrespondenceDistance(10.0);
	gicp_for_map.setTransformationEpsilon(0.01);
	gicp_for_map.setRotationEpsilon(0.01);
	gicp_for_map.setMaximumIterations(1000);
	
	gicp_for_map.setInputSource(cloud_in_boxed_translate_to_near_mapboxed);
	gicp_for_map.setInputTarget(map_final_boxed);
	//gicp_for_map.align(Final);
	
	gicp_for_map.align(Final, ndt.getFinalTransformation());
	//gicp_for_map.align(Final, transformation );
	
	Ti_of_map = gicp_for_map.getFinalTransformation (); // * Ti_of_map;

	Eigen::Matrix4f rotation_matrix = Ti_of_map;

	//double roll = atan2( rotationMatrix(2,1),rotationMatrix(2,2) )/3.1415926*180;
	//std::cout<<"roll is " << roll <<std::endl;
	//double pitch = atan2( -rotationMatrix(2,0), std::pow( rotationMatrix(2,1)*rotationMatrix(2,1) +rotationMatrix(2,2)*rotationMatrix(2,2) ,0.5  )  )/3.1415926*180;
	//std::cout<<"pitch is " << pitch <<std::endl;
	double yaw_of_cloud_ti_to_map = atan2( rotation_matrix(1,0),rotation_matrix(0,0) )/3.1415926*180;
	//std::cout<<"yaw is " << yaw_of_cloud_ti_to_map <<std::endl;
	
	Ti_real = Ti_of_map * Ti;

	
	
	std::cout << "Ti_of_map Transformation matrix:\n" << Ti_of_map << std::endl;

	auto t2_gicp = std::chrono::high_resolution_clock::now();
	double single_gicp = std::chrono::duration_cast<std::chrono::nanoseconds>(t2_gicp - t1_gicp).count() / 1e6;
	std::cout << "single_gicp:" << single_gicp << "[msec] " << std::endl;
	
	
	//*/
	
	
	
	if( abs( Ti_of_map(0,3) ) > 0.2 || abs( Ti_of_map(1,3) ) > 0.2 ||  abs( yaw_of_cloud_ti_to_map ) > 1)
	{
	        //cout<<"===========Ti_real=============="<<endl;
	        //cout<<Ti_of_map<<endl;
		//cout<<Ti<<endl;
		//cout<<Ti_real<<endl;
		Ti = Ti_real;
		

	}

	
	pcl::PointCloud<pcl::PointXYZ>::Ptr Final_cloud_translate (new pcl::PointCloud<pcl::PointXYZ>(5,1));
	pcl::transformPointCloud (*cloud_in_filtered, *Final_cloud_translate, Ti_real); 
	//*/

	

	
	if( counter_gicp % 5 == 0 )
        {
        
            
	    // 3.1 boxed the map_final
	    pcl::CropBox<pcl::PointXYZ> box_filter;
	    float x_min = Ti_real(0,3) - 50, y_min = Ti_real(1,3) - 50, z_min = Ti_real(2,3) - 50;
	    float x_max = Ti_real(0,3) + 50, y_max = Ti_real(1,3) + 50, z_max = Ti_real(2,3) + 50;

	    box_filter.setMin(Eigen::Vector4f(x_min, y_min, z_min, 1.0));
	    box_filter.setMax(Eigen::Vector4f(x_max, y_max, z_max, 1.0));

	    box_filter.setInputCloud(map_final);
	    box_filter.filter(*map_final_boxed_2);
	    //*/
        
	    pcl::PointCloud<pcl::PointXYZ> Final_for_add_to_map;
	    
	    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp_for_add_to_map_final;
	    gicp_for_add_to_map_final.setMaxCorrespondenceDistance(5.0);
	    gicp_for_add_to_map_final.setTransformationEpsilon(0.001);
	    gicp_for_add_to_map_final.setMaximumIterations(1000);

	    gicp_for_add_to_map_final.setInputSource(Final_cloud_translate);
	    gicp_for_add_to_map_final.setInputTarget(map_final_boxed_2);
	    gicp_for_add_to_map_final.align(Final_for_add_to_map);
	    
            
            
            if( abs( gicp_for_add_to_map_final.getFinalTransformation ()(0,3) ) < 1 && abs( gicp_for_add_to_map_final.getFinalTransformation ()(1,3) ) < 1 ) 
            {
            
		Ti = gicp_for_add_to_map_final.getFinalTransformation () * Ti;
		
		*map_final += Final_for_add_to_map;
		
		pcl::toROSMsg(Final_for_add_to_map, *msg_second);
		msg_second->header.stamp.fromSec(0);
		msg_second->header.frame_id = "world";
		pub_history_keyframes_.publish(msg_second);   
		
		/*
		Eigen::Matrix4f Ti_for_submap_start = gicp_for_map.getFinalTransformation (); 
		Eigen::Matrix4f rotation_matrix = Ti_for_submap_start;

		double yaw_of_cloud_ti_to_map = atan2( rotation_matrix(1,0),rotation_matrix(0,0) )/3.1415926*180;
		
		//cout<<"yaw_of_cloud_ti_to_map="<<yaw_of_cloud_ti_to_map<<endl;
		
		if(  yaw_of_cloud_ti_to_map < 0.1  && abs( gicp_for_add_to_map_final.getFinalTransformation ()(0,3) ) < 0.1 && abs( gicp_for_add_to_map_final.getFinalTransformation ()(1,3) ) < 0.1 ) 
		{
			is_a_accurate_Ti = true;
		}
		//*/
		
		
	    }   
	}
    }
     
    counter_gicp++;
    
    *cloud_pre = *cloud_in_filtered;
  
  //*/   
  

    //*/
  
    double time_laser_odom_;
    nav_msgs::OdometryPtr msg_2(new nav_msgs::Odometry);
    msg_2->header.stamp.fromSec(time_laser_odom_);
    msg_2->header.frame_id = "world";
    msg_2->child_frame_id = "/laser";
    msg_2->pose.pose.position.x = Ti_real(0,3);
    msg_2->pose.pose.position.y = Ti_real(1,3);
    msg_2->pose.pose.position.z = Ti_real(2,3);
    msg_2->pose.pose.orientation.w = 1;
    msg_2->pose.pose.orientation.x = 0;
    msg_2->pose.pose.orientation.y = 0;
    msg_2->pose.pose.orientation.z = 0;
    pub_odom_aft_mapped_2.publish(msg_2);


}

  

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    nh.param<double>("time_offset", lidar_time_offset, 0.0);

    nh.param<bool>("publish/path_en",path_en, true);
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);
    nh.param<string>("common/lid_topic",lid_topic,"/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);

    // mapping algorithm params
    nh.param<float>("mapping/det_range",DET_RANGE,300.f);
    nh.param<int>("mapping/max_iteration", NUM_MAX_ITERATIONS, 4);
    nh.param<int>("mapping/max_points_size", max_points_size, 100);
    nh.param<int>("mapping/max_cov_points_size", max_cov_points_size, 100);
    nh.param<vector<double>>("mapping/layer_point_size", layer_point_size,vector<double>());
    nh.param<int>("mapping/max_layer", max_layer, 2);
    nh.param<double>("mapping/voxel_size", max_voxel_size, 1.0);
    nh.param<double>("mapping/down_sample_size", filter_size_surf_min, 0.5);
    std::cout << "filter_size_surf_min:" << filter_size_surf_min << std::endl;
    nh.param<double>("mapping/plannar_threshold", min_eigen_value, 0.01);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());

    // noise model params
    nh.param<double>("noise_model/ranging_cov", ranging_cov, 0.02);
    nh.param<double>("noise_model/angle_cov", angle_cov, 0.05);
    nh.param<double>("noise_model/gyr_cov",gyr_cov,0.1);
    nh.param<double>("noise_model/acc_cov",acc_cov,0.1);
    nh.param<double>("noise_model/b_gyr_cov",b_gyr_cov,0.0001);
    nh.param<double>("noise_model/b_acc_cov",b_acc_cov,0.0001);

    // visualization params
    nh.param<bool>("publish/pub_voxel_map", publish_voxel_map, false);
    nh.param<int>("publish/publish_max_voxel_layer", publish_max_voxel_layer, 0);

    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num, 1);
    nh.param<bool>("preprocess/feature_extract_enable", p_pre->feature_enabled, false);
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;
    for (int i = 0; i < layer_point_size.size(); i++) {
        layer_size.push_back(layer_point_size[i]);
    }

    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";

    /*** variables definition ***/
    int effect_feat_num = 0, frame_num = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    // XXX 暂时现在lidar callback中固定转换到IMU系下
    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, observation_model_share, NUM_MAX_ITERATIONS, epsi);

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 2, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 2, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 50, imu_cbk);
    
    
    
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>
            ("/Odometry", 100000);
    ros::Publisher pubExtrinsic = nh.advertise<nav_msgs::Odometry>
            ("/Extrinsic", 100000);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path>
            ("/path", 100000);
    ros::Publisher voxel_map_pub =
            nh.advertise<visualization_msgs::MarkerArray>("/planes", 10000);
            
    ros::Publisher pub_cloud_store = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_store", 100000);
//------------------------------------------------------------------------------------------------------     
    ros::Subscriber sub_pc_ = nh.subscribe<sensor_msgs::PointCloud2>("/cari_points_top", 10, main_callback);
    pub_history_keyframes_ = nh.advertise<sensor_msgs::PointCloud2>("/history_keyframes", 10);
    pub_recent_keyframes_ = nh.advertise<sensor_msgs::PointCloud2>("/recent_keyframes", 10);
    pub_icp_keyframes_ = nh.advertise<sensor_msgs::PointCloud2>("/icp_keyframes", 10);
    pub_cloud_surround_ = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 10);
 
    pub_odom_aft_mapped_2 = nh.advertise<nav_msgs::Odometry>("/odom_aft_mapped_2", 10);

    direction_pub = nh.advertise<geometry_msgs::Vector3>("/tunnel_longest_direction", 10);
    marker_pub = nh.advertise<visualization_msgs::Marker>("/visualization_marker", 10);
    eigenvalue_pub = nh.advertise<std_msgs::Float64MultiArray>("/eigenvalues", 10);
    marker_pub_Hessien = nh.advertise<visualization_msgs::Marker>("/covariance_marker", 10);
    
    
    
    
    mean_dir_pub = nh.advertise<geometry_msgs::Vector3>("mean_longest_direction", 10);
    std_dir_pub = nh.advertise<geometry_msgs::Vector3>("std_longest_direction", 10);

    


    
     map_final.reset(new pcl::PointCloud<pcl::PointXYZ>);
     map_final_boxed.reset(new pcl::PointCloud<pcl::PointXYZ>);
     map_final_boxed_2.reset(new pcl::PointCloud<pcl::PointXYZ>);
     map_final_boxed_cov.reset(new pcl::PointCloud<pcl::PointXYZ>);
     Final.reset(new pcl::PointCloud<pcl::PointXYZ>);
     cloud_pre.reset(new pcl::PointCloud<pcl::PointXYZ>);
     cloud_in_filtered.reset(new pcl::PointCloud<pcl::PointXYZ>);
     cloud_in_boxed_for_local.reset(new pcl::PointCloud<pcl::PointXYZ>);

     Ti = Eigen::Matrix4f::Identity ();     
     Ti_real = Eigen::Matrix4f::Identity ();
     Ti_of_map = Eigen::Matrix4f::Identity ();
     T = Eigen::Matrix4d::Identity();
     counter_loop = 0;
     counter_stable_map = 0;

//------------------------------------------------------------------------------------------------------
    // for Plane Map
    bool init_map = false;

    double sum_optimize_time = 0, sum_update_time = 0;
    int scan_index = 0;

    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit) break;
        
        ros::spinOnce();
        
        if(sync_packages(Measures))
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }
	    
	    // [KEY_1], get the path lidar data
            p_imu->Process(Measures, kf, feats_undistort);
    
    
            vector<Pose6D> IMUpose = p_imu->IMUpose;

            //cout<<"================IMUpose.size()====================="<<endl;
            //cout<<IMUpose.size()<<endl;

            if(IMUpose.size()>10)
            {
                  auto it_kp = IMUpose.end() - 1;
                  auto head = it_kp - 1;
      
                  M3D R_imu;
                  R_imu << MAT_FROM_ARRAY(head->rot);
                  //cout<<R_imu<<endl;
      
      
                  pos_imu << VEC_FROM_ARRAY(head->pos);
                  //cout<<pos_imu<<endl;
      
                  V3D vel_imu;
                  vel_imu << VEC_FROM_ARRAY(head->vel);
                  //cout<<vel_imu<<endl;
      
            }
 
            
            
            
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            // ===============================================================================================================
            // 第一帧 如果ekf初始化了 就初始化voxel地图
            
            // [STEP_1], build the initile map
            if (flg_EKF_inited && !init_map) 
            {
                PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI);
                transformLidar(state_point, feats_undistort, world_lidar);
                std::vector<pointWithCov> pv_list;

                std::cout << kf.get_P() << std::endl;
                // 计算第一帧所有点的covariance 并用于构建初始地图
                for (size_t i = 0; i < world_lidar->size(); i++) {
                    pointWithCov pv;
                    pv.point << world_lidar->points[i].x, world_lidar->points[i].y,
                            world_lidar->points[i].z;
                    V3D point_this(feats_undistort->points[i].x,
                                   feats_undistort->points[i].y,
                                   feats_undistort->points[i].z);
                    // if z=0, error will occur in calcBodyCov. To be solved
                    if (point_this[2] == 0) {
                        point_this[2] = 0.001;
                    }
                    M3D cov_lidar = calcBodyCov(point_this, ranging_cov, angle_cov);
                    // 转换到world系
                    M3D cov_world = transformLiDARCovToWorld(point_this, kf, cov_lidar);

                    pv.cov = cov_world;
                    pv_list.push_back(pv);
                    Eigen::Vector3d sigma_pv = pv.cov.diagonal();
                    sigma_pv[0] = sqrt(sigma_pv[0]);
                    sigma_pv[1] = sqrt(sigma_pv[1]);
                    sigma_pv[2] = sqrt(sigma_pv[2]);
                }

                buildVoxelMap(pv_list, max_voxel_size, max_layer, layer_size,
                              max_points_size, max_points_size, min_eigen_value,
                              voxel_map);
                std::cout << "build voxel map" << std::endl;

                if (publish_voxel_map) {
                    pubVoxelMap(voxel_map, publish_max_voxel_layer, voxel_map_pub);
                    publish_frame_world(pubLaserCloudFull);
                    publish_frame_body(pubLaserCloudFull_body);
                }
                init_map = true;
            }

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list);

            feats_down_size = feats_down_body->points.size();
            
	    cout<<"==================feats_down_size====================="<<endl;
	    cout<<feats_down_size<<endl;
            
            
            
            
            // 由于点云的body var是一直不变的 因此提前计算 在迭代时可以复用
            var_down_body.clear();
            for (auto & pt:feats_down_body->points) {
                V3D point_this(pt.x, pt.y, pt.z);
                var_down_body.push_back(calcBodyCov(point_this, ranging_cov, angle_cov));
            }

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 30)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }
            

            PointCloudXYZI::Ptr world_lidar_point(new PointCloudXYZI);
            transformLidar(state_point, feats_down_body, world_lidar_point);
            
            /*
            
            // Create a new PointXYZ cloud to store the converted points
            pcl::PointCloud<pcl::PointXYZ>::Ptr world_lidar_point_XYZ(new pcl::PointCloud<pcl::PointXYZ>);


            auto t1_XYZ = std::chrono::high_resolution_clock::now();

            
            // Convert PointXYZI to PointXYZ
            for (const auto& point : world_lidar_point->points) {
                        pcl::PointXYZ new_point;
                        new_point.x = point.x;
                        new_point.y = point.y;
                        new_point.z = point.z;
                        world_lidar_point_XYZ->points.push_back(new_point);
            }

            // Create a new PointXYZ cloud to store the converted points
            pcl::PointCloud<pcl::PointXYZ>::Ptr feats_down_body_XYZ(new pcl::PointCloud<pcl::PointXYZ>);

            // Convert PointXYZI to PointXYZ
            for (const auto& point : feats_down_body->points) {
                        pcl::PointXYZ new_point;
                        new_point.x = point.x;
                        new_point.y = point.y;
                        new_point.z = point.z;
                        feats_down_body_XYZ->points.push_back(new_point);
            }
            
            //*/

            //std::thread worker([=]() { processPointCloud(world_lidar_point_XYZ, feats_down_body_XYZ); });

	    
	    //--------------------------------------------------------------------------------------------backup_system[start]--------------------------------------------------------------------------------------//
            /*
            double t_update_start_gicp = omp_get_wtime();
            
	    if( counter_loop == 0 )
	    {
		// Initialize the map
		*map_final = *world_lidar_point;
		*cloud_pre = *feats_down_body;
		counter_loop++;
	    }
	    else
	    {

		// 1.0 remove the Nan points
		// remove the NaN points and remove the points out of distance
		std::vector<int> indices;
		pcl::removeNaNFromPointCloud(*feats_down_body, *feats_down_body, indices);


		// 1.1 Voxel the cloud

		// Create the filtering object
		pcl::VoxelGrid<pcl::PointXYZINormal> sor_input;
		sor_input.setInputCloud (feats_down_body);
		sor_input.setLeafSize (0.1f, 0.1f, 0.1f);
		sor_input.filter (*cloud_in_filtered);

		
		// 1.2 boxed the cloud
		*cloud_in_boxed_for_local = *cloud_in_filtered;

		pcl::CropBox<pcl::PointXYZINormal> boxFilter_for_in;
		float x_min_for_in = - 50, y_min_for_in = - 50, z_min_for_in = - 0 ;
		float x_max_for_in = + 50, y_max_for_in = + 50, z_max_for_in = + 50;

		boxFilter_for_in.setMin(Eigen::Vector4f(x_min_for_in, y_min_for_in, z_min_for_in, 1.0));
		boxFilter_for_in.setMax(Eigen::Vector4f(x_max_for_in, y_max_for_in, z_max_for_in, 1.0));

		boxFilter_for_in.setInputCloud(cloud_in_boxed_for_local);
		boxFilter_for_in.filter(*cloud_in_boxed_for_local);
		

		//pcl::PointCloud<pcl::PointXYZINormal> Final;

            
		// 2.0 gicp of the cloud_in voxeled and pre cloud
		pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZINormal, pcl::PointXYZINormal> gicp;
		gicp.setMaxCorrespondenceDistance(1.0);
		gicp.setTransformationEpsilon(0.01);
		gicp.setMaximumIterations(500);

		gicp.setInputSource(cloud_in_filtered);
		gicp.setInputTarget(cloud_pre);
		gicp.align(*Final);
		Ti = gicp.getFinalTransformation () * Ti;
		
		cout<<"==============================Ti========================="<<endl;
		cout<<Ti<<endl;            
               
               
               
		// 3.0 voxed the map_final
		// Create the filtering object
		pcl::VoxelGrid<pcl::PointXYZINormal> sor;
		sor.setInputCloud (map_final);
		sor.setLeafSize (0.1f, 0.1f, 0.1f);
		sor.filter (*map_final);
	
		if( counter_stable_map % 20 == 0 )
		{
			// 3.1 boxed the map_final
			pcl::CropBox<pcl::PointXYZINormal> boxFilter;
			float x_min = Ti(0,3) - 50, y_min = Ti(1,3) - 50, z_min = Ti(2,3) - 0;
			float x_max = Ti(0,3) + 50, y_max = Ti(1,3) + 50, z_max = Ti(2,3) + 50;

			boxFilter.setMin(Eigen::Vector4f(x_min, y_min, z_min, 1.0));
			boxFilter.setMax(Eigen::Vector4f(x_max, y_max, z_max, 1.0));

			boxFilter.setInputCloud(map_final);
			boxFilter.filter(*map_final_boxed);
		}
		
		counter_stable_map++;
               
               
		//3.3 get the gicp of the cloud in boxed and the map boxed

		pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_in_boxed_translate_to_near_mapboxed (new pcl::PointCloud<pcl::PointXYZINormal>);
		pcl::transformPointCloud (*cloud_in_boxed_for_local, *cloud_in_boxed_translate_to_near_mapboxed, Ti); 
               
               
		pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZINormal, pcl::PointXYZINormal> gicp_for_map;
		gicp_for_map.setMaxCorrespondenceDistance(5.0);
		gicp_for_map.setTransformationEpsilon(0.05);
		gicp_for_map.setRotationEpsilon(0.05);
		gicp_for_map.setMaximumIterations(500);

		gicp_for_map.setInputSource(cloud_in_boxed_translate_to_near_mapboxed);
		gicp_for_map.setInputTarget(map_final_boxed);
		gicp_for_map.align(*Final);

		Ti_of_map = gicp_for_map.getFinalTransformation (); // * Ti_of_map;
               
               Ti_real = Ti_of_map * Ti;
               
		cout<<"==============================Ti_real========================="<<endl;
		cout<<Ti_real<<endl;   
		
		
		Eigen::Matrix4f rotation_matrix = Ti_of_map;
		double yaw_of_cloud_ti_to_map = atan2( rotation_matrix(1,0),rotation_matrix(0,0) )/3.1415926*180;
		
		if( abs( Ti_of_map(0,3) ) > 0.2 || abs( Ti_of_map(1,3) ) > 0.2 ||  abs( yaw_of_cloud_ti_to_map ) > 1)
		{
			//cout<<"===========Ti_real=============="<<endl;
			//cout<<Ti_of_map<<endl;
			//cout<<Ti<<endl;
			//cout<<Ti_real<<endl;
			Ti = Ti_real;


		}
		
		pcl::PointCloud<pcl::PointXYZINormal>::Ptr Final_cloud_translate (new pcl::PointCloud<pcl::PointXYZINormal>(5,1));
		pcl::transformPointCloud (*cloud_in_filtered, *Final_cloud_translate, Ti_real); 

		if( counter_loop % 20 == 0 )
		{
			//*map_final += *world_lidar_point; 
			

			// 3.1 boxed the map_final
			pcl::CropBox<pcl::PointXYZINormal> box_filter;
			float x_min = Ti_real(0,3) - 50, y_min = Ti_real(1,3) - 50, z_min = Ti_real(2,3) - 50;
			float x_max = Ti_real(0,3) + 50, y_max = Ti_real(1,3) + 50, z_max = Ti_real(2,3) + 50;

			box_filter.setMin(Eigen::Vector4f(x_min, y_min, z_min, 1.0));
			box_filter.setMax(Eigen::Vector4f(x_max, y_max, z_max, 1.0));

			box_filter.setInputCloud(map_final);
			box_filter.filter(*map_final_boxed_2);
			

			pcl::PointCloud<pcl::PointXYZINormal> Final_for_add_to_map;

			pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZINormal, pcl::PointXYZINormal> gicp_for_add_to_map_final;
			gicp_for_add_to_map_final.setMaxCorrespondenceDistance(5.0);
			gicp_for_add_to_map_final.setTransformationEpsilon(0.05);
			gicp_for_add_to_map_final.setMaximumIterations(500);

			gicp_for_add_to_map_final.setInputSource(Final_cloud_translate);
			gicp_for_add_to_map_final.setInputTarget(map_final_boxed_2);
			gicp_for_add_to_map_final.align(Final_for_add_to_map);



			if( abs( gicp_for_add_to_map_final.getFinalTransformation ()(0,3) ) < 0.2 && abs( gicp_for_add_to_map_final.getFinalTransformation ()(1,3) ) < 0.2 ) 
			{

				Ti = gicp_for_add_to_map_final.getFinalTransformation () * Ti;

				*map_final += Final_for_add_to_map;
				

			} 
			
		}
               
	    	counter_loop++;
            }
            
            *cloud_pre = *cloud_in_filtered;
            
            
            double time_laser_odom_;
            nav_msgs::OdometryPtr msg_2(new nav_msgs::Odometry);
            msg_2->header.stamp.fromSec(time_laser_odom_);
            msg_2->header.frame_id = "world";
            msg_2->child_frame_id = "/laser";
            msg_2->pose.pose.position.x = Ti_real(0,3);
            msg_2->pose.pose.position.y = Ti_real(1,3);
            msg_2->pose.pose.position.z = Ti_real(2,3);
            msg_2->pose.pose.orientation.w = 1;
            msg_2->pose.pose.orientation.x = 0;
            msg_2->pose.pose.orientation.y = 0;
            msg_2->pose.pose.orientation.z = 0;
            pub_odom_aft_mapped_2.publish(msg_2);
            
            
            double t_update_end_gicp = omp_get_wtime();
            double optimize_time_gicp = t_update_start_gicp - t_update_end_gicp;
            
            cout<<"optimize_time_gicp="<<optimize_time_gicp<<endl;
            //*/
            //--------------------------------------------------------------------------------------------backup_system[end]--------------------------------------------------------------------------------------//
            
            // [KEY_2], least square
            
            // ===============================================================================================================
            // 开始迭代滤波
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_diagonal();
//            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            double t_update_end = omp_get_wtime();
            sum_optimize_time += t_update_end - t_update_start;

            state_point = kf.get_x();
            
            //cout<<"========================v======================"<<endl;      
            //cout<<state_point.vel<<endl;
            if( abs(state_point.vel[0]) > 0.2 || abs(state_point.vel[1]) > 0.2 || abs(state_point.vel[2]) > 0.2 )
            {
            	 //cout<<"==========vel_execl============"<<endl;
            	 state_point.pos = pos_imu;
                 state_point.vel << 0.0, 0.0, 0.0;
                 //state_point.pos[2] = 0;
		 kf.change_x(state_point);
                
                not_update_vexolmap = true;
                
            }
	    /*
	    //if( ( scales[0] / scales[1] < 0.7 ) || ( scales[0] / scales[2] < 0.7 ) )
	    {

		M3D R_imu;
		R_imu 	= Ti_real.block<3, 3>(0, 0).cast<double>();            
		pos_imu 	= Ti_real.block<3, 1>(0, 3).cast<double>();

		state_point.rot = R_imu;
		state_point.pos = pos_imu;

		//kf.change_x(state_point);
            }
            //*/
            //cout<<"========================Eigen::Matrix4d_T[0]======================"<<endl;  
            
            Eigen::Matrix3d R = state_point.rot.toRotationMatrix();
            
            //Eigen::Quaternionf quaternion = state_point.rot;
            //Eigen::Matrix3f R = quaternion.toRotationMatrix();

            
            T.block<3, 3>(0, 0) = R;
            //T.block<3, 1>(0, 3) = Eigen::Vector3d(state_point.pos(0), state_point.pos(1), state_point.pos(2));
            
            Eigen::Vector3d p_lidar(0, 0, 0);
            V3D p_body = state_point.rot * (state_point.offset_R_L_I * p_lidar + state_point.offset_T_L_I) + state_point.pos;
            T.block<3, 1>(0, 3) = Eigen::Vector3d(p_body(0), p_body(1), p_body(2));
           
            
            
            //cout<<"========================Eigen::Matrix4d_T[1]======================"<<endl;  
            //cout<<T<<endl;
                        
            
            
            // rotaton to euler angle
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];
//
            //std::printf("BA: %.4f %.4f %.4f   BG: %.4f %.4f %.4f   g: %.4f %.4f %.4f\n",
            //            kf.get_x().ba.x(),kf.get_x().ba.y(),kf.get_x().ba.z(),
            //            kf.get_x().bg.x(),kf.get_x().bg.y(),kf.get_x().bg.z(),
            //            kf.get_x().grav.get_vect().x(), kf.get_x().grav.get_vect().y(), kf.get_x().grav.get_vect().z()
            //);


	    // [STEP_2], update the initile map
	    
            // ===============================================================================================================
            // 更新地图
            /*** add the points to the voxel map ***/
            // 用最新的状态估计将点及点的covariance转换到world系
            std::vector<pointWithCov> pv_list;
            PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI);
            transformLidar(state_point, feats_down_body, world_lidar);

            //*map_final += *world_lidar; 

            
            
            for (size_t i = 0; i < feats_down_body->size(); i++) {
                // 保存body系和world系坐标
                pointWithCov pv;
                pv.point << feats_down_body->points[i].x, feats_down_body->points[i].y, feats_down_body->points[i].z;
                // 计算lidar点的cov
                // FIXME 这里错误的使用世界系的点来calcBodyCov时 反倒在某些seq（比如hilti2022的03 15）上效果更好 需要考虑是不是init_plane时使用更大的cov更好
                // 注意这个在每次迭代时是存在重复计算的 因为lidar系的点云covariance是不变的
                // M3D cov_lidar = calcBodyCov(pv.point, ranging_cov, angle_cov);
                M3D cov_lidar = var_down_body[i];
                // 将body系的var转换到world系
                M3D cov_world = transformLiDARCovToWorld(pv.point, kf, cov_lidar);

                // 最终updateVoxelMap需要用的是world系的point
                pv.cov = cov_world;
                pv.point << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
                pv_list.push_back(pv);
            }


            
            //if( not_update_vexolmap == true )
            //{
            //	not_update_vexolmap == false;
            //}
            //else
            //if( abs(state_point.vel[0]) < 0.3 || abs(state_point.vel[1]) < 0.3 || abs(state_point.vel[2]) < 0.3 )
            {
		    t_update_start = omp_get_wtime();
		    std::sort(pv_list.begin(), pv_list.end(), var_contrast);
		    updateVoxelMapOMP(pv_list, max_voxel_size, max_layer, layer_size,
		                   max_points_size, max_points_size, min_eigen_value,
		                   voxel_map);
		    t_update_end = omp_get_wtime();
		    sum_update_time += t_update_end - t_update_start;

		    scan_index++;
		    std::printf("Mean  Topt: %.5fs   Tu: %.5fs\n", sum_optimize_time / scan_index, sum_update_time / scan_index);
            }
            // ===============================================================================================================
            // 可视化相关的shit
            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);
//
//            /*** add the feature points to map kdtree ***/
//            map_incremental();
//
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            if (scan_pub_en)      publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
            if (publish_voxel_map) {
                pubVoxelMap(voxel_map, publish_max_voxel_layer, voxel_map_pub);
            }
            // publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);

            if (scan_pub_en)
                publish_cloud_store(pub_cloud_store);
            
            //worker.join();
            
        }

        status = ros::ok();
        rate.sleep();
    }

    // Wait for the thread to finish
    //worker.join();


    return 0;
}
