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
#include "preprocess.h"
#include "voxel_map_util.hpp"
// LZLZLZ
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <std_msgs/String.h>
#include <cstdlib>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h> //使用OMP需要添加的头文件
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/registration/gicp.h>
#include <pcl/common/transforms.h>

// LZLZLZ
#define INIT_TIME (0.1)
#define LASER_POINT_COV (0.001)
#define MAXN (720000)
#define PUBFRAME_PERIOD (20)

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


#include <stable_gicp/gicp/stable_gicp.hpp>
#include <stable_gicp/gicp/stable_gicp_st.hpp>

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

//#include <Exp_mat.h>




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
    
    //cout<<"========================runGICP========================="<<endl;
    
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

    // Initialize fgicp_mt object (multi-threaded )
    stable_gicp::StableGICP<pcl::PointXYZ, pcl::PointXYZ>::Ptr stable_gicp_speed(new stable_gicp::StableGICP<pcl::PointXYZ, pcl::PointXYZ>());

    // Set number of threads for multi-threading
    stable_gicp_speed->setNumThreads(1); // Adjust number of threads as needed
    stable_gicp_speed->setMaxCorrespondenceDistance(max_correspondence_dist); // Set maximum correspondence distance
    stable_gicp_speed->setMaximumIterations(max_iterations); // Set maximum iterations

    // Set input source and target clouds
    stable_gicp_speed->setInputSource(source);
    stable_gicp_speed->setInputTarget(target);

    // Align the source cloud to the target cloud
    //pcl::PointCloud<pcl::PointXYZ> aligned(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ> aligned;
    stable_gicp_speed->align(aligned);

    // Local results to avoid locking during the whole process
    Eigen::Matrix4f transformation;
    double alignment_error;
    bool converged = stable_gicp_speed->hasConverged();

    if (converged) {
        transformation = stable_gicp_speed->getFinalTransformation();
        alignment_error = stable_gicp_speed->getFitnessScore();

        std::cout << "============= Stable GICP =============" << std::endl;
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






/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
bool time_sync_en = false, extrinsic_est_en = true, path_en = true;
double lidar_time_offset = 0.0;
/**************************/

// float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string lid_topic, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_surf_min = 0;
double lidar_end_time = 0, first_lidar_time = 0.0;
int effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_index = 0;
// bool point_selected_surf[100000] = {0};
bool lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

vector<double> extrinT(3, 0.0);
vector<double> extrinR(9, 0.0);
deque<double> time_buffer;
deque<PointCloudXYZI::Ptr> lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

// PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
std::vector<M3D> var_down_body;

pcl::VoxelGrid<PointType> downSizeFilterSurf;

V3D euler_cur;
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

// params for voxel mapping algorithm
double min_eigen_value = 0.003;
int max_layer = 0;

int max_cov_points_size = 50;
int max_points_size = 50;
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

PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());

// LZLZLZ
bool pcd_save_en = false, Init_coordinates = false;
int pcd_save_interval = -1;
double pos_save[6] = {0.0};      // 数据保存触发条件 xyz大于0.5m 角度大于10
double pos_save_temp[6] = {0.0}; // 数据保存触发条件 xyz大于0.5m 角度大于10
double Vel_threshold = 1.0;
double imu_ang_vel = 0.0;

sensor_msgs::PointCloud2::Ptr msg_old;
int min_lidar_points = 50; //
bool first_lidar_flg = true;
bool lidar_ok = true;

float cloud_storeRadius = 0.25; // 半径 publish_cloud_store()
float cloud_store_Xmin = -10.0; // publish_cloud_store()
float cloud_store_Xmax = 10.0;  // publish_cloud_store()
float cloud_store_Ymin = -5.0;  // publish_cloud_store()
float cloud_store_Ymax = 5.0;   // publish_cloud_store()
PointCloudXYZI::Ptr laserCloudWorld_LZ(new PointCloudXYZI());
// LZ add



/*
pcl::PointCloud<pcl::PointXYZ>::Ptr map_final (new pcl::PointCloud<pcl::PointXYZ>(5,1));
pcl::PointCloud<pcl::PointXYZ>::Ptr map_final_boxed (new pcl::PointCloud<pcl::PointXYZ>(5,1));  
pcl::PointCloud<pcl::PointXYZ>::Ptr map_final_boxed_2 (new pcl::PointCloud<pcl::PointXYZ>(5,1)); 
pcl::PointCloud<pcl::PointXYZ>::Ptr map_final_boxed_cov (new pcl::PointCloud<pcl::PointXYZ>(5,1)); 
pcl::PointCloud<pcl::PointXYZ>::Ptr Final (new pcl::PointCloud<pcl::PointXYZ>(5,1));
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pre (new pcl::PointCloud<pcl::PointXYZ>(5,1));
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_filtered (new pcl::PointCloud<pcl::PointXYZ>(5,1));
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_boxed_for_local (new pcl::PointCloud<pcl::PointXYZ>(5,1));
//*/


PointCloudXYZI::Ptr map_final;
PointCloudXYZI::Ptr map_final_boxed;  
PointCloudXYZI::Ptr map_final_boxed_2; 
PointCloudXYZI::Ptr map_final_boxed_cov; 
PointCloudXYZI::Ptr Final;
PointCloudXYZI::Ptr cloud_pre;
PointCloudXYZI::Ptr cloud_in_filtered;
PointCloudXYZI::Ptr cloud_in_boxed_for_local;


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

const bool var_contrast(pointWithCov &x, pointWithCov &y)
{
    return (x.cov.diagonal().norm() < y.cov.diagonal().norm());
};

void pointBodyToWorld_ikfom(PointType const *const pi, PointType *const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void pointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    //    po->intensity = pi->intensity;
}

template <typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

// LZLZ
void RGBpointBodyToWorld_LZ(pcl::PointXYZRGB const *const pi, pcl::PointXYZRGB *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->r = pi->r;
    po->g = pi->g;
    po->b = pi->b;
}

// LZLZ
////LZLZ  gicp

void gicp_pcl(PointCloudXYZI::Ptr target, PointCloudXYZI::Ptr source, Eigen::Matrix4f &transfor)
{
    auto t1 = std::chrono::high_resolution_clock::now();
    pcl::PointCloud<pcl::PointXYZ>::Ptr PCloud_t(new pcl::PointCloud<pcl::PointXYZ>);
    if (target->size() < 10 || source->size() < 10)
        return;
    for (int i = 0; i < target->size(); i += 2)
    {
        PointType const *const p = &target->points[i];
        PointType p_world;
        pointBodyToWorld(p, &p_world);
        pcl::PointXYZ pointXYZ;
        // Copy the XYZ coordinates from pointXYZI to pointXYZ
        pointXYZ.x = p_world.x;
        pointXYZ.y = p_world.y;
        pointXYZ.z = p_world.z;
        // Add the point to the new cloud
        PCloud_t->points.push_back(pointXYZ);
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr PCloud_s(new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0; i < source->size(); i += 2)
    {
        PointType p_world = source->points[i];
        pcl::PointXYZ pointXYZ;
        pointXYZ.x = p_world.x;
        pointXYZ.y = p_world.y;
        pointXYZ.z = p_world.z;
        PCloud_s->points.push_back(pointXYZ);
    }

    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> pcl_gicp;
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);

    double fitness_score = 1e38;
    pcl_gicp.setMaximumIterations(5);
    pcl_gicp.setInputTarget(PCloud_t);
    pcl_gicp.setInputSource(PCloud_s); // source
    pcl_gicp.align(*aligned);
    auto t2 = std::chrono::high_resolution_clock::now();
    fitness_score = pcl_gicp.getFitnessScore();
    double single = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
    std::cout << "gicp time:" << single << "[msec]" << " Converged " << pcl_gicp.hasConverged() << std::endl;
    // Eigen::Matrix4f transfor;
    transfor = pcl_gicp.getFinalTransformation();
    Eigen::Matrix3d r_matrix = Eigen::Matrix3d::Identity();
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            r_matrix(i, j) = transfor(i, j);
        }
    }
    // Eigen::Quaterniond quat(r_matrix);
    // Eigen::Vector3d rpy = quat.matrix().eulerAngles(2, 1, 0) * 180.0 / 3.1415926;
    std::cout << target->size() << "   " << source->size() << std::endl;
    std::cout << pcl_gicp.getFinalTransformation() << std::endl;
}

////

void RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;

    po->curvature = pi->curvature;
    po->normal_x = pi->normal_x;
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{

    //cout<<"==========================standard_pcl_cbk[0]================================"<<endl;
    if ((msg->width * msg->height) < min_lidar_points)
    {
        lidar_ok = false;
        return;
    }
    else
    {

        first_lidar_flg = false;
    }
    if (first_lidar_flg)
    {
        lidar_ok = false;
        return;
    }
    auto time_offset = lidar_time_offset;
    mtx_buffer.lock();
    // printf("lidar time: %17.3lf\n", msg->header.stamp.toSec());
    //  scan_count++;
    //  double preprocess_start_time = omp_get_wtime();
    
    //cout<<"==========================standard_pcl_cbk[1]================================"<<endl;
    
    if (msg->header.stamp.toSec() + time_offset < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    
    //cout<<"==========================standard_pcl_cbk[2]================================"<<endl;
    
    if ((msg->width * msg->height) > min_lidar_points)
    {
        lidar_ok = true;
        pcl::PointCloud<velodyne_ros::Point> pLZ;
        msg_old.reset(new sensor_msgs::PointCloud2);
        msg_old->width = msg->width;
        msg_old->height = msg->height;
        msg_old->point_step = msg->point_step;
        pcl::fromROSMsg(*msg, pLZ);
        // for (int II = 0; II < pLZ.size(); II++)
        // {
        //     if ((pLZ.points[II].x < 7.5 && pLZ.points[II].x > -2.0) && (pLZ.points[II].y < 2.0 && pLZ.points[II].y > -2.0))
        //     {
        //         pLZ.points[II].x = NAN;
        //         pLZ.points[II].y = NAN;
        //         pLZ.points[II].z = NAN;
        //     }
        // }
        pcl::toROSMsg(pLZ, *msg_old);
    }
    else
    {
        lidar_ok = false;
        msg_old->header.stamp = msg_old->header.stamp + ros::Duration(0.1);
    }
    //cout<<"==========================standard_pcl_cbk[3]================================"<<endl;
    p_pre->process(msg_old, ptr);
    //cout<<"==========================standard_pcl_cbk[4]================================"<<endl;
    lidar_buffer.push_back(ptr);
    //cout<<"==========================standard_pcl_cbk[5]================================"<<endl;
    time_buffer.push_back(msg_old->header.stamp.toSec() + time_offset);
    last_timestamp_lidar = msg_old->header.stamp.toSec() + time_offset;
    // p_pre->process(msg, ptr);
    // lidar_buffer.push_back(ptr);
    // time_buffer.push_back(msg->header.stamp.toSec() + time_offset);
    // last_timestamp_lidar = msg->header.stamp.toSec() + time_offset;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool timediff_set_flg = false;

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{

    //cout<<"==========================imu_cbk[0]================================"<<endl;
    publish_count++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
    // msg->header.stamp=msg->header.stamp+ros::Duration(0.0);
    if (lidar_ok == false) ////LZLZLZLZ2024.07.29
    {
        msg->angular_velocity.x = 0;
        msg->angular_velocity.y = 0;
        msg->angular_velocity.z = 0;
    }
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp =
            ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }
    
    //cout<<"==========================imu_cbk[1]================================"<<endl;

    double timestamp = msg->header.stamp.toSec();

    geometry_msgs::Quaternion quat;
    quat.x = msg->orientation.x;
    quat.y = msg->orientation.y;
    quat.z = msg->orientation.z;
    quat.w = msg->orientation.w;

    // 转换四元数到欧拉角
    tf::Quaternion quat_tf(quat.x, quat.y, quat.z, quat.w);
    tf::Matrix3x3 mat_tf(quat_tf);
    double roll, pitch, yaw;
    mat_tf.getRPY(roll, pitch, yaw);


    //cout<<"==========================imu_cbk[2]================================"<<endl;

    // printf("imu   time: %17.3lf (R: % 6.3f P % 6.3f Y % 6.3f)\n",msg->header.stamp.toSec(),roll*180/3.1415926, pitch*180/3.1415926, yaw*180/3.1415926);
    if (timestamp < last_timestamp_imu)
    {
        //        ROS_WARN("imu loop back, clear buffer");
        //        imu_buffer.clear();
        ROS_WARN("imu loop back, ignoring!!!");
        ROS_WARN("current T: %f, last T: %f", timestamp, last_timestamp_imu);
        return;
    }
    // 剔除异常数据
    // LZLZ
    imu_ang_vel = 0.0;
    imu_ang_vel = (imu_ang_vel) > abs(msg->angular_velocity.x) ? (imu_ang_vel) : abs(msg->angular_velocity.x);
    imu_ang_vel = (imu_ang_vel) > abs(msg->angular_velocity.y) ? (imu_ang_vel) : abs(msg->angular_velocity.y);
    imu_ang_vel = (imu_ang_vel) > abs(msg->angular_velocity.z) ? (imu_ang_vel) : abs(msg->angular_velocity.z);
    //
    if (std::abs(msg->angular_velocity.x) > 6 || std::abs(msg->angular_velocity.y) > 6 || std::abs(msg->angular_velocity.z) > 6)
    {
        ROS_WARN("Large IMU measurement!!! Drop Data!!! %.3f  %.3f  %.3f",
                 msg->angular_velocity.x,
                 msg->angular_velocity.y,
                 msg->angular_velocity.z);
        return;
    }


    //cout<<"==========================imu_cbk[3]================================"<<endl;

    last_timestamp_imu = timestamp;

    mtx_buffer.lock();

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{

    //cout<<"================================[7]==============================="<<endl;
    lidar_buffer.empty();
    //cout<<"================================[7.1]==============================="<<endl;
    imu_buffer.empty();
    //cout<<"================================[7.2]==============================="<<endl;
    if (lidar_buffer.empty() || imu_buffer.empty())
    {
        //cout<<"================================[7.1]==============================="<<endl;
        return false;
    }
    //cout<<"================================[7.3]==============================="<<endl;
    /*** push a lidar scan ***/
    if (!lidar_pushed)
    {
    
        //cout<<"================================[8]==============================="<<endl;
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

            scan_num++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    //cout<<"================================[7.4]==============================="<<endl;
    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    //cout<<"================================[9]==============================="<<endl;
    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        //cout<<"================================[10]==============================="<<endl;
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if (imu_time > lidar_end_time)
            break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }
    
    //cout<<"================================[11]==============================="<<endl;
    
    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

void publish_frame_world(const ros::Publisher &pubLaserCloudFull)
{
    if (scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI laserCloudWorld;
        laserCloudWorld_LZ->clear(); // LZLZLZ
        for (int i = 0; i < size; i++)
        {
            PointType const *const p = &laserCloudFullRes->points[i];
            if (p->intensity < 5)
            {
                continue;
            }

            PointType p_world;

            RGBpointBodyToWorld(p, &p_world);
            laserCloudWorld.push_back(p_world);
        }
        *laserCloudWorld_LZ = laserCloudWorld;
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    // LZLZLZLZLZ
    /**************** save map ****************/
    if (pcd_save_en) // pcd_save_en
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(
            new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i],
                                &laserCloudWorld->points[i]);
        }

        float lll = sqrt((pos_save[0] - pos_save_temp[0]) * (pos_save[0] - pos_save_temp[0]) +
                         (pos_save[1] - pos_save_temp[1]) * (pos_save[1] - pos_save_temp[1]) +
                         (pos_save[2] - pos_save_temp[2]) * (pos_save[2] - pos_save_temp[2]));
        if (lll > 0.20 || abs(pos_save_temp[3] - pos_save[3]) > 0.09 || abs(pos_save_temp[4] - pos_save[4]) > 0.09 || abs(pos_save_temp[5] - pos_save[5]) > 0.09)
        {
            *pcl_wait_save += *laserCloudWorld;
        }
        if (lll > 0.5 || abs(pos_save_temp[3] - pos_save[3]) > 0.1745 || abs(pos_save_temp[4] - pos_save[4]) > 0.1745 || abs(pos_save_temp[5] - pos_save[5]) > 0.1745)
        {

            pos_save_temp[0] = pos_save[0];
            pos_save_temp[1] = pos_save[1];
            pos_save_temp[2] = pos_save[2];
            pos_save_temp[3] = pos_save[3];
            pos_save_temp[4] = pos_save[4];
            pos_save_temp[5] = pos_save[5];

            static int scan_wait_num = 0;
            scan_wait_num++;
            if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
            {
                pcd_index++;
                string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
                pcl::PCDWriter pcd_writer;
                cout << "current scan saved to /PCD/" << all_points_dir << endl;
                pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
                pcl_wait_save->clear();
                scan_wait_num = 0;
                FILE *fp;
                string all_points_dirscanNo(string(string(ROOT_DIR) + "PCD/") + "scan_No.txt");
                fp = fopen(all_points_dirscanNo.c_str(), "w");
                fprintf(fp, "%d", pcd_index);
                fclose(fp);

                FILE *fppose;
                string all_points_dirpose(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".txt"));
                fppose = fopen(all_points_dirpose.c_str(), "w");
                fprintf(fppose, "%012.3f %012.3f %012.3f %012.3f %012.3f %012.3f", pos_save[0], pos_save[1], pos_save[2], pos_save[3], pos_save[4], pos_save[5]);
                fclose(fppose);
            }
        }
    }
    // LZLZLZLZLZ
}

void publish_cloud_store(const ros::Publisher &pub_cloud_store)
{
    //cout<<"========================publish_cloud_store========================="<<endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_store(new pcl::PointCloud<pcl::PointXYZRGB>);
    if (scan_pub_en)
    {
        PointCloudXYZI::Ptr feature_undistort(new PointCloudXYZI());
        PointCloudXYZI::Ptr laserCloudFullRes(false ? feature_undistort : feats_down_body);
        pcl::PointCloud<pcl::PointXYZ>::Ptr lasercloudfull_LZ(new pcl::PointCloud<pcl::PointXYZ>());
        for (int i = 0; i < laserCloudFullRes->points.size(); i++)
        {
            pcl::PointXYZ p;
            p.x = laserCloudFullRes->points[i].x;
            p.y = laserCloudFullRes->points[i].y;
            p.z = laserCloudFullRes->points[i].z;
            lasercloudfull_LZ->points.push_back(p);
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudFullRes_filter(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PassThrough<pcl::PointXYZ> pass_throuth;
        pass_throuth.setInputCloud(lasercloudfull_LZ);
        pass_throuth.setFilterFieldName("x");
        pass_throuth.setFilterLimits(cloud_store_Xmin, cloud_store_Xmax);
        pass_throuth.setFilterFieldName("y");
        pass_throuth.setFilterLimits(cloud_store_Ymin, cloud_store_Ymax);
        pass_throuth.filter(*laserCloudFullRes_filter);

        if (laserCloudFullRes_filter->size() > 0)
        {
            pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n; // OMP加速
            pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
            // 建立kdtree来进行近邻点集搜索
            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
            n.setNumberOfThreads(5); // 设置openMP的线程数
            n.setInputCloud(laserCloudFullRes_filter);
            n.setSearchMethod(tree);
            // n.setKSearch(10);//点云法向计算时，需要所搜的近邻点大小
            n.setRadiusSearch(cloud_storeRadius); // 半径搜素
            n.compute(*normals);                  // 开始进行法向计

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_store_new(new pcl::PointCloud<pcl::PointXYZRGB>);
            for (int i = 0; i < normals->size(); i++)
            {
                if (abs(normals->points[i].normal_x) > 0.8)
                {
                    pcl::PointXYZRGB pointXYZ;
                    pointXYZ.x = laserCloudFullRes_filter->points[i].x;
                    pointXYZ.y = laserCloudFullRes_filter->points[i].y;
                    pointXYZ.z = laserCloudFullRes_filter->points[i].z;
                    pointXYZ.r = 255;
                    pointXYZ.g = 0;
                    pointXYZ.b = 0;
                    cloud_store_new->points.push_back(pointXYZ);
                }
                else if (abs(normals->points[i].normal_y) > 0.95)
                {
                    pcl::PointXYZRGB pointXYZ;
                    pointXYZ.x = laserCloudFullRes_filter->points[i].x;
                    pointXYZ.y = laserCloudFullRes_filter->points[i].y;
                    pointXYZ.z = laserCloudFullRes_filter->points[i].z;
                    pointXYZ.r = 0;
                    pointXYZ.g = 255;
                    pointXYZ.b = 0;
                    cloud_store_new->points.push_back(pointXYZ);
                }
                else if (abs(normals->points[i].normal_z) > 0.8)
                {
                    pcl::PointXYZRGB pointXYZ;
                    pointXYZ.x = laserCloudFullRes_filter->points[i].x;
                    pointXYZ.y = laserCloudFullRes_filter->points[i].y;
                    pointXYZ.z = laserCloudFullRes_filter->points[i].z;
                    pointXYZ.r = 0;
                    pointXYZ.g = 0;
                    pointXYZ.b = 255;
                    cloud_store_new->points.push_back(pointXYZ);
                }
            }
            int size = cloud_store_new->points.size();
            // PointCloudXYZI laserCloudWorld;
            for (int i = 0; i < size; i++)
            {
                pcl::PointXYZRGB const *const p = &cloud_store_new->points[i];
                pcl::PointXYZRGB p_world;
                RGBpointBodyToWorld_LZ(p, &p_world);
                pcl::PointXYZRGB pointXYZ;
                // Copy the XYZ coordinates from pointXYZI to pointXYZ
                pointXYZ.x = p_world.x;
                pointXYZ.y = p_world.y;
                pointXYZ.z = p_world.z;
                pointXYZ.r = p_world.r;
                pointXYZ.g = p_world.g;
                pointXYZ.b = p_world.b;
                // Add the point to the new cloud
                cloud_store->points.push_back(pointXYZ);
            }

            sensor_msgs::PointCloud2 laserCloudmsg;
            pcl::toROSMsg(*cloud_store, laserCloudmsg);
            laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
            laserCloudmsg.header.frame_id = "camera_init";
            pub_cloud_store.publish(laserCloudmsg);
            publish_count -= PUBFRAME_PERIOD;
        }
    }
}
/// ZXB need topic //LZ add

void publish_frame_body(const ros::Publisher &pubLaserCloudFull_body)
{
    //    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));
    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&laserCloudFullRes->points[i],
                               &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

// 无用回调
/*
void publish_map(const ros::Publisher &pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}
*/

template <typename T>
void set_posestamp(T &out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    // LZLZ
    if (!flg_exit)
    {
        tf2::Quaternion quat_tf;
        tf2::convert(geoQuat, quat_tf);
        quat_tf.normalize();
        FILE *fp;
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + "path.txt");
        fp = fopen(all_points_dir.c_str(), "w");
        fprintf(fp, "%012.3f %012.3f %012.3f %012.3f %012.3f %012.3f %012.3f", out.pose.position.x, out.pose.position.y,
                out.pose.position.z, out.pose.orientation.w, out.pose.orientation.x, out.pose.orientation.y, out.pose.orientation.z);
        fclose(fp);
        pos_save[0] = out.pose.position.x;
        pos_save[1] = out.pose.position.y;
        pos_save[2] = out.pose.position.z;
        tf2::Matrix3x3(quat_tf).getRPY(pos_save[3], pos_save[4], pos_save[5]);
    }
    // LZLZLZ
}

void publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x,
                                    odomAftMapped.pose.pose.position.y,
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "body"));

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
    // offset_R_L_I矩阵转四元数后转matrix取了转置，所以逆时针为负
    trans_cloud->clear();
    for (size_t i = 0; i < input_cloud->size(); i++)
    {
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
    /*
        std::cout << "LZLZLZ state_point.pos " << state_point.pos << std::endl;
        std::cout << "state_point.offset_T_L_I" << state_point.pos << std::endl;
        std::cout << "R quat: " << state_point.offset_R_L_I << std::endl;
        std::cout << "R matr: " << state_point.offset_R_L_I.matrix() << std::endl;
        std::cout << "rot quat: " << state_point.rot << std::endl;
        std::cout << "rot matr: " << state_point.rot.matrix() << std::endl;
        std::cout << "lidar: " << input_cloud->points[0] << std::endl;
        std::cout << "body:  " << trans_cloud->points[0] << std::endl;
    */
    //
    /*
    M3D Lidar_R_test_LZ(Eye3d);
    Lidar_R_test_LZ<<0,1,0,-1,0,0,0,0,1;
    V3D point_test(1.0,1.0,1.0);
    V3D point_test_out(0.0,0.0,0.0);
    point_test_out=Lidar_R_test_LZ*point_test;
    std::cout<<"LZLZLZ "<<point_test_out<<std::endl;
*/
}

M3D transformLiDARCovToWorld(Eigen::Vector3d &p_lidar, const esekfom::esekf<state_ikfom, 12, input_ikfom> &kf, const Eigen::Matrix3d &COV_lidar)
{
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(p_lidar);
    auto state = kf.get_x();
    // lidar到body的方差传播
    // 注意外参的var是先rot 后pos
    M3D il_rot_var = kf.get_P().block<3, 3>(6, 6);
    M3D il_t_var = kf.get_P().block<3, 3>(9, 9);

    M3D COV_body =
        state.offset_R_L_I * COV_lidar * state.offset_R_L_I.conjugate() + state.offset_R_L_I * (-point_crossmat) * il_rot_var * (-point_crossmat).transpose() * state.offset_R_L_I.conjugate() + il_t_var;

    // body的坐标
    V3D p_body = state.offset_R_L_I * p_lidar + state.offset_T_L_I;

    // body到world的方差传播
    // 注意pose的var是先pos 后rot
    point_crossmat << SKEW_SYM_MATRX(p_body);
    M3D rot_var = kf.get_P().block<3, 3>(3, 3);
    M3D t_var = kf.get_P().block<3, 3>(0, 0);
    // Eq. (3)
    M3D COV_world =
        state.rot * COV_body * state.rot.conjugate() + state.rot * (-point_crossmat) * rot_var * (-point_crossmat).transpose() * state.rot.conjugate() + t_var;

    return COV_world;
    // Voxel map 真实实现
    //    M3D cov_world = R_body * COV_lidar * R_body.conjugate() +
    //          (-point_crossmat) * rot_var * (-point_crossmat).transpose() + t_var;
}

void observation_model_share(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    //    laserCloudOri->clear();
    //    corr_normvect->clear();
    // feats_with_correspondence->clear();
    total_residual = 0.0;

    // =================================================================================================================
    // 用当前迭代轮最新的位姿估计值 将点云转换到world地图系
    vector<pointWithCov> pv_list;
    PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI);
    // FIXME stupid mistake 这里应该用迭代的最新线性化点
    // FIXME stupid mistake 这里应该用迭代的最新线性化点
    transformLidar(s, feats_down_body, world_lidar);
    pv_list.resize(feats_down_body->size());
    for (size_t i = 0; i < feats_down_body->size(); i++)
    {
        // 保存body系和world系坐标
        pointWithCov pv;
        pv.point << feats_down_body->points[i].x, feats_down_body->points[i].y, feats_down_body->points[i].z;
        pv.point_world << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
        // 计算lidar点的cov
        // 注意这个在每次迭代时是存在重复计算的 因为lidar系的点云covariance是不变的
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
    BuildResidualListOMP(voxel_map, max_voxel_size, 3.0, max_layer, pv_list,
                         ptpl_list, non_match_list);
    double match_end = omp_get_wtime();
    // std::printf("Match Time: %f\n", match_end - match_start);

    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    // 根据匹配结果 设置H和R的维度
    // h_x是观测值对状态量的导数 TODO 为什么不加上状态量对状态量误差的导数？？？？像quaternion那本书？
    effct_feat_num = ptpl_list.size();
    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        flg_exit = true;
        return;
    }
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); // 23 因为点面距离只和位姿 外参有关 对其他状态量的导数都是0
    ekfom_data.h.resize(effct_feat_num);
    ekfom_data.R.resize(effct_feat_num, 1); // 把R作为向量 用的时候转换成diag
    //    ekfom_data.R.setZero();
    //    printf("isDiag: %d  R norm: %f\n", ekfom_data.R.isDiagonal(1e-10), ekfom_data.R.norm());
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < effct_feat_num; i++)
    {
        //        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(ptpl_list[i].point);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        //        const PointType &norm_p = corr_normvect->points[i];
        //        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);
        V3D norm_vec(ptpl_list[i].normal);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() * norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); // s.rot.conjugate()*norm_vec);
            // ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec.x(), norm_vec.y(), norm_vec.z(), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            // ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec.x(), norm_vec.y(), norm_vec.z(), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        //        ekfom_data.h(i) = -norm_p.intensity;
        float pd2 = norm_vec.x() * ptpl_list[i].point_world.x() + norm_vec.y() * ptpl_list[i].point_world.y() + norm_vec.z() * ptpl_list[i].point_world.z() + ptpl_list[i].d;
        ekfom_data.h(i) = -pd2;

        // norm_p中存了匹配的平面法向 还有点面距离
        // V3D point_world = s.rot * (s.offset_R_L_I * ptpl_list[i].point + s.offset_T_L_I) + s.pos;
        V3D point_world = ptpl_list[i].point_world;
        // /*** get the normal vector of closest surface/corner ***/
        Eigen::Matrix<double, 1, 6> J_nq;
        J_nq.block<1, 3>(0, 0) = point_world - ptpl_list[i].center;
        J_nq.block<1, 3>(0, 3) = -ptpl_list[i].normal;
        double sigma_l = J_nq * ptpl_list[i].plane_cov * J_nq.transpose();

        // M3D cov_lidar = calcBodyCov(ptpl_list[i].point, ranging_cov, angle_cov);
        M3D cov_lidar = ptpl_list[i].cov_lidar;
        M3D R_cov_Rt = s.rot * s.offset_R_L_I * cov_lidar * s.offset_R_L_I.conjugate() * s.rot.conjugate();
        // HACK 1. 因为是标量 所以求逆直接用1除
        // HACK 2. 不同分量的方差用加法来合成 因为公式(12)中的Sigma是对角阵，逐元素运算之后就是对角线上的项目相加
        double R_inv = 1.0 / (sigma_l + norm_vec.transpose() * R_cov_Rt * norm_vec);

        // 计算测量方差R并赋值 目前暂时使用固定值
        // ekfom_data.R(i) = 1.0 / LASER_POINT_COV;
        ekfom_data.R(i) = R_inv;
    }
    float RR = 0.0, RR_inv = 0.0;
    for (int ii = 0; ii < effct_feat_num; ii++)
    {
        RR += ekfom_data.R(ii);
        RR_inv += 1.0 / ekfom_data.R(ii);
    }
    // std::cout << "测量方差R " << RR / effct_feat_num << " RR_inv " << RR_inv / effct_feat_num << " No " << effct_feat_num << std::endl;
    if (effct_feat_num < 100)
        flg_exit = true;
    // std::printf("Effective Points: %d\n", effct_feat_num);
    res_mean_last = total_residual / effct_feat_num;
}

////LZLZ add filedelete 2024 03 18
void Getfilepath(const char *path, const char *filename, char *filepath)
{
    strcpy(filepath, path);
    if (filepath[strlen(path) - 1] != '/')
        strcat(filepath, "/");
    strcat(filepath, filename);
    printf("path is = %s\n", filepath);
}

bool DeleteFile(const char *path)
{
    DIR *dir;
    struct dirent *dirinfo;
    struct stat statbuf;
    char filepath[256] = {0};
    lstat(path, &statbuf);

    if (S_ISREG(statbuf.st_mode)) // 判断是否是常规文件
    {
        remove(path);
    }
    else if (S_ISDIR(statbuf.st_mode)) // 判断是否是目录
    {
        if ((dir = opendir(path)) == NULL)
            return 1;
        while ((dirinfo = readdir(dir)) != NULL)
        {
            Getfilepath(path, dirinfo->d_name, filepath);
            if (strcmp(dirinfo->d_name, ".") == 0 || strcmp(dirinfo->d_name, "..") == 0) // 判断是否是特殊目录
                continue;
            DeleteFile(filepath);
            rmdir(filepath);
        }
        closedir(dir);
    }
    return 0;
}
////LZLZ add filedelete 2024 03 18



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

    //cout<<"========================downsamplePointCloud========================="<<endl;
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

    //cout<<"========================run_ICP========================="<<endl;
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

    //cout<<"========================is_50_percent_in_field========================="<<endl;
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




/*

int counter_gicp = 0;
  
//process the input cloud
void main_callback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
  
  
    //cout<<"========================main_callback========================="<<endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>(5,1));
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_boxed_for_local (new pcl::PointCloud<pcl::PointXYZ>(5,1));
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_filtered (new pcl::PointCloud<pcl::PointXYZ>(5,1));
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_cut (new pcl::PointCloud<pcl::PointXYZ>(5,1));
    
    //cout<<"========================main_callback[1]========================="<<endl;
    pcl::fromROSMsg(*msg, *cloud_in);

    if( counter_gicp == 0 )
    {
        //cout<<"========================main_callback[2]========================="<<endl;
        // Initialize the map
    	*map_final = *cloud_in;
    	//cout<<"========================main_callback[2.1]========================="<<endl;
    	
    }
    else
    {
    
        //cout<<"========================main_callback[3]========================="<<endl;
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
	sor_input.setLeafSize (0.5f, 0.5f, 0.5f);
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
	
	//cout<<"========================main_callback[4]========================="<<endl;
	
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

		std::cout << "[ best_result for fly [0] ]\n" << Ti << std::endl;
		
		
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

                //cout<<"===============================runGICP=================================="<<endl;
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
		
		std::cout << "[ best_result for fly [1] ]\n" << Ti << std::endl;
		
		Ti = best_result.transformation * Ti;


		std::cout << "[ best_result for fly [2] ]\n" << Ti << std::endl;

		//clock_t end = clock(); // Get the current time in clock ticks

		//double runningTime = (double)(end - start) / CLOCKS_PER_SEC; // Convert clock ticks to seconds

		//std::cout << "Running time: " << runningTime << " seconds" << std::endl;
//*

		auto t2_multi = std::chrono::high_resolution_clock::now();
		double single_multi = std::chrono::duration_cast<std::chrono::nanoseconds>(t2_multi - t1_multi).count() / 1e6;
		std::cout << "single_multi:" << single_multi << "[msec] " << std::endl;

		// ------------------------------------------5_line[end] ---------------------------------------------------// 
	
	
	
	
	}
	//*
	
	
	//cout<<"========================main_callback[5]========================="<<endl;
	
	// 2.1 output the cloud_in_boxed_for_local and cloud_pre
	//brown is the input cloud of right now frame
	sensor_msgs::PointCloud2Ptr msg_second(new sensor_msgs::PointCloud2);
	////cout<<"==================cloud_in====================="<<endl;
	pcl::toROSMsg(*cloud_in_boxed_for_local, *msg_second);
	msg_second->header.stamp.fromSec(0);
	msg_second->header.frame_id = "world";
	pub_cloud_surround_.publish(msg_second);
	//*

  
	
	
	
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

	//cout<<"========================main_callback[6]========================="<<endl;
	
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
      
      

	//cout<<"========================main_callback[7]========================="<<endl;
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
  //*

	//cout<<"=================== cloud_in_filtered->size() ==================" <<endl;
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
	
	
	//*
	
	
	
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
	//*

	

	
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
	    //*
        
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
		//*
		
		
	    }   
	}
    }
     
    counter_gicp++;
    
    *cloud_pre = *cloud_in_filtered;
 
  //*
  

    //*
  
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
//*/





Eigen::Vector3f getLongestDirection(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud) {
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
Eigen::Matrix<double, 6, 6> computeHessian(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud) {

    cout<<"size="<<cloud->size()<<endl;
    Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Zero();

   // Create the normal estimation class, and pass the input dataset to it
    pcl::NormalEstimation<pcl::PointXYZINormal, pcl::Normal> ne;
    ne.setInputCloud(cloud);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    pcl::search::KdTree<pcl::PointXYZINormal>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZINormal>());
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

vector<Eigen::Vector3f> longest_direction_vector;

// Function to be executed in the new thread
void processPointCloud(PointCloudXYZI::Ptr world_lidar_point, PointCloudXYZI::Ptr feats_down_body, MeasureGroup &meas) {

	    //--------------------------------------------------------------------------------------------backup_system[start]--------------------------------------------------------------------------------------//
            
            double t_update_start_gicp = omp_get_wtime();
            
            int feats_down_size = feats_down_body->points.size();

            cout<<"==================feats_down_size[1]====================="<<endl;
            cout<<feats_down_size<<endl;
            
            pcl::PointCloud<pcl::PointXYZINormal>::Ptr Final_cloud_translate (new pcl::PointCloud<pcl::PointXYZINormal>(5,1));
            
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


		// 1.1 Voxel the cloud

		// Create the filtering object
		pcl::VoxelGrid<pcl::PointXYZINormal> sor_input;
		sor_input.setInputCloud (feats_down_body);
		sor_input.setLeafSize (0.15f, 0.15f, 0.15f);
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
		
		
		// Optional: Apply Statistical Outlier Removal filter
		pcl::StatisticalOutlierRemoval<pcl::PointXYZINormal> statistical_sor;
		statistical_sor.setInputCloud(cloud_in_filtered);
		statistical_sor.setMeanK(50);
		statistical_sor.setStddevMulThresh(1.0);
		statistical_sor.filter(*cloud_in_filtered);
		

		
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

		pcl::CropBox<pcl::PointXYZINormal> boxFilter_for_in;
		float x_min_for_in = - 50, y_min_for_in = - 50, z_min_for_in = - 0 ;
		float x_max_for_in = + 50, y_max_for_in = + 50, z_max_for_in = + 50;

		boxFilter_for_in.setMin(Eigen::Vector4f(x_min_for_in, y_min_for_in, z_min_for_in, 1.0));
		boxFilter_for_in.setMax(Eigen::Vector4f(x_max_for_in, y_max_for_in, z_max_for_in, 1.0));

		boxFilter_for_in.setInputCloud(cloud_in_boxed_for_local);
		boxFilter_for_in.filter(*cloud_in_boxed_for_local);
		



		//pcl::PointCloud<pcl::PointXYZINormal> Final;

            
               std::cout << "================================[0]================================[" << std::endl;
		
		/*
		// 2.0 gicp of the cloud_in voxeled and pre cloud
		pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZINormal, pcl::PointXYZINormal> gicp;
		//pcl::registration::GICPWithRobustKernel<pcl::PointXYZ> gicp;
		gicp.setMaxCorrespondenceDistance(1.0);
		gicp.setTransformationEpsilon(0.02);
		gicp.setMaximumIterations(500);

		gicp.setInputSource(cloud_in_filtered);
		gicp.setInputTarget(cloud_pre);
		gicp.align(*Final);
		Ti = gicp.getFinalTransformation () * Ti;
		//*/
		
		Eigen::Vector3f angvel_avr, acc_avr;
		double dt = 0;
		
		auto v_imu = meas.imu;
		
		for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
		{
			auto &&head = *(it_imu);
			auto &&tail = *(it_imu + 1);

			if (tail->header.stamp.toSec() < p_imu->last_lidar_end_time_)
				continue;

			angvel_avr << 	0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
					0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
					0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
			acc_avr << 	0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
					0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
					0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);


			acc_avr = acc_avr * G_m_s2 / p_imu->mean_acc.norm(); // - state_inout.ba;

			if (head->header.stamp.toSec() < p_imu->last_lidar_end_time_)
			{
				dt = tail->header.stamp.toSec() - p_imu->last_lidar_end_time_;
				// dt = tail->header.stamp.toSec() - pcl_beg_time;
			}
			else
			{
				dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
			}


			// Assuming you have functions/variables like angvel_avr, dt, acc_avr already defined
			Eigen::Matrix3f R_delta = Exp(angvel_avr, dt); // Assuming Exp() returns Eigen::Matrix3f
			Eigen::Vector3f T_delta = 0.5 * acc_avr * dt * dt; // Assuming acc_avr is Eigen::Vector3f or can be cast

			// Define a 4x4 transformation matrix as an identity matrix
			Eigen::Matrix4f imu_transformation = Eigen::Matrix4f::Identity();

			// Assign the rotation part to the top-left 3x3 block
			imu_transformation.block<3, 3>(0, 0) = R_delta;

			// Assign the translation part to the top-right 3x1 block
			imu_transformation.block<3, 1>(0, 3) = T_delta;

			Ti = imu_transformation * Ti;
		}


		
		cout<<"==============================Ti========================="<<endl;
		cout<<Ti<<endl;            
               
               
		// 3.0 voxed the map_final
		// Create the filtering object
		pcl::VoxelGrid<pcl::PointXYZINormal> sor;
		sor.setInputCloud (map_final);
		sor.setLeafSize (0.15f, 0.15f, 0.15f);
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
      
      
                // Check if point clouds are populated
		if (cloud_in_boxed_translate_to_near_mapboxed->size() < 30 )
		{
			std::cerr << "Source point cloud is empty!" << std::endl;
			return ;
		}

		if (map_final_boxed->size() < 30 )
		{
			std::cerr << "Target point cloud is empty!" << std::endl;
			return ;
		}        
      
               
               std::cout << "================================[0.1]================================" << std::endl;
		// Step 1: Initial alignment using NDT
		pcl::NormalDistributionsTransform<pcl::PointXYZINormal, pcl::PointXYZINormal> ndt;
		ndt.setResolution(2.0);
		ndt.setInputSource(cloud_in_boxed_translate_to_near_mapboxed);
		ndt.setInputTarget(map_final_boxed);

		// NDT parameters
		ndt.setMaximumIterations(20);
		ndt.setStepSize(0.2);
		ndt.setTransformationEpsilon(0.05);

		// Initial guess for the transformation (identity matrix)
		Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();

		// Perform the initial alignment
		ndt.align(*Final, init_guess);
		std::cout << "================================[0.2]================================" << std::endl;
		// Check if NDT has converged
		if (ndt.hasConverged()) {
			std::cout << "NDT converged with score: " << ndt.getFitnessScore() << std::endl;
			std::cout << "NDT Transformation matrix:\n" << ndt.getFinalTransformation() << std::endl;
		} else {
			std::cout << "NDT did not converge." << std::endl;
			//return -1;
		}

               
               

               
               
               
               std::cout << "================================[1]================================" << std::endl;
		pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZINormal, pcl::PointXYZINormal> gicp_for_map;
		gicp_for_map.setMaxCorrespondenceDistance(5.0);
		gicp_for_map.setTransformationEpsilon(0.05);
		gicp_for_map.setRotationEpsilon(0.05);
		gicp_for_map.setMaximumIterations(500);

		gicp_for_map.setInputSource(cloud_in_boxed_translate_to_near_mapboxed);
		gicp_for_map.setInputTarget(map_final_boxed);
		//gicp_for_map.align(*Final);
		gicp_for_map.align(*Final, ndt.getFinalTransformation());

		Ti_of_map = gicp_for_map.getFinalTransformation (); // * Ti_of_map;
               
               Ti_real = Ti_of_map * Ti;
               





		
		
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
		

		cout<<"========================Eigen::Matrix4d_T[1.1]======================"<<endl;  
		cout<<T<<endl;
		cout<< T(0,3) - Ti_real(0,3) <<","<<T(1,3) - Ti_real(1,3)<<","<<T(2,3) - Ti_real(2,3)<<endl;


		if( abs( T(0,3) - Ti_real(0,3) ) > 0.2 || abs( T(1,3) - Ti_real(1,3) ) > 0.2  || abs( T(2,3) - Ti_real(2,3) ) > 0.2 )
		{

			//Ti_real(0,3)  = T.cast<float>()(0,3) ;
			//Ti_real(1,3)  = T.cast<float>()(1,3) ;
			//Ti_real(2,3)  = T.cast<float>()(2,3) ;

		}
		
		cout<<"========================Eigen::Matrix4d_Ti[1.2]======================"<<endl;  
		cout<<Ti_real<<endl;
		cout<<Ti<<endl;

		
		
		
		pcl::transformPointCloud (*cloud_in_filtered, *Final_cloud_translate, Ti_real); 

		if( counter_loop % 10 == 0 )
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

			std::cout << "================================[3]================================[" << std::endl;
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
            
            
            /*
            
            // Get the longest direction of the tunnel
            Eigen::Vector3f longest_direction = getLongestDirection(cloud_pre);

	    
	    longest_direction_vector.push_back(longest_direction);

            // Convert Eigen::Vector3f to geometry_msgs::Vector3
            geometry_msgs::Vector3 direction_msg;
            direction_msg.x = longest_direction(0);
            direction_msg.y = longest_direction(1);
            direction_msg.z = longest_direction(2);

            //direction_pub.publish(direction_msg);
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
	    
	    //mean_dir_pub.publish(mean_msg);
	    //std_dir_pub.publish(std_msg);
	    
	    

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

	    //marker_pub.publish(marker);
            
            cout<<"longest_direction[1]="<<longest_direction<<endl;
            
            
            
            // 3.1 boxed the map_final
            pcl::CropBox<pcl::PointXYZINormal> box_filter_cov;
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
		    Eigen::Vector3d scales;
		    for(int i = 0; i < 3; ++i) {
		    	scales[i] = std::sqrt(rotation_eigenvalues[i]);
		    }

		    eigenvalue_pub.publish(eigenvalues_msg);

		    // Publish ellipsoid marker
		    //publishEllipsoidMarker( marker_pub_Hessien, centroid.head<3>(), scales,  rotation_eigenvectors, "world" // Replace with appropriate frame_id);
		}
	    }

            cout<<"longest_direction[2]="<<longest_direction<<endl;
            
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
            
            
            double t_update_end_gicp = omp_get_wtime();
            double optimize_time_gicp = t_update_start_gicp - t_update_end_gicp;
            
            cout<<"optimize_time_gicp="<<optimize_time_gicp<<endl;
            //std::this_thread::sleep_for(std::chrono::milliseconds(30));
            //*/
            //--------------------------------------------------------------------------------------------backup_system[end]--------------------------------------------------------------------------------------//
            
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;
    ///////////////////////LZLZLZLZLZ/////////
    int save_map_rightnow = -1;
    ///////////////////////LZLZLZLZLZ/////////
    nh.param<double>("time_offset", lidar_time_offset, 0.0);
    nh.param<bool>("publish/path_en", path_en, true);
    nh.param<bool>("publish/scan_publish_en", scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en", dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en, true);
    nh.param<string>("common/lid_topic", lid_topic, "/lidar");
    nh.param<string>("common/imu_topic", imu_topic, "/imu");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    // LZLZLZ
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<bool>("Init_coordinates", Init_coordinates, false);
    nh.param<double>("Velthreshold", Vel_threshold, 1.0);
    // LZLZLZ
    //  mapping algorithm params
    nh.param<float>("mapping/det_range", DET_RANGE, 300.f);
    nh.param<int>("mapping/max_iteration", NUM_MAX_ITERATIONS, 4);
    nh.param<int>("mapping/max_points_size", max_points_size, 100);
    nh.param<int>("mapping/max_cov_points_size", max_cov_points_size, 100);
    nh.param<vector<double>>("mapping/layer_point_size", layer_point_size, vector<double>());
    nh.param<int>("mapping/max_layer", max_layer, 2);
    nh.param<double>("mapping/voxel_size", max_voxel_size, 1.0);
    nh.param<double>("mapping/down_sample_size", filter_size_surf_min, 0.5);
    nh.param<double>("mapping/plannar_threshold", min_eigen_value, 0.01);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>()); // offset_R_L_I矩阵转四元数后转matrix取了转置，所以逆时针为负
    // noise model params
    nh.param<double>("noise_model/ranging_cov", ranging_cov, 0.02);
    nh.param<double>("noise_model/angle_cov", angle_cov, 0.05);
    nh.param<double>("noise_model/gyr_cov", gyr_cov, 0.1);
    nh.param<double>("noise_model/acc_cov", acc_cov, 0.1);
    nh.param<double>("noise_model/b_gyr_cov", b_gyr_cov, 0.0001);
    nh.param<double>("noise_model/b_acc_cov", b_acc_cov, 0.0001);
    // visualization params
    nh.param<bool>("publish/pub_voxel_map", publish_voxel_map, false);
    nh.param<int>("publish/publish_max_voxel_layer", publish_max_voxel_layer, 0);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num, 1);
    nh.param<bool>("preprocess/feature_extract_enable", p_pre->feature_enabled, false);
    cout << "p_pre->lidar_type " << p_pre->lidar_type << endl;
    for (int i = 0; i < layer_point_size.size(); i++)
    {
        layer_size.push_back(layer_point_size[i]);
    }

    path.header.stamp = ros::Time::now();
    path.header.frame_id = "camera_init";

    /*** variables definition 可变变量定义***/
    // memset(point_selected_surf, true, sizeof(point_selected_surf));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    // memset(point_selected_surf, true, sizeof(point_selected_surf));//???什么东西，初始化两次

    //  开始时lidar callback中固定转换到IMU系下
    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    double epsi[23] = {0.001};
    fill(epsi, epsi + 23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, observation_model_share, NUM_MAX_ITERATIONS, epsi);

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = nh.subscribe(lid_topic, 1, standard_pcl_cbk);                                      // LZ   订阅lidar话题
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 100, imu_cbk);                                             // LZ   订阅imu话题
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 3);           // 世界坐标系下，所有点云
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 3); // body坐标系下，所有点云数据
    // ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100);//无用topic
    // ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100);//无用topic 注释掉 publish_map
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 1); // 全局坐标系下，里程计信息
    // ros::Publisher pubExtrinsic = nh.advertise<nav_msgs::Odometry>("/Extrinsic", 10);//无用topic
    ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 1);
    ros::Publisher voxel_map_pub = nh.advertise<visualization_msgs::MarkerArray>("/planes", 10);
    ros::Publisher pubMapping = nh.advertise<std_msgs::String>("/mappingok", 1); // LZ 添加构图成功

    /// ZXB need topic //LZ add
    ros::Publisher pub_cloud_store = nh.advertise<sensor_msgs::PointCloud2>("/cloud_store", 1);
    /// ZXB need topic //LZ add
    
//------------------------------------------------------------------------------------------------------   
    //ros::Subscriber sub_pc_ = nh.subscribe<sensor_msgs::PointCloud2>("/cari_points", 10, main_callback); //cari_points_top
    pub_history_keyframes_ = nh.advertise<sensor_msgs::PointCloud2>("/history_keyframes", 10);
    pub_recent_keyframes_ = nh.advertise<sensor_msgs::PointCloud2>("/recent_keyframes", 10);
    pub_icp_keyframes_ = nh.advertise<sensor_msgs::PointCloud2>("/icp_keyframes", 10);
    pub_cloud_surround_ = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 10);
 
    pub_odom_aft_mapped_2 = nh.advertise<nav_msgs::Odometry>("/odom_aft_mapped_2", 10);
    
    /*
     map_final.reset(new pcl::PointCloud<pcl::PointXYZ>);
     map_final_boxed.reset(new pcl::PointCloud<pcl::PointXYZ>);
     map_final_boxed_2.reset(new pcl::PointCloud<pcl::PointXYZ>);
     map_final_boxed_cov.reset(new pcl::PointCloud<pcl::PointXYZ>);
     Final.reset(new pcl::PointCloud<pcl::PointXYZ>);
     cloud_pre.reset(new pcl::PointCloud<pcl::PointXYZ>);
     cloud_in_filtered.reset(new pcl::PointCloud<pcl::PointXYZ>);
     cloud_in_boxed_for_local.reset(new pcl::PointCloud<pcl::PointXYZ>);
     //*/
     
     map_final.reset(new PointCloudXYZI);
     map_final_boxed.reset(new PointCloudXYZI);
     map_final_boxed_2.reset(new PointCloudXYZI);
     map_final_boxed_cov.reset(new PointCloudXYZI);
     Final.reset(new PointCloudXYZI);
     cloud_pre.reset(new PointCloudXYZI);
     cloud_in_filtered.reset(new PointCloudXYZI);
     cloud_in_boxed_for_local.reset(new PointCloudXYZI);

     Ti = Eigen::Matrix4f::Identity ();     
     Ti_real = Eigen::Matrix4f::Identity ();
     Ti_of_map = Eigen::Matrix4f::Identity ();
     T = Eigen::Matrix4d::Identity();
     counter_loop = 0;
     counter_stable_map = 0;
//------------------------------------------------------------------------------------------------------


    bool init_map = false; // 地图是否已经初始化
    bool icp_point = false;
    // double sum_optimize_time = 0, sum_update_time = 0;//计算时间打印变量，实际无用
    int scan_index = 0; // 扫描次数计算平均匹配时间差

    signal(SIGINT, SigHandle);
    ros::Rate rate(500);
    bool status = ros::ok();

    // check
    // int checka = 0;

    //cout<<"================================[1]==============================="<<endl;

    while (status)
    {
        if (flg_exit)
            break;
        ros::spinOnce();
        
	// Ensure proper initialization
	if (p_pre == nullptr) {
		std::cerr << "Error: PointCloud pointer is not initialized! [p_pre]" << std::endl;
		continue;
	}

	// Ensure proper initialization
	if (p_imu == nullptr) {
		std::cerr << "Error: PointCloud pointer is not initialized! [p_imu]" << std::endl;
		continue;
	}        
        
        bool sync_LZ = sync_packages(Measures);
        //cout<<"================================[2.1]==============================="<<endl;
        if (sync_LZ)
        {
            //cout<<"================================[2]==============================="<<endl;
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time; // EKF初始时间，激光雷达时间
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                // LZ 初始化，读取开始时位姿【XYZwxyz】
                double readpos[7] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
                if (!Init_coordinates)
                {
                    FILE *fp;
                    string all_points_dir(string(string(ROOT_DIR) + "PCD/") + "path.txt");
                    fp = fopen(all_points_dir.c_str(), "r");
                    int fsr = fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf", &readpos[0], &readpos[1], &readpos[2], &readpos[3], &readpos[4], &readpos[5], &readpos[6]);
                    fclose(fp);

                    FILE *fpscan;
                    int scan_No = 0;
                    string all_points_dirscan_No(string(string(ROOT_DIR) + "PCD/") + "scan_No.txt");
                    fpscan = fopen(all_points_dirscan_No.c_str(), "r");
                    if (fpscan != NULL)
                    {
                        int fsr = fscanf(fpscan, "%d", &scan_No);
                        fclose(fpscan);
                    }
                    pcd_index = scan_No;
                }
                else
                {
                    pcd_index = 0;
                    string all_points_dir(string(string(ROOT_DIR) + "PCD/"));
                    DeleteFile(all_points_dir.c_str());
                }
                double pos[3] = {readpos[0], readpos[1], readpos[2]};
                double rot[4] = {readpos[3], readpos[4], readpos[5], readpos[6]};
                p_imu->set_init_pos_rot(pos, rot);
                // LZLZLZ
                continue;
            }

	    //cout<<"================================[3]==============================="<<endl;
            p_imu->Process(Measures, kf, feats_undistort);                          // imu计算均值等初始化
            state_point = kf.get_x();                                               // IMU   pos vel anglevel cov
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I; // imu的坐标转为激光雷达坐标

            if (feats_undistort->size() < 100 || (feats_undistort == NULL)) // feats_undistort->empty() ||
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }
            
            cout<<"============================feats_undistort==========================="<<endl;
            cout<<feats_undistort->size()<<endl;
            // 3.0 voxed the map_final
            // Create the filtering object
            pcl::VoxelGrid<pcl::PointXYZINormal> sor;
            sor.setInputCloud (feats_undistort);
            sor.setLeafSize (0.3f, 0.3f, 0.3f);
            sor.filter (*feats_undistort);
            cout<<feats_undistort->size()<<endl;


            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true; // 判断是不是第一帧时间，小于0.1秒为false，大于0.1S为true；
            // ===============================================================================================================
            // 第一帧 如果ekf初始化了 就初始化voxel地图
            if (flg_EKF_inited && !init_map)
            {
                PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI);
                transformLidar(state_point, feats_undistort, world_lidar);
                std::vector<pointWithCov> pv_list;
                //  计算第一帧所有点的covariance 并用于构建初始地图
                for (size_t i = 0; i < world_lidar->size(); i++)
                {
                    pointWithCov pv;
                    pv.point << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
                    V3D point_this(feats_undistort->points[i].x,
                                   feats_undistort->points[i].y,
                                   feats_undistort->points[i].z);
                    // if z=0, error will occur in calcBodyCov. To be solved(计算方差)
                    if (point_this[2] == 0)
                    {
                        point_this[2] = 0.001;
                    }
                    M3D cov_lidar = calcBodyCov(point_this, ranging_cov, angle_cov); //
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

                if (publish_voxel_map)
                {
                    pubVoxelMap(voxel_map, publish_max_voxel_layer, voxel_map_pub);
                    publish_frame_world(pubLaserCloudFull);
                    publish_frame_body(pubLaserCloudFull_body);
                }
                init_map = true;
                continue;
            }

            cout<<"============================feats_undistort==========================="<<endl;
            cout<<feats_undistort->size()<<endl;
	    //cout<<"================================[4]==============================="<<endl;
            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list);
            
            cout<<feats_undistort->size()<<endl;

            feats_down_size = feats_down_body->points.size();
            // 由于点云的body var是一直不变的 因此提前计算 在迭代时可以复用
            var_down_body.clear();
            for (auto &pt : feats_down_body->points)
            {
                V3D point_this(pt.x, pt.y, pt.z);
                var_down_body.push_back(calcBodyCov(point_this, ranging_cov, angle_cov));
            }

            /*** ICP and iterated Kalman filter update ***/
            //
            if (feats_down_size < 500)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }
            
            
            PointCloudXYZI::Ptr world_lidar_point(new PointCloudXYZI);
            transformLidar(state_point, feats_down_body, world_lidar_point);

            std::thread worker([=]() { processPointCloud(world_lidar_point, feats_down_body, Measures); });
            
            
            
            
            // ===============================================================================================================
            // 开始迭代滤波
            /*** iterated state estimation ***/

            kf.update_iterated_dyn_share_diagonal(); // 更新动态共享对角矩阵？？？？？？

            state_point = kf.get_x();                                               // 获取kf的状态
            euler_cur = SO3ToEuler(state_point.rot);                                // SO3转为欧拉角（未使用LZLZ）
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I; ////世界系下雷达坐标系的位置 式子的意义是W^p_L = W^p_I + W^R_I * I^t_L
            geoQuat.x = state_point.rot.coeffs()[0];                                // 内联直接rot替换为quater
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            // std::cout<<"second up kf:"<<euler_cur.x()<<" "<<euler_cur.y()<<" "<<euler_cur.z()<<std::endl;

            /*
                state_ikfom init_state = kf.get_x();
                tf2::Quaternion quat_tf(init_state.rot.x(),init_state.rot.y(),init_state.rot.z(),init_state.rot.w());
                //tf2::convert(geoQuat, quat_tf);
                //quad_tf.setRPY();
                //quat_tf.x()=init_state.rot.x();
                //0.0,init_state.rot.z,init_state.rot.w);
                quat_tf.normalize();
                init_state.rot.w()=quat_tf.w();init_state.rot.x()=quat_tf.x();
                init_state.rot.y()=quat_tf.y();init_state.rot.z()=quat_tf.z();
                kf.change_x(init_state);

            */

            // std::printf("BA: %.4f %.4f %.4f   BG: %.4f %.4f %.4f   g: %.4f %.4f %.4f\n",kf.get_x().ba.x(), kf.get_x().ba.y(), kf.get_x().ba.z(),kf.get_x().bg.x(), kf.get_x().bg.y(), kf.get_x().bg.z(),kf.get_x().grav.get_vect().x(), kf.get_x().grav.get_vect().y(), kf.get_x().grav.get_vect().z());
            if ((state_point.vel(0) * state_point.vel(0) + state_point.vel(1) * state_point.vel(1) + state_point.vel(2) * state_point.vel(2)) < (Vel_threshold * Vel_threshold))
            {
                float aaa = sqrt(state_point.vel(0) * state_point.vel(0) + state_point.vel(1) * state_point.vel(1) + state_point.vel(2) * state_point.vel(2));
                // std::printf("vel: %.4f %.4f %.4f T: %.4f <>  %.4f\n",state_point.vel(0),state_point.vel(1),state_point.vel(2),aaa,Vel_threshold);
                std_msgs::String msg_mappingok;
                msg_mappingok.data = "true";
                pubMapping.publish(msg_mappingok);
                ros::param::set("/mappingok", true);
            }
            else
            {
                float aaa = sqrt(state_point.vel(0) * state_point.vel(0) + state_point.vel(1) * state_point.vel(1) + state_point.vel(2) * state_point.vel(2));
                // std::printf("vel: %.4f %.4f %.4f T: %.4f <>  %.4f\n",state_point.vel(0),state_point.vel(1),state_point.vel(2),aaa,Vel_threshold);
                std_msgs::String msg_mappingok;
                msg_mappingok.data = "false";
                pubMapping.publish(msg_mappingok);
                ros::param::set("/mappingok", false);
                // int revsystem = system("rosnode kill /pv_lio_node /rviz"); // SigHandle(100);
                // revsystem = system(string(string(ROOT_DIR) + "closepvlio.sh").c_str());
                // std::raise(SIGINT);
            }

            // Eigen::Matrix4f transfor;
            // gicp_pcl(feats_undistort, laserCloudWorld_LZ, transfor); // feats_undistort
            // state_point.pos << (state_point.pos(0) + transfor(0, 3)), (state_point.pos(1) + transfor(1, 3)), (state_point.pos(2) + transfor(2, 3));
            // Eigen::Matrix3d r_matrix = Eigen::Matrix3d::Identity();
            // for (int i = 0; i < 3; i++)
            // {
            //     for (int j = 0; j < 3; j++)
            //     {
            //         r_matrix(i, j) = transfor(i, j);
            //     }
            // }
            // Eigen::Quaterniond quat_gicp(r_matrix);
            // Eigen::Quaterniond quat_I(state_point.rot.coeffs()[0], state_point.rot.coeffs()[1], state_point.rot.coeffs()[2], state_point.rot.coeffs()[3]);
            // Eigen::Quaterniond quat_IG = quat_gicp * quat_I ;
            // state_point.rot.coeffs()[0] = quat_IG.x();
            // state_point.rot.coeffs()[1] = quat_IG.y();
            // state_point.rot.coeffs()[2] = quat_IG.z();
            // state_point.rot.coeffs()[3] = quat_IG.w();
            // kf.change_x(state_point);

            //  ===============================================================================================================
            //  更新地图
            /*** add the points to the voxel map ***/
            // 用最新的状态估计   将点及点的covariance转换到world系
            std::vector<pointWithCov> pv_list;
            PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI);
            transformLidar(state_point, feats_down_body, world_lidar); // 利用state_point把feats_down_body（body系）转换为世界坐标系下的点云world_lidar


	     //cout<<"================================[5]==============================="<<endl;
            for (size_t i = 0; i < feats_down_body->size(); i++)
            {
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
            std::sort(pv_list.begin(), pv_list.end(), var_contrast); // 世界坐标系下点按照方差排序

            // gicp_pcl(feats_undistort, laserCloudWorld_LZ);

            // pv_list.clear();
            updateVoxelMapOMP(pv_list, max_voxel_size, max_layer, layer_size,
                              max_points_size, max_points_size, min_eigen_value, voxel_map); // 更新体素地图？？？？？？？？？？？？

	    //cout<<"================================[6]==============================="<<endl;

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped); // 发布构图后的odom

            /******* Publish points *******/
            if (path_en)
                publish_path(pubPath);
            if (scan_pub_en)
                publish_frame_world(pubLaserCloudFull);

            /// ZXB need topic //LZ add
            if (scan_pub_en)
                publish_cloud_store(pub_cloud_store);
            /// ZXB need topic //LZ add

            if (scan_pub_en && scan_body_pub_en)
                publish_frame_body(pubLaserCloudFull_body);
            if (publish_voxel_map)
            {
                pubVoxelMap(voxel_map, publish_max_voxel_layer, voxel_map_pub);
            }
            
            worker.join();
        }
        
        //cout<<"================================[2.2]==============================="<<endl;

        ///////////////////////LZLZLZLZLZ/////////
        /**************** save map right now ***************
        bool ifget1 = ros::param::get("save_map_rightnow", save_map_rightnow);
        if (save_map_rightnow > 0)
        {
            if (pcl_wait_save->size() > 0)
            {
                string file_name = string("scans_rt.pcd");
                string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
                pcl::PCDWriter pcd_writer;
                cout << "current scan saved to /PCD/" << file_name << endl;
                pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
                pcl_wait_save->clear();
            }
            else
            {
                std::cout << "pc size =0 no map save" << std::endl;
            }
            nh.setParam("save_map_rightnow", -1);
            // save_map_rightnow=-1;
        }
        */
        ///////////////////////LZLZLZLZLZ/////////

        status = ros::ok();
        rate.sleep();
    }
    
    
    SigHandle(100);
    return 0;
}
