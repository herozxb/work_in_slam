#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
#include "use-ikfom.hpp"

/// *************Preconfiguration

#include <deque>

// A small struct to hold one snapshot (timestamp, state, covariance)
struct KFBackup {
  double                             t;   // external timestamp (seconds)
  state_ikfom                        x;   // full KF state vector at time t
  esekfom::esekf<state_ikfom,12,input_ikfom>::cov P; // covariance at time t
};





#define MAX_INI_COUNT (20)

const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

V3D angular_velocity;
V3D xyz_velocity;

/// *************IMU Process and undistortion
class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();
  
  void Reset();
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void set_extrinsic(const V3D &transl, const M3D &rot);
  void set_extrinsic(const V3D &transl);
  void set_extrinsic(const MD(4,4) &T);
  void set_gyr_cov(const V3D &scaler);
  void set_acc_cov(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  Eigen::Matrix<double, 12, 12> Q;
  void Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr pcl_un_);

  ofstream fout_imu;
  V3D cov_acc;
  V3D cov_gyr;
  V3D cov_acc_scale;
  V3D cov_gyr_scale;
  V3D cov_bias_gyr;
  V3D cov_bias_acc;
  double first_lidar_time;

  SO3 Initial_R_wrt_G;
  vector<Pose6D> IMUpose;
  
 private:
  void IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);
  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out);

  PointCloudXYZI::Ptr cur_pcl_un_;
  sensor_msgs::ImuConstPtr last_imu_;
  deque<sensor_msgs::ImuConstPtr> v_imu_;

  vector<M3D>    v_rot_pcl_;
  M3D Lidar_R_wrt_IMU;
  V3D Lidar_T_wrt_IMU;
  V3D mean_acc;
  V3D mean_gyr;
  V3D angvel_last;
  V3D acc_s_last;
  double start_timestamp_;
  double last_lidar_end_time_;
  int    init_iter_num = 1;
  bool   b_first_frame_ = true;
  bool   imu_need_init_ = true;
  
  
  std::deque<KFBackup> buffer_;
  
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1)
{
  init_iter_num = 1;
  Q = process_noise_cov();
  cov_acc       = V3D(0.1, 0.1, 0.1);
  cov_gyr       = V3D(0.1, 0.1, 0.1);
  cov_bias_gyr  = V3D(0.0001, 0.0001, 0.0001);
  cov_bias_acc  = V3D(0.0001, 0.0001, 0.0001);
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last     = Zero3d;
  Lidar_T_wrt_IMU = Zero3d;
  Lidar_R_wrt_IMU = Eye3d;
  last_imu_.reset(new sensor_msgs::Imu());
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() 
{
  // ROS_WARN("Reset ImuProcess");
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last       = Zero3d;
  imu_need_init_    = true;
  start_timestamp_  = -1;
  init_iter_num     = 1;
  v_imu_.clear();
  IMUpose.clear();
  last_imu_.reset(new sensor_msgs::Imu());
  cur_pcl_un_.reset(new PointCloudXYZI());
}

void ImuProcess::set_extrinsic(const MD(4,4) &T)
{
  Lidar_T_wrt_IMU = T.block<3,1>(0,3);
  Lidar_R_wrt_IMU = T.block<3,3>(0,0);
}

void ImuProcess::set_extrinsic(const V3D &transl)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU.setIdentity();
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
}

void ImuProcess::set_gyr_cov(const V3D &scaler)
{
  cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov(const V3D &scaler)
{
  cov_acc_scale = scaler;
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g)
{
  cov_bias_gyr = b_g;
}

void ImuProcess::set_acc_bias_cov(const V3D &b_a)
{
  cov_bias_acc = b_a;
}

void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  
  V3D cur_acc, cur_gyr;
  
  if (b_first_frame_)
  {
    Reset();
    N = 1;
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    first_lidar_time = meas.lidar_beg_time;
  }

  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc      += (cur_acc - mean_acc) / N;
    mean_gyr      += (cur_gyr - mean_gyr) / N;

    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);

    // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;

    N ++;
  }
  state_ikfom init_state = kf_state.get_x();
  init_state.grav = S2(- mean_acc / mean_acc.norm() * G_m_s2);
  
  //state_inout.rot = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
  init_state.bg  = mean_gyr;
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;
  init_state.offset_R_L_I = Lidar_R_wrt_IMU;
  kf_state.change_x(init_state);

  // initial world pose
  Initial_R_wrt_G = SO3(g2R(mean_acc));

  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();
  init_P.setIdentity();
  // init_P.setZero();
  init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;
  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;
  init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;
  init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;
  init_P(21,21) = init_P(22,22) = 0.00001; 
  kf_state.change_P(init_P);
  last_imu_ = meas.imu.back();

}

void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_out)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();
  const double &pcl_beg_time = meas.lidar_beg_time;
  const double &pcl_end_time = meas.lidar_end_time;
  
  /*** sort point clouds by offset time ***/
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;

  /*** Initialize IMU pose ***/
  state_ikfom imu_state = kf_state.get_x();
  IMUpose.clear();
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));

  /*** forward propagation at each imu point ***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;

  double dt = 0;

  input_ikfom in;
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);
    
    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;
    
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    // fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;

    acc_avr     = acc_avr * G_m_s2 / mean_acc.norm(); // - state_inout.ba;

    if(head->header.stamp.toSec() < last_lidar_end_time_)
    {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_;
      // dt = tail->header.stamp.toSec() - pcl_beg_time;
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }
    
    in.acc = acc_avr;
    in.gyro = angvel_avr;
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
    // 这里应该判断是否有GPS measurements， 如果有GPS， 那么先传播到GPS，然后更新state，再重新从头开始向前传播
    // 看一下r2live的实现
    kf_state.predict(dt, Q, in);

    /* save the poses at each IMU measurements */
    imu_state = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg;
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += imu_state.grav[i];
    }
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);
  kf_state.predict(dt, Q, in);
  
  imu_state = kf_state.get_x();
  last_imu_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;
  
  
  //cout<<"========================v======================"<<endl;  
  state_ikfom state_point;
  state_point = kf_state.get_x();    
  //cout<<state_point.vel<<endl;
  //cout<<angvel_avr[0]<<","<<angvel_avr[1]<<","<<angvel_avr[2]<<endl;
  
  //cout<<acc_avr[0]<<","<<acc_avr[1]<<","<<acc_avr[2]<<endl;
  
  //==========================================================================================//
  angular_velocity << angvel_avr[0], angvel_avr[1], angvel_avr[2]; 
  xyz_velocity << state_point.vel[0], state_point.vel[1], state_point.vel[2]; 
  
  // (D) Now predict from last IMU time to the end‐of‐LIDAR frame:
  double raw_dt_end = pcl_end_time - imu_end_time;
  double dt_end     = (raw_dt_end < 0.0) ? 0.0 : raw_dt_end;

  in.acc  = acc_avr;    // from the last IMU pair
  in.gyro = angvel_avr; // from the last IMU pair

  // Clamp rotation again:
  {
    double rotAngle = in.gyro.norm() * dt_end;
    static constexpr double MAX_ROT_PER_STEP = M_PI;
    if (rotAngle > MAX_ROT_PER_STEP) {
      double scale = MAX_ROT_PER_STEP / rotAngle;
      in.gyro *= scale;
    }
  }

  // (D1) Final predict to LIDAR end:
  kf_state.predict(dt_end, Q, in);



  // 3 key point
  // 1. (predict_time - buffer_.front().t) > 0.5, how many second back tracing
  // 2. fly away speed
  // 3. backup speed recover limit
  
  double time_to_store = 1;
  double fly_speed = 20.0;
  double recover_limit = 9.0;
  
  
  // (D2) Back up at LIDAR end time:
  {
    double predict_time = pcl_end_time;
    
    KFBackup snap;
    snap.t = predict_time;
    snap.x = kf_state.get_x();
    snap.P = kf_state.get_P();
    buffer_.push_back(std::move(snap));
    
    while (!buffer_.empty() && (predict_time - buffer_.front().t) > time_to_store ) 
    {
      buffer_.pop_front();
    }
  }

  // 3) Grab this final predicted state
  state_ikfom state_point2 = kf_state.get_x();

  // 4) If KF has “flown away” (any vel component > 5 m/s), rewind:
  if (std::abs(state_point2.vel[0]) > fly_speed ||
      std::abs(state_point2.vel[1]) > fly_speed ||
      std::abs(state_point2.vel[2]) > fly_speed)
  {
    ROS_WARN("[ImuProcess] KF vel = (%.2f, %.2f, %.2f) > 5 m/s → rewinding.",
             state_point2.vel[0],
             state_point2.vel[1],
             state_point2.vel[2]);

    bool restored = false;
    // Walk backward through buffer_ until we find |v| < 3 m/s
    while (!buffer_.empty()) {
      KFBackup b = buffer_.back();
      if (b.x.vel.norm() < recover_limit) {
        // Restore state and covariance from that backup:
        kf_state.change_x(b.x);
        kf_state.change_P(b.P);
        ROS_WARN("[ImuProcess] Rewound to t = %.3f (|v| = %.2f m/s).",
                 b.t, b.x.vel.norm());
        restored = true;
        break;
      }
      buffer_.pop_back();
    }
    
    if (!restored) {
      // No valid snapshot in the last second → do a fallback (e.g., zero velocity)
      ROS_ERROR("[ImuProcess] No valid backup in 1 s! Zeroing velocity.");
      state_ikfom zeroVel = kf_state.get_x();
      zeroVel.vel.setZero();
      kf_state.change_x(zeroVel);
    }
    
  }
  
  
  
  
  
  
  
  /*
// 2) Propagate to the LIDAR‐frame end (clamp dt ≥ 0)
double raw_dt_end = pcl_end_time - imu_end_time;
double dt_end     = (raw_dt_end < 0.0) ? 0.0 : raw_dt_end;

// Compute average angular velocity and acceleration as before

in.acc  = acc_avr;
in.gyro = angvel_avr;

// (A) Clamp |ω|·dt ≤ π so quaternion update can’t break
double rotAngle = in.gyro.norm() * dt_end;
static constexpr double MAX_ROT_PER_STEP = M_PI;
if (rotAngle > MAX_ROT_PER_STEP) {
  double scale = MAX_ROT_PER_STEP / rotAngle;
  in.gyro *= scale;
}

// (B) Do the final predict step
kf_state.predict(dt_end, Q, in);

// 3) Grab the newly predicted state
state_ikfom state_point2 = kf_state.get_x();

// 4) If KF velocity blew past 5 m/s in any axis, reset it to the last IMU-only velocity:
if (std::abs(state_point2.vel[0]) > 5.0 ||
    std::abs(state_point2.vel[1]) > 5.0 ||
    std::abs(state_point2.vel[2]) > 5.0)
{
    // (A) Grab the last IMU-only pose from IMUpose[]:
    auto it_kp = IMUpose.end() - 1;   // one‐past‐end
    const Pose6D &head = *(it_kp - 1);

    // (B) Copy the previous full state so we keep position/orientation/biases/extrinsics:
    state_ikfom fixedState = kf_state.get_x();

    // (C) Overwrite _only_ the velocity with what the IMU integration says:
    fixedState.vel = V3D(head.vel[0],
                         head.vel[1],
                         head.vel[2]);

    // (D) Push that “IMU velocity” back into the filter:
    kf_state.change_x(fixedState);

    ROS_WARN(
      "[ImuProcess] KF vel > 5 m/s; replaced KF vel with IMU-only vel = (%.2f, %.2f, %.2f).",
      head.vel[0], head.vel[1], head.vel[2]
    );
}

  /*
  
  if( abs(state_point.vel[0]) > 5 || abs(state_point.vel[1]) > 5 || abs(state_point.vel[2]) > 5  )
  {
    //cout<<"==========vel_execl============"<<endl;
    
    auto it_kp = IMUpose.end() - 1;
    auto head = it_kp - 1;

    M3D R_imu;
    R_imu << MAT_FROM_ARRAY(head->rot);
    //cout<<R_imu<<endl;

    V3D imu_pos;
    imu_pos << VEC_FROM_ARRAY(head->pos);
    //cout<<imu_pos<<endl;

    V3D vel_imu;
    vel_imu << VEC_FROM_ARRAY(head->vel);
    //cout<<vel_imu<<endl;

    //state_inout.rot_end << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 
    state_point.pos = imu_pos;
    state_point.vel << 0.0, 0.0, 0.0;
    
    kf_state.change_x(state_point);
  }


//*/


  //cout<<"========================[1]======================"<<endl;  
  
  Eigen::Quaterniond quaternion_1; // Assuming quaternion_1 is initialized
  
  quaternion_1.w() = v_imu.back()->orientation.w;
  quaternion_1.x() = v_imu.back()->orientation.x;
  quaternion_1.y() = v_imu.back()->orientation.y;
  quaternion_1.z() = v_imu.back()->orientation.z;
  
  
  Eigen::Quaterniond imu_quaternion_2; // Assuming imu quaternion_2 is initialized
  
  imu_quaternion_2.w() = state_point.rot.w();
  imu_quaternion_2.x() = state_point.rot.x();
  imu_quaternion_2.y() = state_point.rot.y();
  imu_quaternion_2.z() = state_point.rot.z();

  // Compute the relative rotation from imu_quaternion_2 to quaternion_1
  Eigen::Quaterniond relative_rotation = quaternion_1 * imu_quaternion_2.inverse();

  // Convert relative rotation to an axis-angle representation (optional)
  Eigen::AngleAxisd relative_angle_axis(relative_rotation);

  // Extract the angle in radians
  double relative_angle_degrees = relative_angle_axis.angle() * 180.0 / M_PI;
  //cout<<"=========relative_angle_degrees========"<<endl;
  //std::cout << "Relative angle (degrees): " << relative_angle_degrees << std::endl;
  
  //cout<<"========================[2]======================"<<endl;  

  /*** undistort each lidar point (backward propagation) ***/
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu<<MAT_FROM_ARRAY(head->rot);
    // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
    vel_imu<<VEC_FROM_ARRAY(head->vel);
    pos_imu<<VEC_FROM_ARRAY(head->pos);
    acc_imu<<VEC_FROM_ARRAY(tail->acc);
    angvel_avr<<VEC_FROM_ARRAY(tail->gyr);

    for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
    {
      dt = it_pcl->curvature / double(1000) - head->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
      M3D R_i(R_imu * Exp(angvel_avr, dt));
      
      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
      V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);// not accurate!
      
      // save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
}

void ImuProcess::Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr cur_pcl_un_)
{
  double t1,t2,t3;
  t1 = omp_get_wtime();

  if(meas.imu.empty()) {return;};
  ROS_ASSERT(meas.lidar != nullptr);

  if (imu_need_init_)
  {
    /// The very first lidar frame
    IMU_init(meas, kf_state, init_iter_num);

    imu_need_init_ = true;
    
    last_imu_   = meas.imu.back();

    state_ikfom imu_state = kf_state.get_x();
    if (init_iter_num > MAX_INI_COUNT)
    {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;

      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;
      ROS_INFO("IMU Initial Done");
      // ROS_INFO("IMU Initial Done: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
      //          imu_state.grav[0], imu_state.grav[1], imu_state.grav[2], mean_acc.norm(), cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);
    }

    return;
  }

  UndistortPcl(meas, kf_state, *cur_pcl_un_);

  t2 = omp_get_wtime();
  t3 = omp_get_wtime();
  
  // cout<<"[ IMU Process ]: Time: "<<t3 - t1<<endl;
}
