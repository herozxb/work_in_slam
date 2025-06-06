#ifndef VOXEL_MAP_UTIL_HPP
#define VOXEL_MAP_UTIL_HPP
#include "common_lib.h"
#include "omp.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
//#include <execution>
#include <openssl/md5.h>
#include <pcl/common/io.h>
#include <rosbag/bag.h>
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>


#include <vector>
#include <cmath>
#include <iostream>

#include <algorithm>
#include <random>


#include <unordered_set>




#define HASH_P 116101
#define MAX_N 10000000000





static int plane_id = 0;

extern V3D angular_velocity;
extern V3D xyz_velocity;
extern V3D pos_imu;

// a point to plane matching structure
typedef struct ptpl {
  Eigen::Vector3d point;
  Eigen::Vector3d point_world;
  Eigen::Vector3d normal;
  Eigen::Vector3d center;
  Eigen::Matrix<double, 6, 6> plane_cov;
  double d;
  int layer;
  Eigen::Matrix3d cov_lidar;
} ptpl;

// 3D point with covariance
typedef struct pointWithCov {
  Eigen::Vector3d point;
  Eigen::Vector3d point_world;
  Eigen::Matrix3d cov;
  Eigen::Matrix3d cov_lidar;
} pointWithCov;

typedef struct Plane {
  Eigen::Vector3d center;
  Eigen::Vector3d normal;
  Eigen::Vector3d y_normal;
  Eigen::Vector3d x_normal;
  Eigen::Matrix3d covariance;
  Eigen::Matrix<double, 6, 6> plane_cov;
  float radius = 0;
  float min_eigen_value = 1;
  float mid_eigen_value = 1;
  float max_eigen_value = 1;
  float d = 0;
  int points_size = 0;

  bool is_plane = false;
  bool is_init = false;
  int id;
  // is_update and last_update_points_size are only for publish plane
  bool is_update = false;
  int last_update_points_size = 0;
  bool update_enable = true;
} Plane;

class VOXEL_LOC {
public:
  int64_t x, y, z;

  VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0)
      : x(vx), y(vy), z(vz) {}

  bool operator==(const VOXEL_LOC &other) const {
    return (x == other.x && y == other.y && z == other.z);
  }
};

// Hash value
namespace std {
template <> struct hash<VOXEL_LOC> {
  int64_t operator()(const VOXEL_LOC &s) const {
    using std::hash;
    using std::size_t;
    return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
  }
};
} // namespace std

class OctoTree {
public:
  std::vector<pointWithCov> temp_points_; // all points in an octo tree
  std::vector<pointWithCov> new_points_;  // new points in an octo tree
  Plane *plane_ptr_;
  int max_layer_;
  bool indoor_mode_;
  int layer_;
  int octo_state_; // 0 is end of tree, 1 is not
  OctoTree *leaves_[8];
  double voxel_center_[3]; // x, y, z
  std::vector<int> layer_point_size_;
  float quater_length_;
  float planer_threshold_;
  int max_plane_update_threshold_;
  int update_size_threshold_;
  int all_points_num_;
  int new_points_num_;
  int max_points_size_;
  int max_cov_points_size_;
  bool init_octo_;
  bool update_cov_enable_;
  bool update_enable_;
  OctoTree(int max_layer, int layer, std::vector<int> layer_point_size,
           int max_point_size, int max_cov_points_size, float planer_threshold)
      : max_layer_(max_layer), layer_(layer),
        layer_point_size_(layer_point_size), max_points_size_(max_point_size),
        max_cov_points_size_(max_cov_points_size),
        planer_threshold_(planer_threshold) {
    temp_points_.clear();
    octo_state_ = 0;
    new_points_num_ = 0;
    all_points_num_ = 0;
    // when new points num > 5, do a update
    update_size_threshold_ = 5;
    init_octo_ = false;
    update_enable_ = true;
    update_cov_enable_ = true;
    max_plane_update_threshold_ = layer_point_size_[layer_];
    for (int i = 0; i < 8; i++) {
      leaves_[i] = nullptr;
    }
    plane_ptr_ = new Plane;
  }

  // check is plane , calc plane parameters including plane covariance
  void init_plane(const std::vector<pointWithCov> &points, Plane *plane) {
    plane->plane_cov = Eigen::Matrix<double, 6, 6>::Zero();
    plane->covariance = Eigen::Matrix3d::Zero();
    plane->center = Eigen::Vector3d::Zero();
    plane->normal = Eigen::Vector3d::Zero();
    plane->points_size = points.size();
    plane->radius = 0;
    for (auto pv : points) {
      plane->covariance += pv.point * pv.point.transpose();
      plane->center += pv.point;
    }
    plane->center = plane->center / plane->points_size;
    plane->covariance = plane->covariance / plane->points_size -
                        plane->center * plane->center.transpose();
    Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance);
    Eigen::Matrix3cd evecs = es.eigenvectors();
    Eigen::Vector3cd evals = es.eigenvalues();
    Eigen::Vector3d evalsReal;
    evalsReal = evals.real();
    Eigen::Matrix3f::Index evalsMin, evalsMax;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    int evalsMid = 3 - evalsMin - evalsMax;
    Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
    Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
    Eigen::Vector3d evecMax = evecs.real().col(evalsMax);
    // plane covariance calculation
    Eigen::Matrix3d J_Q;
    J_Q << 1.0 / plane->points_size, 0, 0, 0, 1.0 / plane->points_size, 0, 0, 0,
        1.0 / plane->points_size;
    if (evalsReal(evalsMin) < planer_threshold_) {
      std::vector<int> index(points.size());
      std::vector<Eigen::Matrix<double, 6, 6>> temp_matrix(points.size());
      for (int i = 0; i < points.size(); i++) {
        Eigen::Matrix<double, 6, 3> J;
        Eigen::Matrix3d F;
        for (int m = 0; m < 3; m++) {
          if (m != (int)evalsMin) {
            Eigen::Matrix<double, 1, 3> F_m =
                (points[i].point - plane->center).transpose() /
                ((plane->points_size) * (evalsReal[evalsMin] - evalsReal[m])) *
                (evecs.real().col(m) * evecs.real().col(evalsMin).transpose() +
                 evecs.real().col(evalsMin) * evecs.real().col(m).transpose());
            F.row(m) = F_m;
          } else {
            Eigen::Matrix<double, 1, 3> F_m;
            F_m << 0, 0, 0;
            F.row(m) = F_m;
          }
        }
        J.block<3, 3>(0, 0) = evecs.real() * F;
        J.block<3, 3>(3, 0) = J_Q;
        plane->plane_cov += J * points[i].cov * J.transpose();
      }

      plane->normal << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
          evecs.real()(2, evalsMin);
      plane->y_normal << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid),
          evecs.real()(2, evalsMid);
      plane->x_normal << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax),
          evecs.real()(2, evalsMax);
      plane->min_eigen_value = evalsReal(evalsMin);
      plane->mid_eigen_value = evalsReal(evalsMid);
      plane->max_eigen_value = evalsReal(evalsMax);
      plane->radius = sqrt(evalsReal(evalsMax));
      plane->d = -(plane->normal(0) * plane->center(0) +
                   plane->normal(1) * plane->center(1) +
                   plane->normal(2) * plane->center(2));
      plane->is_plane = true;
      if (plane->last_update_points_size == 0) {
        plane->last_update_points_size = plane->points_size;
        plane->is_update = true;
      } else if (plane->points_size - plane->last_update_points_size > 100) {
        plane->last_update_points_size = plane->points_size;
        plane->is_update = true;
      }

      if (!plane->is_init) {
        plane->id = plane_id;
        plane_id++;
        plane->is_init = true;
      }

    } else {
      if (!plane->is_init) {
        plane->id = plane_id;
        plane_id++;
        plane->is_init = true;
      }
      if (plane->last_update_points_size == 0) {
        plane->last_update_points_size = plane->points_size;
        plane->is_update = true;
      } else if (plane->points_size - plane->last_update_points_size > 100) {
        plane->last_update_points_size = plane->points_size;
        plane->is_update = true;
      }
      plane->is_plane = false;
      plane->normal << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
          evecs.real()(2, evalsMin);
      plane->y_normal << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid),
          evecs.real()(2, evalsMid);
      plane->x_normal << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax),
          evecs.real()(2, evalsMax);
      plane->min_eigen_value = evalsReal(evalsMin);
      plane->mid_eigen_value = evalsReal(evalsMid);
      plane->max_eigen_value = evalsReal(evalsMax);
      plane->radius = sqrt(evalsReal(evalsMax));
      plane->d = -(plane->normal(0) * plane->center(0) +
                   plane->normal(1) * plane->center(1) +
                   plane->normal(2) * plane->center(2));
    }
  }

  // only updaye plane normal, center and radius with new points
  void update_plane(const std::vector<pointWithCov> &points, Plane *plane) {
    Eigen::Matrix3d old_covariance = plane->covariance;
    Eigen::Vector3d old_center = plane->center;
    Eigen::Matrix3d sum_ppt =
        (plane->covariance + plane->center * plane->center.transpose()) *
        plane->points_size;
    Eigen::Vector3d sum_p = plane->center * plane->points_size;
    for (size_t i = 0; i < points.size(); i++) {
      Eigen::Vector3d pv = points[i].point;
      sum_ppt += pv * pv.transpose();
      sum_p += pv;
    }
    plane->points_size = plane->points_size + points.size();
    plane->center = sum_p / plane->points_size;
    plane->covariance = sum_ppt / plane->points_size -
                        plane->center * plane->center.transpose();
    Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance);
    Eigen::Matrix3cd evecs = es.eigenvectors();
    Eigen::Vector3cd evals = es.eigenvalues();
    Eigen::Vector3d evalsReal;
    evalsReal = evals.real();
    Eigen::Matrix3d::Index evalsMin, evalsMax;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    int evalsMid = 3 - evalsMin - evalsMax;
    Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
    Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
    Eigen::Vector3d evecMax = evecs.real().col(evalsMax);
    if (evalsReal(evalsMin) < planer_threshold_) {
      plane->normal << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
          evecs.real()(2, evalsMin);
      plane->y_normal << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid),
          evecs.real()(2, evalsMid);
      plane->x_normal << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax),
          evecs.real()(2, evalsMax);
      plane->min_eigen_value = evalsReal(evalsMin);
      plane->mid_eigen_value = evalsReal(evalsMid);
      plane->max_eigen_value = evalsReal(evalsMax);
      plane->radius = sqrt(evalsReal(evalsMax));
      plane->d = -(plane->normal(0) * plane->center(0) +
                   plane->normal(1) * plane->center(1) +
                   plane->normal(2) * plane->center(2));

      plane->is_plane = true;
      plane->is_update = true;
    } else {
      plane->normal << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
          evecs.real()(2, evalsMin);
      plane->y_normal << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid),
          evecs.real()(2, evalsMid);
      plane->x_normal << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax),
          evecs.real()(2, evalsMax);
      plane->min_eigen_value = evalsReal(evalsMin);
      plane->mid_eigen_value = evalsReal(evalsMid);
      plane->max_eigen_value = evalsReal(evalsMax);
      plane->radius = sqrt(evalsReal(evalsMax));
      plane->d = -(plane->normal(0) * plane->center(0) +
                   plane->normal(1) * plane->center(1) +
                   plane->normal(2) * plane->center(2));
      plane->is_plane = false;
      plane->is_update = true;
    }
  }

  void init_octo_tree() {
    if (temp_points_.size() > max_plane_update_threshold_) {
      init_plane(temp_points_, plane_ptr_);
      if (plane_ptr_->is_plane == true) {
        octo_state_ = 0;
        if (temp_points_.size() > max_cov_points_size_) {
          update_cov_enable_ = false;
        }
        if (temp_points_.size() > max_points_size_) {
          update_enable_ = false;
        }
      } else {
        octo_state_ = 1;
        cut_octo_tree();
      }
      init_octo_ = true;
      new_points_num_ = 0;
      //      temp_points_.clear();
    }
  }

  void cut_octo_tree() {
    if (layer_ >= max_layer_) {
      octo_state_ = 0;
      return;
    }
    for (size_t i = 0; i < temp_points_.size(); i++) {
      int xyz[3] = {0, 0, 0};
      if (temp_points_[i].point[0] > voxel_center_[0]) {
        xyz[0] = 1;
      }
      if (temp_points_[i].point[1] > voxel_center_[1]) {
        xyz[1] = 1;
      }
      if (temp_points_[i].point[2] > voxel_center_[2]) {
        xyz[2] = 1;
      }
      int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
      if (leaves_[leafnum] == nullptr) {
        leaves_[leafnum] = new OctoTree(
            max_layer_, layer_ + 1, layer_point_size_, max_points_size_,
            max_cov_points_size_, planer_threshold_);
        leaves_[leafnum]->voxel_center_[0] =
            voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
        leaves_[leafnum]->voxel_center_[1] =
            voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
        leaves_[leafnum]->voxel_center_[2] =
            voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
        leaves_[leafnum]->quater_length_ = quater_length_ / 2;
      }
      leaves_[leafnum]->temp_points_.push_back(temp_points_[i]);
      leaves_[leafnum]->new_points_num_++;
    }
    for (uint i = 0; i < 8; i++) {
      if (leaves_[i] != nullptr) {
        if (leaves_[i]->temp_points_.size() >
            leaves_[i]->max_plane_update_threshold_) {
          init_plane(leaves_[i]->temp_points_, leaves_[i]->plane_ptr_);
          if (leaves_[i]->plane_ptr_->is_plane) {
            leaves_[i]->octo_state_ = 0;
          } else {
            leaves_[i]->octo_state_ = 1;
            leaves_[i]->cut_octo_tree();
          }
          leaves_[i]->init_octo_ = true;
          leaves_[i]->new_points_num_ = 0;
        }
      }
    }
  }

  void UpdateOctoTree(const pointWithCov &pv) {
    if (!init_octo_) {
      new_points_num_++;
      all_points_num_++;
      temp_points_.push_back(pv);
      if (temp_points_.size() > max_plane_update_threshold_) {
        init_octo_tree();
      }
    } else {
      if (plane_ptr_->is_plane) {
        if (update_enable_) {
          new_points_num_++;
          all_points_num_++;
          if (update_cov_enable_) {
            temp_points_.push_back(pv);
          } else {
            new_points_.push_back(pv);
          }
          if (new_points_num_ > update_size_threshold_) {
            if (update_cov_enable_) {
              init_plane(temp_points_, plane_ptr_);
            }
            new_points_num_ = 0;
          }
          if (all_points_num_ >= max_cov_points_size_) {
            update_cov_enable_ = false;
            std::vector<pointWithCov>().swap(temp_points_);
          }
          if (all_points_num_ >= max_points_size_) {
            update_enable_ = false;
            plane_ptr_->update_enable = false;
            std::vector<pointWithCov>().swap(new_points_);
          }
        } else {
          return;
        }
      } else {
        if (layer_ < max_layer_) {
          if (temp_points_.size() != 0) {
            std::vector<pointWithCov>().swap(temp_points_);
          }
          if (new_points_.size() != 0) {
            std::vector<pointWithCov>().swap(new_points_);
          }
          int xyz[3] = {0, 0, 0};
          if (pv.point[0] > voxel_center_[0]) {
            xyz[0] = 1;
          }
          if (pv.point[1] > voxel_center_[1]) {
            xyz[1] = 1;
          }
          if (pv.point[2] > voxel_center_[2]) {
            xyz[2] = 1;
          }
          int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
          if (leaves_[leafnum] != nullptr) {
            leaves_[leafnum]->UpdateOctoTree(pv);
          } else {
            leaves_[leafnum] = new OctoTree(
                max_layer_, layer_ + 1, layer_point_size_, max_points_size_,
                max_cov_points_size_, planer_threshold_);
            leaves_[leafnum]->layer_point_size_ = layer_point_size_;
            leaves_[leafnum]->voxel_center_[0] =
                voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
            leaves_[leafnum]->voxel_center_[1] =
                voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
            leaves_[leafnum]->voxel_center_[2] =
                voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
            leaves_[leafnum]->quater_length_ = quater_length_ / 2;
            leaves_[leafnum]->UpdateOctoTree(pv);
          }
        } else {
          if (update_enable_) {
            new_points_num_++;
            all_points_num_++;
            if (update_cov_enable_) {
              temp_points_.push_back(pv);
            } else {
              new_points_.push_back(pv);
            }
            if (new_points_num_ > update_size_threshold_) {
              if (update_cov_enable_) {
                init_plane(temp_points_, plane_ptr_);
              } else {
                update_plane(new_points_, plane_ptr_);
                new_points_.clear();
              }
              new_points_num_ = 0;
            }
            if (all_points_num_ >= max_cov_points_size_) {
              update_cov_enable_ = false;
              std::vector<pointWithCov>().swap(temp_points_);
            }
            if (all_points_num_ >= max_points_size_) {
              update_enable_ = false;
              plane_ptr_->update_enable = false;
              std::vector<pointWithCov>().swap(new_points_);
            }
          }
        }
      }
    }
  }
};

void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g,
            uint8_t &b) {
  r = 255;
  g = 255;
  b = 255;

  if (v < vmin) {
    v = vmin;
  }

  if (v > vmax) {
    v = vmax;
  }

  double dr, dg, db;

  if (v < 0.1242) {
    db = 0.504 + ((1. - 0.504) / 0.1242) * v;
    dg = dr = 0.;
  } else if (v < 0.3747) {
    db = 1.;
    dr = 0.;
    dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
  } else if (v < 0.6253) {
    db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
    dg = 1.;
    dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
  } else if (v < 0.8758) {
    db = 0.;
    dr = 1.;
    dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
  } else {
    db = 0.;
    dg = 0.;
    dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
  }

  r = (uint8_t)(255 * dr);
  g = (uint8_t)(255 * dg);
  b = (uint8_t)(255 * db);
}


std::vector<VOXEL_LOC> all_voxel_point;

// Function to calculate the Euclidean distance between two points
float distance(const VOXEL_LOC& a, const VOXEL_LOC& b) {
    return std::sqrt((a.x - b.x) * (a.x - b.x) +
                     (a.y - b.y) * (a.y - b.y) +
                     (a.z - b.z) * (a.z - b.z));
}

// Function to find the nearest point in the vector to the target point
VOXEL_LOC findNearestPoint(const std::vector<VOXEL_LOC>& points, const VOXEL_LOC& target, double& minDist ) {
    if (points.empty()) {
        //throw std::runtime_error("The points vector is empty.");
        cout<<"The points vector is empty."<<endl;
        return target;
    }

    minDist = std::numeric_limits<double>::max();
    VOXEL_LOC nearestPoint;

    for (const auto& point : points) {
        float dist = distance(point, target);
        if (dist < minDist) {
            minDist = dist;
            nearestPoint = point;
            
            if ( minDist == 1)
            {
            	break;
            }
        }
    }

    return nearestPoint;
}


int map_erease_refine_counter = 0;

int map_size_counter = 0;

void refine_map(std::vector<VOXEL_LOC>& points, V3D pos_imu) {
    if (points.empty()) {
        std::cout << "The points vector is empty." << std::endl;
    }


    VOXEL_LOC imu_position((int64_t)pos_imu[0], (int64_t)pos_imu[1], (int64_t)pos_imu[2]);

    //cout<<"=============pos_imu==================="<<endl;
    //cout<<imu_position.x<<","<<imu_position.y<<","<<imu_position.z<<endl;


    for( int i = 0; i < points.size(); i++ )
    {
	float dist = distance(points[i], imu_position);
	map_size_counter++;
	//cout<<"dist="<<dist<<endl;
	//cout<<"map_size_counter="<<map_size_counter<<endl;
	// Delete points farther than 30 units from the target
	if (dist > 3) {
		//cout<<"=============erease==================="<<endl;
		points.erase(points.begin() + i);
	}
    
    }


/*
    // Iterate backwards to safely delete elements
    for (auto it = points.begin(); it != points.end(); it++) {
        float dist = distance(*it, imu_position);
        map_size_counter++;
	cout<<"dist="<<dist<<endl;
	cout<<"map_size_counter="<<map_size_counter<<endl;
        // Delete points farther than 30 units from the target
        if (dist > 30) {
	    cout<<"=============erease==================="<<endl;
	    cout<<(*it).x<<","<<(*it).y<<","<<(*it).z<<endl;
            it = points.erase(it); // erase() returns the next valid iterator
        }
    }
    //*/
    map_size_counter=0;
}


// Function to check if the vector contains a specific VOXEL_LOC
bool containsVoxel(const std::vector<VOXEL_LOC>& voxels, const VOXEL_LOC& target) {
    return std::find(voxels.begin(), voxels.end(), target) != voxels.end();
}


void buildVoxelMap(const std::vector<pointWithCov> &input_points,
                   const float voxel_size, const int max_layer,
                   const std::vector<int> &layer_point_size,
                   const int max_points_size, const int max_cov_points_size,
                   const float planer_threshold,
                   std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map) {
  uint plsize = input_points.size();
  for (uint i = 0; i < plsize; i++) {
    const pointWithCov p_v = input_points[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_v.point[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    
    if (iter != feat_map.end()   ) {
      feat_map[position]->temp_points_.push_back(p_v);
      feat_map[position]->new_points_num_++;
      
      //cout<<"================all_voxel_point=================="<<endl;
      //cout<<position.x<<endl;
      //cout<<position.y<<endl;
      //cout<<position.z<<endl;
      
      if (containsVoxel(all_voxel_point, position)) {
      	std::cout << "The voxel with coordinates (" << position.x << ", " << position.y << ", " << position.z << ") is present in the vector." << std::endl;
      } else {
      	std::cout << "The voxel with coordinates (" << position.x << ", " << position.y << ", " << position.z << ") is not present in the vector." << std::endl;
      	all_voxel_point.push_back(position);
      }
      //*/
      
    } else {
    
      /*
      double minDist = 1000000;
      VOXEL_LOC nearest = findNearestPoint(all_voxel_point, position, minDist);
      //std::cout << "Nearest point to (" << position.x << ", " << position.y << ", " << position.z << ") is (" << nearest.x << ", " << nearest.y << ", " << nearest.z << ")" << std::endl;
      //cout<<"minDist="<<minDist<<endl;

    
    
      if( minDist > 3 )
      {
      
      }
      else//*/
      {
	      OctoTree *octo_tree =
		  new OctoTree(max_layer, 0, layer_point_size, max_points_size,
		               max_cov_points_size, planer_threshold);
	      feat_map[position] = octo_tree;
	      feat_map[position]->quater_length_ = voxel_size / 4;
	      feat_map[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
	      feat_map[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
	      feat_map[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
	      feat_map[position]->temp_points_.push_back(p_v);
	      feat_map[position]->new_points_num_++;
	      feat_map[position]->layer_point_size_ = layer_point_size;
      }
    }
  }
  for (auto iter = feat_map.begin(); iter != feat_map.end(); ++iter) {
    iter->second->init_octo_tree();
  }
}

void updateVoxelMap(const std::vector<pointWithCov> &input_points,
                    const float voxel_size, const int max_layer,
                    const std::vector<int> &layer_point_size,
                    const int max_points_size, const int max_cov_points_size,
                    const float planer_threshold,
                    std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map) {
    uint plsize = input_points.size();
    for (uint i = 0; i < plsize; i++) {
        const pointWithCov p_v = input_points[i];
        float loc_xyz[3];
        for (int j = 0; j < 3; j++) {
            loc_xyz[j] = p_v.point[j] / voxel_size;
            if (loc_xyz[j] < 0) {
                loc_xyz[j] -= 1.0;
            }
        }
        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                           (int64_t)loc_xyz[2]);
        auto iter = feat_map.find(position);
        // 如果点的位置已经存在voxel 那么就更新点的位置 否则创建新的voxel
        if (iter != feat_map.end()) {
            feat_map[position]->UpdateOctoTree(p_v);
        } else {
            OctoTree *octo_tree =
                    new OctoTree(max_layer, 0, layer_point_size, max_points_size,
                                 max_cov_points_size, planer_threshold);
            feat_map[position] = octo_tree;
            feat_map[position]->quater_length_ = voxel_size / 4;
            feat_map[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
            feat_map[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
            feat_map[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
            feat_map[position]->UpdateOctoTree(p_v);
        }
    }
}


std::vector<VOXEL_LOC> generate_voxel_line(const VOXEL_LOC &start, const VOXEL_LOC &end, double voxel_size) {
    std::vector<VOXEL_LOC> voxelLine;

    int64_t x1 = static_cast<int64_t>(start.x / voxel_size);
    int64_t y1 = static_cast<int64_t>(start.y / voxel_size);
    int64_t z1 = static_cast<int64_t>(start.z / voxel_size);

    int64_t x2 = static_cast<int64_t>(end.x / voxel_size);
    int64_t y2 = static_cast<int64_t>(end.y / voxel_size);
    int64_t z2 = static_cast<int64_t>(end.z / voxel_size);

    int64_t dx = std::abs(x2 - x1), dy = std::abs(y2 - y1), dz = std::abs(z2 - z1);
    int64_t xs = (x2 > x1) ? 1 : -1;
    int64_t ys = (y2 > y1) ? 1 : -1;
    int64_t zs = (z2 > z1) ? 1 : -1;

    int64_t ax = 2 * dx, ay = 2 * dy, az = 2 * dz;
    int64_t sx = xs * dx, sy = ys * dy, sz = zs * dz;

    if (dx >= dy && dx >= dz) {
        int64_t yd = ay - dx;
        int64_t zd = az - dx;
        while (x1 != x2) {
            voxelLine.push_back({x1, y1, z1});
            if (yd >= 0) {
                y1 += ys;
                yd -= ax;
            }
            if (zd >= 0) {
                z1 += zs;
                zd -= ax;
            }
            x1 += xs;
            yd += ay;
            zd += az;
        }
    } else if (dy >= dx && dy >= dz) {
        int64_t xd = ax - dy;
        int64_t zd = az - dy;
        while (y1 != y2) {
            voxelLine.push_back({x1, y1, z1});
            if (xd >= 0) {
                x1 += xs;
                xd -= ay;
            }
            if (zd >= 0) {
                z1 += zs;
                zd -= ay;
            }
            y1 += ys;
            xd += ax;
            zd += az;
        }
    } else {
        int64_t xd = ax - dz;
        int64_t yd = ay - dz;
        while (z1 != z2) {
            voxelLine.push_back({x1, y1, z1});
            if (xd >= 0) {
                x1 += xs;
                xd -= az;
            }
            if (yd >= 0) {
                y1 += ys;
                yd -= az;
            }
            z1 += zs;
            xd += ax;
            yd += ay;
        }
    }

    voxelLine.push_back({x2, y2, z2});

    return voxelLine;
}


/*
bool does_line_from_origin_intersect_voxel_wall(const VOXEL_LOC &newPosition, const std::vector<VOXEL_LOC> &voxelWall, double voxel_size) {

    VOXEL_LOC origin((int64_t)pos_imu[0], (int64_t)pos_imu[1], (int64_t)pos_imu[2]);
    
    // Generate the line of voxels
    auto voxelLine = generate_voxel_line(origin, newPosition, voxel_size);

    // Check if any voxel on the line intersects with the voxel wall
    for (const auto &voxel : voxelLine) {
        for (const auto &wall_voxel : voxelWall) {
            if (voxel == wall_voxel) {
                return true;
            }
        }
    }
    return false;
}
//*/



bool does_line_from_origin_intersect_voxel_wall(const VOXEL_LOC &newPosition, const std::vector<VOXEL_LOC> &voxelWall, double voxel_size) {

    VOXEL_LOC origin((int64_t)pos_imu[0], (int64_t)pos_imu[1], (int64_t)pos_imu[2]);

    // Generate the line of voxels
    std::vector<VOXEL_LOC> voxelLine = generate_voxel_line(origin, newPosition, voxel_size);

    // Create a hash set of the voxel wall for fast lookup
    std::unordered_set<VOXEL_LOC> voxelWallSet(voxelWall.begin(), voxelWall.end());

    // Check if any voxel on the line intersects with the voxel wall
    for (const VOXEL_LOC &voxel : voxelLine) {
        if (voxelWallSet.find(voxel) != voxelWallSet.end()) {
            return true;
        }
    }
    return false;
}
//*/

void random_delete_half(std::vector<VOXEL_LOC> &all_voxel_point) {
    if (all_voxel_point.size() > 700) {
        // Randomly shuffle the points
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(all_voxel_point.begin(), all_voxel_point.end(), g);

        // Erase the last half of the points
        all_voxel_point.erase(all_voxel_point.begin() + all_voxel_point.size() / 2, all_voxel_point.end());
    }
}

int counter = 0;

int time_counter = 0;

void updateVoxelMapOMP(const std::vector<pointWithCov> &input_points,
                    const float voxel_size, const int max_layer,
                    const std::vector<int> &layer_point_size,
                    const int max_points_size, const int max_cov_points_size,
                    const float planer_threshold,
                    std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map) {



  std::unordered_map<VOXEL_LOC, vector<pointWithCov>> position_index_map;
  int insert_count = 0, update_count = 0;
  uint plsize = input_points.size();

  counter = 0;
  time_counter++;
  
  cout<<"time_counter="<<time_counter<<endl;


  map_erease_refine_counter++;
  
  if( map_erease_refine_counter % 100 == 0)
  { 
    refine_map(all_voxel_point,pos_imu);
  }

  double t_update_start = omp_get_wtime();
  for (uint i = 0; i < plsize; i++) {
    const pointWithCov p_v = input_points[i];
    // 计算voxel坐标
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_v.point[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    // 如果点的位置已经存在voxel 那么就更新点的位置 否则创建新的voxel
    if (iter != feat_map.end()) {
      // 更新的点总是很多 先缓存 再延迟并行更新
      update_count++;
      position_index_map[position].push_back(p_v);
      

      
      if (containsVoxel(all_voxel_point, position)) {
      	//std::cout << "The voxel with coordinates (" << position.x << ", " << position.y << ", " << position.z << ") is present in the vector." << std::endl;
      } else {
      	//std::cout << "The voxel with coordinates (" << position.x << ", " << position.y << ", " << position.z << ") is not present in the vector." << std::endl;
	all_voxel_point.push_back(position);
	//cout<<"================all_voxel_point[1]=================="<<endl;
	//cout<<position.x<<endl;
	//cout<<position.y<<endl;
	//cout<<position.z<<endl;
      }
      //*/
      
      

     
    } else {
    
      
      
      //cout<<"===========all_voxel_point.size()============["<<all_voxel_point.size()<<"]"<<endl;
    
      /*
      //counter = all_voxel_point.size();
    
      double minDist = 2000000;
      
      VOXEL_LOC nearest = findNearestPoint(all_voxel_point, position, minDist);
      //std::cout << "Nearest point to [1] (" << position.x << ", " << position.y << ", " << position.z << ") is (" << nearest.x << ", " << nearest.y << ", " << nearest.z << ")" << std::endl;
      //cout<<"minDist="<<minDist<<endl;
      double trace = feat_map[nearest]->plane_ptr_->plane_cov.sum();
      
      double dot_product  = ( position.x - nearest.x ) * feat_map[nearest]->plane_ptr_->normal[0] + ( position.y - nearest.y ) * feat_map[nearest]->plane_ptr_->normal[1] + ( position.z - nearest.z ) * feat_map[nearest]->plane_ptr_->normal[2] ;
      
      cout<<"=================[trace]====================="<<endl;
      //cout<<trace<<endl;
      cout<<dot_product<<endl;
      cout<<"minDist="<<minDist<<endl;
      //cout<<xyz_velocity[0]<<"," <<xyz_velocity[1]<<"," <<xyz_velocity[2]<<endl;
      
      /*
      cout<<"==============================plane[start]======================================"<<endl;
      cout<<" near vector" << position.x - nearest.x << "," << position.y - nearest.y << "," << position.z - nearest.z <<endl;
      cout<<"plane normal ="<<feat_map[nearest]->plane_ptr_->normal<<endl;
      cout<<"plane center ="<<feat_map[nearest]->plane_ptr_->center<<endl;
      double dot_product  = ( position.x - nearest.x ) * feat_map[nearest]->plane_ptr_->normal[0] + ( position.y - nearest.y ) * feat_map[nearest]->plane_ptr_->normal[1] + ( position.z - nearest.z ) * feat_map[nearest]->plane_ptr_->normal[2] ;
      cout<<"dot_product  ="<<dot_product<<endl;
      cout<<"==============================plane[end]======================================"<<endl;
      
      
      
      if ( abs(angular_velocity[0]) > 3.0 || abs(angular_velocity[1]) > 3.0 || abs(angular_velocity[2]) > 3.0 )
      {
      	cout<<"========================angular_velocity_too_high======================"<<endl;
      	cout<<angular_velocity[0]<<","<<angular_velocity[1]<<","<<angular_velocity[2]<<endl;
      	std::cout << "Nearest point to [1] (" << position.x << ", " << position.y << ", " << position.z << ") is (" << nearest.x << ", " << nearest.y << ", " << nearest.z << ")" << std::endl;
        cout<<"minDist="<<minDist<<endl;
      }
      
      //*/
      
      if( all_voxel_point.size() > 700 )
      {
      	// Randomly delete half of the points if the number of points exceneds 700
      	//random_delete_half(all_voxel_point);
      }
      
      bool intersected = does_line_from_origin_intersect_voxel_wall(position, all_voxel_point, voxel_size);
      
      //if (doesLineFromOriginIntersectVoxelWall(position, all_voxel_point, voxel_size)) {
      //if (intersected) {
      //      std::cout << "The line from the origin intersects the voxel wall.[yes]" << std::endl;
      //} else {
      //      std::cout << "The line from the origin does not intersect the voxel wall.[no]" << std::endl;
      //}

     
      //if(  minDist > 2 && 1 < 50  && minDist != 2000000 )
      //if(  minDist > 1 && minDist != 2000000 )
      if(  intersected )
      {
      		//cout<<"==========too_far==============="<<endl;
      }
      //else if ( counter < 10 && angular_velocity[0] < 2 && angular_velocity[1] < 2 && angular_velocity[2] < 2 )
      //else if ( 1 < 300 && 1 < 20 && false == false && angular_velocity[0] < 1.0 && angular_velocity[1] < 1.0 && angular_velocity[2] < 1.0 )
      //else if ( abs(dot_product) < 0.7 && abs(trace) < 0.7 && abs(angular_velocity[0]) < 3.0 && abs(angular_velocity[1]) < 3.0 && abs(angular_velocity[2]) < 3.0 && abs(xyz_velocity[0]) < 0.2 && abs(xyz_velocity[1]) < 0.2 && abs(xyz_velocity[2]) < 0.2 )//*/
      else if (  abs(angular_velocity[0]) < 3.0 && abs(angular_velocity[1]) < 3.0 && abs(angular_velocity[2]) < 3.0  )//*/
      {
              //counter++;
              //cout<<"======[2]======["<<minDist<<"]"<<endl;
      	      // 插入的点总是少的 直接单线程插入
	      // 保存position位置对应的点
	      cout<<"======[2]======"<<endl;
	      insert_count++;
	      OctoTree *octo_tree =
		  new OctoTree(max_layer, 0, layer_point_size, max_points_size,
		               max_cov_points_size, planer_threshold);
	      feat_map[position] = octo_tree;
	      feat_map[position]->quater_length_ = voxel_size / 4;
	      feat_map[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
	      feat_map[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
	      feat_map[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
	      feat_map[position]->UpdateOctoTree(p_v);
	      //*/
      
      }
    }
  }
  double t_update_end = omp_get_wtime();
  //std::printf("Insert & store time:  %.4fs\n", t_update_end - t_update_start);
    t_update_start = omp_get_wtime();
    // 并行延迟更新
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for default(none) shared(position_index_map, feat_map)
#endif
    for (size_t b = 0; b < position_index_map.bucket_count(); b++) {
        // 先遍历bucket 理想情况下bucket一般只有一个元素 这样还是相当于完全并行的遍历position_index_map
        // XXX 需要确定最坏情况下bucket的元素数量
        for (auto bi = position_index_map.begin(b); bi != position_index_map.end(b); bi++) {
            VOXEL_LOC position = bi->first;
            for (const pointWithCov &p_v:bi->second) {
                feat_map[position]->UpdateOctoTree(p_v);
            }
        }
    }
    t_update_end = omp_get_wtime();
    //std::printf("Update:  %.4fs\n", t_update_end - t_update_start);

  //std::printf("Insert: %d  Update: %d \n", insert_count, update_count);
  
}


void build_single_residual(const pointWithCov &pv, const OctoTree *current_octo,
                           const int current_layer, const int max_layer,
                           const double sigma_num, bool &is_sucess,
                           double &prob, ptpl &single_ptpl) {
  double radius_k = 3;
  Eigen::Vector3d p_w = pv.point_world;
  // 如果当前voxel是平面 则构建voxel block 否则递归搜索当前voxel的leaves 直到找到平面
  // XXX 如果不是平面是不是可以在构建的时候直接剪掉？
  if (current_octo->plane_ptr_->is_plane) {
    Plane &plane = *current_octo->plane_ptr_;
    // HACK 这个是LiDAR点到地图plane的点面距离
    float dis_to_plane =
        fabs(plane.normal(0) * p_w(0) + plane.normal(1) * p_w(1) +
             plane.normal(2) * p_w(2) + plane.d);
    // HACK 这个是LiDAR点到构建地图plane的点簇中心的距离
    float dis_to_center =
        (plane.center(0) - p_w(0)) * (plane.center(0) - p_w(0)) +
        (plane.center(1) - p_w(1)) * (plane.center(1) - p_w(1)) +
        (plane.center(2) - p_w(2)) * (plane.center(2) - p_w(2));
    // HACK 差值是 点在平面上投影 与 平面点簇中心的距离
    // HACK 目的是不要用距离平面点簇太远的点来做残差，因为估计的平面在这些远点的位置可能不满足平面假设了
    // HACK 因为将点划分进voxel的时候只用了第一层voxel 这个voxel可能比较大 遍历到的这个子voxel距离点可能还比较远
    float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);

    if (range_dis <= radius_k * plane.radius) {
      // 计算点面距离的方差
      Eigen::Matrix<double, 1, 6> J_nq;
      J_nq.block<1, 3>(0, 0) = p_w - plane.center;
      J_nq.block<1, 3>(0, 3) = -plane.normal;
      double sigma_l = J_nq * plane.plane_cov * J_nq.transpose();
      sigma_l += plane.normal.transpose() * pv.cov * plane.normal;
      // 只选择距离在3sigma之内的匹配
      if (dis_to_plane < 30 * sqrt(sigma_l)) {
        is_sucess = true;
        // 求对应正态分布的概率密度值 意思是落在当前平面有多大可能性 注意这个分布的u=0 所以直接用dis_to_plane平方来求
        // HACK 这里比fast lio和任何loam系的都要clever得多
        double this_prob = 1.0 / (sqrt(sigma_l)) *
                           exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
        // 在递归的过程中不断比较 最后保留一个最大概率对应的residual
        if (this_prob > prob) {
          prob = this_prob;
          single_ptpl.point = pv.point;
          single_ptpl.point_world = pv.point_world;
          single_ptpl.plane_cov = plane.plane_cov;
          single_ptpl.normal = plane.normal;
          single_ptpl.center = plane.center;
          single_ptpl.d = plane.d;
          single_ptpl.layer = current_layer;
          single_ptpl.cov_lidar = pv.cov_lidar;
        }
        return;
      } else {
        // is_sucess = false;
        return;
      }
    } else {
      // is_sucess = false;
      return;
    }
  } else {
    if (current_layer < max_layer) {
      // 遍历当前节点的所有叶子 往下递归
      for (size_t leafnum = 0; leafnum < 8; leafnum++) {
        if (current_octo->leaves_[leafnum] != nullptr) {

          OctoTree *leaf_octo = current_octo->leaves_[leafnum];
          build_single_residual(pv, leaf_octo, current_layer + 1, max_layer,
                                sigma_num, is_sucess, prob, single_ptpl);
        }
      }
      return;
    } else {
      // is_sucess = false;
      return;
    }
  }
}

void GetUpdatePlane(const OctoTree *current_octo, const int pub_max_voxel_layer,
                    std::vector<Plane> &plane_list) {
  if (current_octo->layer_ > pub_max_voxel_layer) {
    return;
  }
  if (current_octo->plane_ptr_->is_update) {
    plane_list.push_back(*current_octo->plane_ptr_);
  }
  if (current_octo->layer_ < current_octo->max_layer_) {
    if (!current_octo->plane_ptr_->is_plane) {
      for (size_t i = 0; i < 8; i++) {
        if (current_octo->leaves_[i] != nullptr) {
          GetUpdatePlane(current_octo->leaves_[i], pub_max_voxel_layer,
                         plane_list);
        }
      }
    }
  }
  return;
}

// void BuildResidualListTBB(const unordered_map<VOXEL_LOC, OctoTree *>
// &voxel_map,
//                           const double voxel_size, const double sigma_num,
//                           const int max_layer,
//                           const std::vector<pointWithCov> &pv_list,
//                           std::vector<ptpl> &ptpl_list,
//                           std::vector<Eigen::Vector3d> &non_match) {
//   std::mutex mylock;
//   ptpl_list.clear();
//   std::vector<ptpl> all_ptpl_list(pv_list.size());
//   std::vector<bool> useful_ptpl(pv_list.size());
//   std::vector<size_t> index(pv_list.size());
//   for (size_t i = 0; i < index.size(); ++i) {
//     index[i] = i;
//     useful_ptpl[i] = false;
//   }
//   std::for_each(
//       std::execution::par_unseq, index.begin(), index.end(),
//       [&](const size_t &i) {
//         pointWithCov pv = pv_list[i];
//         float loc_xyz[3];
//         for (int j = 0; j < 3; j++) {
//           loc_xyz[j] = pv.point_world[j] / voxel_size;
//           if (loc_xyz[j] < 0) {
//             loc_xyz[j] -= 1.0;
//           }
//         }
//         VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
//                            (int64_t)loc_xyz[2]);
//         auto iter = voxel_map.find(position);
//         if (iter != voxel_map.end()) {
//           OctoTree *current_octo = iter->second;
//           ptpl single_ptpl;
//           bool is_sucess = false;
//           double prob = 0;
//           build_single_residual(pv, current_octo, 0, max_layer, sigma_num,
//                                 is_sucess, prob, single_ptpl);
//           if (!is_sucess) {
//             VOXEL_LOC near_position = position;
//             if (loc_xyz[0] > (current_octo->voxel_center_[0] +
//                               current_octo->quater_length_)) {
//               near_position.x = near_position.x + 1;
//             } else if (loc_xyz[0] < (current_octo->voxel_center_[0] -
//                                      current_octo->quater_length_)) {
//               near_position.x = near_position.x - 1;
//             }
//             if (loc_xyz[1] > (current_octo->voxel_center_[1] +
//                               current_octo->quater_length_)) {
//               near_position.y = near_position.y + 1;
//             } else if (loc_xyz[1] < (current_octo->voxel_center_[1] -
//                                      current_octo->quater_length_)) {
//               near_position.y = near_position.y - 1;
//             }
//             if (loc_xyz[2] > (current_octo->voxel_center_[2] +
//                               current_octo->quater_length_)) {
//               near_position.z = near_position.z + 1;
//             } else if (loc_xyz[2] < (current_octo->voxel_center_[2] -
//                                      current_octo->quater_length_)) {
//               near_position.z = near_position.z - 1;
//             }
//             auto iter_near = voxel_map.find(near_position);
//             if (iter_near != voxel_map.end()) {
//               build_single_residual(pv, iter_near->second, 0, max_layer,
//                                     sigma_num, is_sucess, prob, single_ptpl);
//             }
//           }
//           if (is_sucess) {

//             mylock.lock();
//             useful_ptpl[i] = true;
//             all_ptpl_list[i] = single_ptpl;
//             mylock.unlock();
//           } else {
//             mylock.lock();
//             useful_ptpl[i] = false;
//             mylock.unlock();
//           }
//         }
//       });
//   for (size_t i = 0; i < useful_ptpl.size(); i++) {
//     if (useful_ptpl[i]) {
//       ptpl_list.push_back(all_ptpl_list[i]);
//     }
//   }
// }

void BuildResidualListOMP(const unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
                          const double voxel_size, const double sigma_num,
                          const int max_layer,
                          const std::vector<pointWithCov> &pv_list,
                          std::vector<ptpl> &ptpl_list,
                          std::vector<Eigen::Vector3d> &non_match) {
  std::mutex mylock;
  ptpl_list.clear();
  std::vector<ptpl> all_ptpl_list(pv_list.size());
  std::vector<bool> useful_ptpl(pv_list.size());
  std::vector<size_t> index(pv_list.size());
  for (size_t i = 0; i < index.size(); ++i) {
    index[i] = i;
    useful_ptpl[i] = false;
  }
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
  // 这个文章在实现的时候 第一层voxel并没有严格作为根节点，而是现有一个层次的结构，这样方便管理
  for (int i = 0; i < index.size(); i++) {
    pointWithCov pv = pv_list[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = pv.point_world[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    // 查找当前点所属的voxel
    auto iter = voxel_map.find(position);

    if (iter != voxel_map.end()) {
      OctoTree *current_octo = iter->second;
      ptpl single_ptpl;
      bool is_sucess = false;
      double prob = 0;
      // 找到之后构建residual 返回值是single_ptpl 包含了与点匹配的平面的所有信息
      build_single_residual(pv, current_octo, 0, max_layer, sigma_num,
                            is_sucess, prob, single_ptpl);
      // 如果不成功 根据当前点偏离voxel的程度 查找临近的voxel
      // HACK 这里是为了处理点落在两个voxel边界的情况 可能真实匹配的平面在临近的voxel中
      if (!is_sucess) {
        VOXEL_LOC near_position = position;
        if (loc_xyz[0] >
            (current_octo->voxel_center_[0] + current_octo->quater_length_)) {
          near_position.x = near_position.x + 1;
        } else if (loc_xyz[0] < (current_octo->voxel_center_[0] -
                                 current_octo->quater_length_)) {
          near_position.x = near_position.x - 1;
        }
        if (loc_xyz[1] >
            (current_octo->voxel_center_[1] + current_octo->quater_length_)) {
          near_position.y = near_position.y + 1;
        } else if (loc_xyz[1] < (current_octo->voxel_center_[1] -
                                 current_octo->quater_length_)) {
          near_position.y = near_position.y - 1;
        }
        if (loc_xyz[2] >
            (current_octo->voxel_center_[2] + current_octo->quater_length_)) {
          near_position.z = near_position.z + 1;
        } else if (loc_xyz[2] < (current_octo->voxel_center_[2] -
                                 current_octo->quater_length_)) {
          near_position.z = near_position.z - 1;
        }
        auto iter_near = voxel_map.find(near_position);
        if (iter_near != voxel_map.end()) {
          build_single_residual(pv, iter_near->second, 0, max_layer, sigma_num,
                                is_sucess, prob, single_ptpl);
        }
      }

      // 所有点的匹配结果储存到list中
      if (is_sucess) {

        mylock.lock();
        useful_ptpl[i] = true;
        all_ptpl_list[i] = single_ptpl;
        mylock.unlock();
      } else {
        mylock.lock();
        useful_ptpl[i] = false;
        mylock.unlock();
      }
    }
  }
  for (size_t i = 0; i < useful_ptpl.size(); i++) {
    if (useful_ptpl[i]) {
      ptpl_list.push_back(all_ptpl_list[i]);
    }
  }
}

void BuildResidualListNormal(
    const unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
    const double voxel_size, const double sigma_num, const int max_layer,
    const std::vector<pointWithCov> &pv_list, std::vector<ptpl> &ptpl_list,
    std::vector<Eigen::Vector3d> &non_match) {
  ptpl_list.clear();
  std::vector<size_t> index(pv_list.size());
  for (size_t i = 0; i < pv_list.size(); ++i) {
    pointWithCov pv = pv_list[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = pv.point_world[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = voxel_map.find(position);
    if (iter != voxel_map.end()) {
      OctoTree *current_octo = iter->second;
      ptpl single_ptpl;
      bool is_sucess = false;
      double prob = 0;
      build_single_residual(pv, current_octo, 0, max_layer, sigma_num,
                            is_sucess, prob, single_ptpl);

      if (!is_sucess) {
        VOXEL_LOC near_position = position;
        if (loc_xyz[0] >
            (current_octo->voxel_center_[0] + current_octo->quater_length_)) {
          near_position.x = near_position.x + 1;
        } else if (loc_xyz[0] < (current_octo->voxel_center_[0] -
                                 current_octo->quater_length_)) {
          near_position.x = near_position.x - 1;
        }
        if (loc_xyz[1] >
            (current_octo->voxel_center_[1] + current_octo->quater_length_)) {
          near_position.y = near_position.y + 1;
        } else if (loc_xyz[1] < (current_octo->voxel_center_[1] -
                                 current_octo->quater_length_)) {
          near_position.y = near_position.y - 1;
        }
        if (loc_xyz[2] >
            (current_octo->voxel_center_[2] + current_octo->quater_length_)) {
          near_position.z = near_position.z + 1;
        } else if (loc_xyz[2] < (current_octo->voxel_center_[2] -
                                 current_octo->quater_length_)) {
          near_position.z = near_position.z - 1;
        }
        auto iter_near = voxel_map.find(near_position);
        if (iter_near != voxel_map.end()) {
          build_single_residual(pv, iter_near->second, 0, max_layer, sigma_num,
                                is_sucess, prob, single_ptpl);
        }
      }
      if (is_sucess) {
        ptpl_list.push_back(single_ptpl);
      } else {
        non_match.push_back(pv.point_world);
      }
    }
  }
}

void CalcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec,
                     const Eigen::Vector3d &z_vec,
                     geometry_msgs::Quaternion &q) {

  Eigen::Matrix3d rot;
  rot << x_vec(0), x_vec(1), x_vec(2), y_vec(0), y_vec(1), y_vec(2), z_vec(0),
      z_vec(1), z_vec(2);
  Eigen::Matrix3d rotation = rot.transpose();
  Eigen::Quaterniond eq(rotation);
  q.w = eq.w();
  q.x = eq.x();
  q.y = eq.y();
  q.z = eq.z();
}

void CalcQuation(const Eigen::Vector3d &vec, const int axis,
                 geometry_msgs::Quaternion &q) {
  Eigen::Vector3d x_body = vec;
  Eigen::Vector3d y_body(1, 1, 0);
  if (x_body(2) != 0) {
    y_body(2) = -(y_body(0) * x_body(0) + y_body(1) * x_body(1)) / x_body(2);
  } else {
    if (x_body(1) != 0) {
      y_body(1) = -(y_body(0) * x_body(0)) / x_body(1);
    } else {
      y_body(0) = 0;
    }
  }
  y_body.normalize();
  Eigen::Vector3d z_body = x_body.cross(y_body);
  Eigen::Matrix3d rot;

  rot << x_body(0), x_body(1), x_body(2), y_body(0), y_body(1), y_body(2),
      z_body(0), z_body(1), z_body(2);
  Eigen::Matrix3d rotation = rot.transpose();
  if (axis == 2) {
    Eigen::Matrix3d rot_inc;
    rot_inc << 0, 0, 1, 0, 1, 0, -1, 0, 0;
    rotation = rotation * rot_inc;
  }
  Eigen::Quaterniond eq(rotation);
  q.w = eq.w();
  q.x = eq.x();
  q.y = eq.y();
  q.z = eq.z();
}

void pubSinglePlane(visualization_msgs::MarkerArray &plane_pub,
                    const std::string plane_ns, const Plane &single_plane,
                    const float alpha, const Eigen::Vector3d rgb) {
  visualization_msgs::Marker plane;
  plane.header.frame_id = "camera_init";
  plane.header.stamp = ros::Time();
  plane.ns = plane_ns;
  plane.id = single_plane.id;
  plane.type = visualization_msgs::Marker::CYLINDER;
  plane.action = visualization_msgs::Marker::ADD;
  plane.pose.position.x = single_plane.center[0];
  plane.pose.position.y = single_plane.center[1];
  plane.pose.position.z = single_plane.center[2];
  geometry_msgs::Quaternion q;
  CalcVectQuation(single_plane.x_normal, single_plane.y_normal,
                  single_plane.normal, q);
  plane.pose.orientation = q;
  plane.scale.x = 3 * sqrt(single_plane.max_eigen_value);
  plane.scale.y = 3 * sqrt(single_plane.mid_eigen_value);
  plane.scale.z = 2 * sqrt(single_plane.min_eigen_value);
  plane.color.a = alpha;
  plane.color.r = rgb(0);
  plane.color.g = rgb(1);
  plane.color.b = rgb(2);
  plane.lifetime = ros::Duration();
  plane_pub.markers.push_back(plane);
}

void pubNoPlaneMap(const std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map,
                   const ros::Publisher &plane_map_pub) {
  int id = 0;
  ros::Rate loop(500);
  float use_alpha = 0.8;
  visualization_msgs::MarkerArray voxel_plane;
  voxel_plane.markers.reserve(1000000);
  for (auto iter = feat_map.begin(); iter != feat_map.end(); iter++) {
    if (!iter->second->plane_ptr_->is_plane) {
      for (uint i = 0; i < 8; i++) {
        if (iter->second->leaves_[i] != nullptr) {
          OctoTree *temp_octo_tree = iter->second->leaves_[i];
          if (!temp_octo_tree->plane_ptr_->is_plane) {
            for (uint j = 0; j < 8; j++) {
              if (temp_octo_tree->leaves_[j] != nullptr) {
                if (!temp_octo_tree->leaves_[j]->plane_ptr_->is_plane) {
                  Eigen::Vector3d plane_rgb(1, 1, 1);
                  pubSinglePlane(voxel_plane, "no_plane",
                                 *(temp_octo_tree->leaves_[j]->plane_ptr_),
                                 use_alpha, plane_rgb);
                }
              }
            }
          }
        }
      }
    }
  }
  plane_map_pub.publish(voxel_plane);
  loop.sleep();
}

void pubVoxelMap(const std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
                 const int pub_max_voxel_layer,
                 const ros::Publisher &plane_map_pub) {
  double max_trace = 0.25;
  double pow_num = 0.2;
  ros::Rate loop(500);
  float use_alpha = 0.8;
  visualization_msgs::MarkerArray voxel_plane;
  voxel_plane.markers.reserve(1000000);
  std::vector<Plane> pub_plane_list;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    GetUpdatePlane(iter->second, pub_max_voxel_layer, pub_plane_list);
  }
  for (size_t i = 0; i < pub_plane_list.size(); i++) {
    V3D plane_cov = pub_plane_list[i].plane_cov.block<3, 3>(0, 0).diagonal();
    double trace = plane_cov.sum();
    if (trace >= max_trace) {
      trace = max_trace;
    }
    trace = trace * (1.0 / max_trace);
    trace = pow(trace, pow_num);
    uint8_t r, g, b;
    mapJet(trace, 0, 1, r, g, b);
    Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
    double alpha;
    if (pub_plane_list[i].is_plane) {
      alpha = use_alpha;
    } else {
      alpha = 0;
    }
    pubSinglePlane(voxel_plane, "plane", pub_plane_list[i], alpha, plane_rgb);
  }
  plane_map_pub.publish(voxel_plane);
  loop.sleep();
}

void pubPlaneMap(const std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map,
                 const ros::Publisher &plane_map_pub) {
  OctoTree *current_octo = nullptr;

  double max_trace = 0.25;
  double pow_num = 0.2;
  ros::Rate loop(500);
  float use_alpha = 1.0;
  visualization_msgs::MarkerArray voxel_plane;
  voxel_plane.markers.reserve(1000000);

  for (auto iter = feat_map.begin(); iter != feat_map.end(); iter++) {
    if (iter->second->plane_ptr_->is_update) {
      Eigen::Vector3d normal_rgb(0.0, 1.0, 0.0);

      V3D plane_cov =
          iter->second->plane_ptr_->plane_cov.block<3, 3>(0, 0).diagonal();
      double trace = plane_cov.sum();
      if (trace >= max_trace) {
        trace = max_trace;
      }
      trace = trace * (1.0 / max_trace);
      trace = pow(trace, pow_num);
      uint8_t r, g, b;
      mapJet(trace, 0, 1, r, g, b);
      Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
      // Eigen::Vector3d plane_rgb(1, 0, 0);
      float alpha = 0.0;
      if (iter->second->plane_ptr_->is_plane) {
        alpha = use_alpha;
      } else {
        // std::cout << "delete plane" << std::endl;
      }
      // if (iter->second->update_enable_) {
      //   plane_rgb << 1, 0, 0;
      // } else {
      //   plane_rgb << 0, 0, 1;
      // }
      pubSinglePlane(voxel_plane, "plane", *(iter->second->plane_ptr_), alpha,
                     plane_rgb);

      iter->second->plane_ptr_->is_update = false;
    } else {
      for (uint i = 0; i < 8; i++) {
        if (iter->second->leaves_[i] != nullptr) {
          if (iter->second->leaves_[i]->plane_ptr_->is_update) {
            Eigen::Vector3d normal_rgb(0.0, 1.0, 0.0);

            V3D plane_cov = iter->second->leaves_[i]
                                ->plane_ptr_->plane_cov.block<3, 3>(0, 0)
                                .diagonal();
            double trace = plane_cov.sum();
            if (trace >= max_trace) {
              trace = max_trace;
            }
            trace = trace * (1.0 / max_trace);
            // trace = (max_trace - trace) / max_trace;
            trace = pow(trace, pow_num);
            uint8_t r, g, b;
            mapJet(trace, 0, 1, r, g, b);
            Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
            plane_rgb << 0, 1, 0;
            // fabs(iter->second->leaves_[i]->plane_ptr_->normal[0]),
            //     fabs(iter->second->leaves_[i]->plane_ptr_->normal[1]),
            //     fabs(iter->second->leaves_[i]->plane_ptr_->normal[2]);
            float alpha = 0.0;
            if (iter->second->leaves_[i]->plane_ptr_->is_plane) {
              alpha = use_alpha;
            } else {
              // std::cout << "delete plane" << std::endl;
            }
            pubSinglePlane(voxel_plane, "plane",
                           *(iter->second->leaves_[i]->plane_ptr_), alpha,
                           plane_rgb);
            // loop.sleep();
            iter->second->leaves_[i]->plane_ptr_->is_update = false;
            // loop.sleep();
          } else {
            OctoTree *temp_octo_tree = iter->second->leaves_[i];
            for (uint j = 0; j < 8; j++) {
              if (temp_octo_tree->leaves_[j] != nullptr) {
                if (temp_octo_tree->leaves_[j]->octo_state_ == 0 &&
                    temp_octo_tree->leaves_[j]->plane_ptr_->is_update) {
                  if (temp_octo_tree->leaves_[j]->plane_ptr_->is_plane) {
                    // std::cout << "subsubplane" << std::endl;
                    Eigen::Vector3d normal_rgb(0.0, 1.0, 0.0);
                    V3D plane_cov =
                        temp_octo_tree->leaves_[j]
                            ->plane_ptr_->plane_cov.block<3, 3>(0, 0)
                            .diagonal();
                    double trace = plane_cov.sum();
                    if (trace >= max_trace) {
                      trace = max_trace;
                    }
                    trace = trace * (1.0 / max_trace);
                    // trace = (max_trace - trace) / max_trace;
                    trace = pow(trace, pow_num);
                    uint8_t r, g, b;
                    mapJet(trace, 0, 1, r, g, b);
                    Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
                    plane_rgb << 0, 0, 1;
                    float alpha = 0.0;
                    if (temp_octo_tree->leaves_[j]->plane_ptr_->is_plane) {
                      alpha = use_alpha;
                    }

                    pubSinglePlane(voxel_plane, "plane",
                                   *(temp_octo_tree->leaves_[j]->plane_ptr_),
                                   alpha, plane_rgb);
                    // loop.sleep();
                    temp_octo_tree->leaves_[j]->plane_ptr_->is_update = false;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  plane_map_pub.publish(voxel_plane);
  // plane_map_pub.publish(voxel_norm);
  loop.sleep();
  // cout << "[Map Info] Plane counts:" << plane_count
  //      << " Sub Plane counts:" << sub_plane_count
  //      << " Sub Sub Plane counts:" << sub_sub_plane_count << endl;
  // cout << "[Map Info] Update plane counts:" << update_count
  //      << "total size: " << feat_map.size() << endl;
}

M3D calcBodyCov(Eigen::Vector3d &pb, const float range_inc, const float degree_inc)
{
  float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
  float range_var = range_inc * range_inc;
  Eigen::Matrix2d direction_var;
  direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0,
      pow(sin(DEG2RAD(degree_inc)), 2);
  Eigen::Vector3d direction(pb);
  direction.normalize();
  Eigen::Matrix3d direction_hat;
  direction_hat << 0, -direction(2), direction(1), direction(2), 0,
      -direction(0), -direction(1), direction(0), 0;
  Eigen::Vector3d base_vector1(1, 1,
                               -(direction(0) + direction(1)) / direction(2));
  base_vector1.normalize();
  Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
  base_vector2.normalize();
  Eigen::Matrix<double, 3, 2> N;
  N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1),
      base_vector1(2), base_vector2(2);
  Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
  return direction * range_var * direction.transpose() +
        A * direction_var * A.transpose();
};

#endif
