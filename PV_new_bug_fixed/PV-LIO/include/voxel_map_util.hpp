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

#define HASH_P 116101
#define MAX_N 10000000000


extern esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
extern state_ikfom state_point;
extern vect3 position_lidar;

static int plane_id = 0;

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


  ~OctoTree() {
  
    // Recursively delete any child sub-trees
    for (int i = 0; i < 8; ++i) {
        if (leaves_[i]) {
            delete leaves_[i];
            leaves_[i] = nullptr;
        }
    }
    
    // Delete the plane pointer
    if (plane_ptr_) {
        delete plane_ptr_;
        plane_ptr_ = nullptr;
    }
    
  }
  

  // check is plane , calc plane parameters including plane covariance
  void init_plane(const std::vector<pointWithCov> &points, Plane *plane) {
  
  
    // every octo map has one plane_ptr_ of plane pointer
    //initilze the plane
    plane->plane_cov = Eigen::Matrix<double, 6, 6>::Zero();
    plane->covariance = Eigen::Matrix3d::Zero();
    plane->center = Eigen::Vector3d::Zero();
    plane->normal = Eigen::Vector3d::Zero();
    plane->points_size = points.size();
    plane->radius = 0;
    
    // calculate the point covariance and center
    for (auto pv : points) {
      plane->covariance += pv.point * pv.point.transpose();
      plane->center += pv.point;
    }
    
    // calculate the point covariance and center, means
    plane->center = plane->center / plane->points_size;
    plane->covariance = plane->covariance / plane->points_size - plane->center * plane->center.transpose();
    
    // EigenSolver
    Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance);
    
    //evecs, eigen vector
    Eigen::Matrix3cd evecs = es.eigenvectors();
    //evals, eigen value
    Eigen::Vector3cd evals = es.eigenvalues();
    //evalsReal eigne value real
    Eigen::Vector3d evalsReal;
    evalsReal = evals.real();
    
    //evalsMin, evalsMax, max eigen value index, min eigen value index, 
    Eigen::Matrix3f::Index evalsMin, evalsMax;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    
    //evalsMid, middle eigne value index
    int evalsMid = 3 - evalsMin - evalsMax;
    
    //eigen vector minium, middle, max
    Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
    Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
    Eigen::Vector3d evecMax = evecs.real().col(evalsMax);
    // plane covariance calculation
    
    // matrix J_Q
    Eigen::Matrix3d J_Q;
    
    J_Q << 1.0 / plane->points_size, 0, 0, 0, 1.0 / plane->points_size, 0, 0, 0,
        1.0 / plane->points_size;
        
    // if less than the planer_threshold_
    if (evalsReal(evalsMin) < planer_threshold_) {
    
      // create the index
      std::vector<int> index(points.size());
      std::vector<Eigen::Matrix<double, 6, 6>> temp_matrix(points.size());
      
      // for all points
      for (int i = 0; i < points.size(); i++) {
      
        // make the J, F
        Eigen::Matrix<double, 6, 3> J;
        Eigen::Matrix3d F;
        
        // 
        for (int m = 0; m < 3; m++) {
        
          // if m not equal to the eigen value minium
          if (m != (int)evalsMin) {
          
            // create the matrix of F_m
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
        
        // get the covariance of the plane
        plane->plane_cov += J * points[i].cov * J.transpose();
      }

      //get the plane noraml
      plane->normal   << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
      plane->y_normal << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid), evecs.real()(2, evalsMid);
      plane->x_normal << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax), evecs.real()(2, evalsMax);
      
      plane->min_eigen_value = evalsReal(evalsMin);
      plane->mid_eigen_value = evalsReal(evalsMid);
      plane->max_eigen_value = evalsReal(evalsMax);
      
      //get the plane radius
      plane->radius = sqrt(evalsReal(evalsMax));
      
      //get the plane center distance
      plane->d = -(plane->normal(0) * plane->center(0) +
                   plane->normal(1) * plane->center(1) +
                   plane->normal(2) * plane->center(2));
      // this plane is plane
      plane->is_plane = true;
      
      // is updated of the plane
      if (plane->last_update_points_size == 0) {
        plane->last_update_points_size = plane->points_size;
        plane->is_update = true;
      } else if (plane->points_size - plane->last_update_points_size > 100) {
        plane->last_update_points_size = plane->points_size;
        plane->is_update = true;
      }

      // if the plane is not initilized
      if (!plane->is_init) {
        plane->id = plane_id;
        plane_id++;
        plane->is_init = true;
      }

    } else {
      
      // if the plane is not initilized
      if (!plane->is_init) {
        plane->id = plane_id;
        plane_id++;
        plane->is_init = true;
      }
      
      // is updated of the plane
      if (plane->last_update_points_size == 0) {
        plane->last_update_points_size = plane->points_size;
        plane->is_update = true;
      } else if (plane->points_size - plane->last_update_points_size > 100) {
        plane->last_update_points_size = plane->points_size;
        plane->is_update = true;
      }
      
      // the plane is not a plane, but calculate the parameter of the plane
      plane->is_plane = false;
      plane->normal   << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
      plane->y_normal << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid), evecs.real()(2, evalsMid);
      plane->x_normal << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax), evecs.real()(2, evalsMax);
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
    
    // old covariance and old center
    Eigen::Matrix3d old_covariance = plane->covariance;
    Eigen::Vector3d old_center = plane->center;
    
    Eigen::Matrix3d sum_ppt = (plane->covariance + plane->center * plane->center.transpose()) * plane->points_size;
        
    Eigen::Vector3d sum_p = plane->center * plane->points_size;
    
    // sum new point to the map old covariance and old center, get new covariance and new center
    for (size_t i = 0; i < points.size(); i++) {
    
      Eigen::Vector3d pv = points[i].point;
      sum_ppt += pv * pv.transpose();
      sum_p += pv;
      
    }
    
    // get the plane points_size, center, covariance
    plane->points_size = plane->points_size + points.size();
    plane->center = sum_p / plane->points_size;
    plane->covariance = sum_ppt / plane->points_size - plane->center * plane->center.transpose();
    
    // the eigen vector and eigen value of the plane by its covariance
    Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance);
    
    Eigen::Matrix3cd evecs = es.eigenvectors();
    Eigen::Vector3cd evals = es.eigenvalues();
    Eigen::Vector3d evalsReal;
    
    evalsReal = evals.real();
    
    // index of minium and max eigne value
    Eigen::Matrix3d::Index evalsMin, evalsMax;
    
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    
    // index of middle eigne value
    int evalsMid = 3 - evalsMin - evalsMax;
    
    Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
    Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
    Eigen::Vector3d evecMax = evecs.real().col(evalsMax);
    
    // if it is a plane by the planer_threshold_
    if (evalsReal(evalsMin) < planer_threshold_) {
    
      // set all the plane parameters
      plane->normal << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
      plane->y_normal << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid), evecs.real()(2, evalsMid);
      plane->x_normal << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax), evecs.real()(2, evalsMax);
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
    
      // if it is not a plane by the planer_threshold_
      plane->normal   << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
      plane->y_normal << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid), evecs.real()(2, evalsMid);
      plane->x_normal << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax), evecs.real()(2, evalsMax);
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
  
    // if not initilize the octo map
    if (!init_octo_) {
      
      // add the number
      new_points_num_++;
      all_points_num_++;
      
      // temp_point push back this point
      temp_points_.push_back(pv);
      
      if (temp_points_.size() > max_plane_update_threshold_) {
      
        //renew the octo map
        init_octo_tree();
      }
      
    } else {
    
      // if it is a plane    
      if (plane_ptr_->is_plane) {
      
        // update is enabled      
        if (update_enable_) {
        
          // add the number
          new_points_num_++;
          all_points_num_++;
          
          // if the covariane is enabled
          if (update_cov_enable_) {
            
            //temp point push back this point
            temp_points_.push_back(pv);
            
          } else {
            //new point push back this point
            new_points_.push_back(pv);
            
          }
          
          // if new point number is bigger than setting
          if (new_points_num_ > update_size_threshold_) {
          
            // initial and reset the plane
            if (update_cov_enable_) {
              init_plane(temp_points_, plane_ptr_);
            }
            
            // when reset, make the number = 0
            new_points_num_ = 0;
            
          }
          
          //clear the date in the temp_points_
          if (all_points_num_ >= max_cov_points_size_) {
          
            update_cov_enable_ = false;
            std::vector<pointWithCov>().swap(temp_points_);
            
          }
          
          //clear the date in the new_points_
          if (all_points_num_ >= max_points_size_) {
          
            update_enable_ = false;
            plane_ptr_->update_enable = false;
            std::vector<pointWithCov>().swap(new_points_);
            
          }
          
        } else {
          return;
        }
        
      } else {
      
        // less than the max layer
        if (layer_ < max_layer_) {
        
          //clear the date in the temp_points
          if (temp_points_.size() != 0) {
            std::vector<pointWithCov>().swap(temp_points_);
          }
          //clear the date in the new_points_
          if (new_points_.size() != 0) {
            std::vector<pointWithCov>().swap(new_points_);
          }
          
          // get the position of voxel
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
          
          // get the leafnum
          int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
          
          
          if (leaves_[leafnum] != nullptr) {
          
            //if we have the octo map, then update the plane          
            leaves_[leafnum]->UpdateOctoTree(pv);
            
          } else {
          
            //make the octo tree and give to the octo map
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
            
            //update the octotree with the point
            leaves_[leafnum]->UpdateOctoTree(pv);
            
          }
        } else {
        
          // if can update
          if (update_enable_) {
          
            //add the number
            new_points_num_++;
            all_points_num_++;
            
            // push back the point
            if (update_cov_enable_) {
            
              temp_points_.push_back(pv);
              
            } else {
            
              new_points_.push_back(pv);
              
            }
            
            // if bigger than the max number
            if (new_points_num_ > update_size_threshold_) {
            
              //reset the plan to zero
              if (update_cov_enable_) {
              
                init_plane(temp_points_, plane_ptr_);
                
              } else {
              
                update_plane(new_points_, plane_ptr_);
                new_points_.clear();
                
              }
              
              new_points_num_ = 0;
              
            }
            //clear the date in the temp_points_
            if (all_points_num_ >= max_cov_points_size_) {
            
              update_cov_enable_ = false;
              std::vector<pointWithCov>().swap(temp_points_);
              
            }
            //clear the date in the new_points_
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

void buildVoxelMap(const std::vector<pointWithCov> &input_points,
                   const float voxel_size, const int max_layer,
                   const std::vector<int> &layer_point_size,
                   const int max_points_size, const int max_cov_points_size,
                   const float planer_threshold,
                   std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map) {
                   
  // size of point list
  uint plsize = input_points.size();
  
  // do with every point
  for (uint i = 0; i < plsize; i++) {
  
    // get the i-th point with covariance
    const pointWithCov p_v = input_points[i];
    
    // location xyz
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_v.point[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    
    // Voxel position
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    
    // find the map by the position
    auto iter = feat_map.find(position);
    
    if (iter != feat_map.end()) {
    
      // make the point into the map finded by position
      feat_map[position]->temp_points_.push_back(p_v);
      feat_map[position]->new_points_num_++;
      
    } else {
    
      // make a octo_tree 
      OctoTree *octo_tree = new OctoTree(max_layer, 0, layer_point_size, max_points_size, max_cov_points_size, planer_threshold);
      
      // set the octo_tree the map finded by position
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
  
  // all of map finded by position, do init_octo_tree()
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
                    
    // size of point list   
    uint plsize = input_points.size();
    
    for (uint i = 0; i < plsize; i++) {
    
        // get the point
        const pointWithCov p_v = input_points[i];
        // get the position
        float loc_xyz[3];
        for (int j = 0; j < 3; j++) {
            loc_xyz[j] = p_v.point[j] / voxel_size;
            if (loc_xyz[j] < 0) {
                loc_xyz[j] -= 1.0;
            }
        }
        
        // key of octo_map by position
        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
        
        auto iter = feat_map.find(position);
        // 如果点的位置已经存在voxel 那么就更新点的位置 否则创建新的voxel
        if (iter != feat_map.end()) {
        
            // UpdateOctoTree() by one point
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
            
            // UpdateOctoTree() by one point
            feat_map[position]->UpdateOctoTree(p_v);
        }
    }
}

void updateVoxelMapOMP(const std::vector<pointWithCov> &input_points,
                    const float voxel_size, const int max_layer,
                    const std::vector<int> &layer_point_size,
                    const int max_points_size, const int max_cov_points_size,
                    const float planer_threshold,
                    std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map) {

  // get the voxel map
  std::unordered_map<VOXEL_LOC, vector<pointWithCov>> position_index_map;
  int insert_count = 0, update_count = 0;
  uint plsize = input_points.size();


  double t_update_start = omp_get_wtime();
  
  
  for (uint i = 0; i < plsize; i++) {
  
    //get one point
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
    } else {
      // 插入的点总是少的 直接单线程插入
      // 保存position位置对应的点
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
    }
  }
  double t_update_end = omp_get_wtime();
  std::printf("Insert & store time:  %.4fs\n", t_update_end - t_update_start);
  
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
  std::printf("Update:  %.4fs\n", t_update_end - t_update_start);

  std::printf("Insert: %d  Update: %d \n", insert_count, update_count);
  

    ///////LZLZLZ

  std::unordered_map<VOXEL_LOC, OctoTree *>::iterator it = feat_map.begin();
  
  while(it != feat_map.end())
  {
    VOXEL_LOC loc = it->first;

    
    if ( ( abs( loc.x * voxel_size - position_lidar[0] ) > 20 || abs( loc.y * voxel_size - position_lidar[1] ) > 20 || abs( loc.z * voxel_size - position_lidar[2] ) > 20 ) )
    {
    
      delete it->second;
      it->second =NULL;
      feat_map.erase(it++);
         
      //std::printf("delete  VOXEL_LOC : x = %d, y = %d, z = %d\n", loc.x, loc.y, loc.z );
      
    }  
    else
    {
      it++;
    }
  }
  ///////LZLZLZ  
  
  
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
      if (dis_to_plane < sigma_num * sqrt(sigma_l)) {
        is_sucess = true;
        // 求对应正态分布的概率密度值 意思是落在当前平面有多大可能性 注意这个分布的u=0 所以直接用dis_to_plane平方来求
        // HACK 这里比fast lio和任何loam系的都要clever得多
        double this_prob = 1.0 / (sqrt(sigma_l)) *
                           exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
        // 在递归的过程中不断比较 最后保留一个最大概率PDF对应的residual
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
  
  //all pv_list size point
  for (int i = 0; i < index.size(); i++) {
    // one point
    pointWithCov pv = pv_list[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = pv.point_world[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    
    // get the position
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    // 查找当前点所属的voxel
    auto iter = voxel_map.find(position);

    if (iter != voxel_map.end()) {
    
      // get the octo map
      OctoTree *current_octo = iter->second;
      ptpl single_ptpl;
      bool is_sucess = false;
      double prob = 0;
      // 找到之后构建residual 返回值是single_ptpl 包含了与点匹配的平面的所有信息
      build_single_residual(pv, current_octo, 0, max_layer, sigma_num,
                            is_sucess, prob, single_ptpl);
      // 如果不成功 根据当前点偏离voxel的程度 查找临近的voxel
      // HACK 这里是为了处理点落在两个voxel边界的情况 可能真实匹配的平面在临近的voxel中
      
      // if not build single residual
      if (!is_sucess) {
      
        // get the position
        VOXEL_LOC near_position = position;
        
        if (loc_xyz[0] > (current_octo->voxel_center_[0] + current_octo->quater_length_)) {
          near_position.x = near_position.x + 1;
        } else if (loc_xyz[0] < (current_octo->voxel_center_[0] -
                                 current_octo->quater_length_)) {
          near_position.x = near_position.x - 1;
        }
        
        if (loc_xyz[1] > (current_octo->voxel_center_[1] + current_octo->quater_length_)) {
          near_position.y = near_position.y + 1;
        } else if (loc_xyz[1] < (current_octo->voxel_center_[1] -
                                 current_octo->quater_length_)) {
          near_position.y = near_position.y - 1;
        }
        
        if (loc_xyz[2] > (current_octo->voxel_center_[2] + current_octo->quater_length_)) {
          near_position.z = near_position.z + 1;
        } else if (loc_xyz[2] < (current_octo->voxel_center_[2] -
                                 current_octo->quater_length_)) {
          near_position.z = near_position.z - 1;
        }
        
        
        //build single residual in nearby voxel
        auto iter_near = voxel_map.find(near_position);
        if (iter_near != voxel_map.end()) { build_single_residual(pv, iter_near->second, 0, max_layer, sigma_num,
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
                 
  //cout<<"===========pubVoxelMap[0]=========="<<endl;
                 
  double max_trace = 0.25;
  double pow_num = 0.2;
  ros::Rate loop(500);
  float use_alpha = 0.8;
  visualization_msgs::MarkerArray voxel_plane;
  voxel_plane.markers.reserve(1000000);
  std::vector<Plane> pub_plane_list;
  
  //cout<<voxel_map.size()<<endl;
  
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    GetUpdatePlane(iter->second, pub_max_voxel_layer, pub_plane_list);
  }
  //cout<<"===========pubVoxelMap[1]=========="<<endl;
  //cout<<pub_plane_list.size()<<endl;
  
  for (size_t i = 0; i < pub_plane_list.size() / 10; i++) {
    V3D plane_cov = pub_plane_list[i].plane_cov.block<3, 3>(0, 0).diagonal();
    //cout<<"===========pubVoxelMap[1.1]=========="<<endl;
    //cout<<plane_cov<<endl;
    double trace = plane_cov.sum();
    if (trace >= max_trace) {
      trace = max_trace;
    }
    trace = trace * (1.0 / max_trace);
    trace = pow(trace, pow_num);
    uint8_t r, g, b;
    
    // trace of P is the rgb color
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
  //cout<<"===========pubVoxelMap[2]=========="<<endl;
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

// range_inc = 0.02
// degree_inc = 0.05

M3D calcBodyCov(Eigen::Vector3d &pb, const float range_inc, const float degree_inc)
{
  float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
  float range_var = range_inc * range_inc;
  
  Eigen::Matrix2d direction_var;
  direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0, pow(sin(DEG2RAD(degree_inc)), 2);
  
  Eigen::Vector3d direction(pb);
  direction.normalize();
  
  Eigen::Matrix3d direction_hat;
  direction_hat << 0, -direction(2), direction(1), direction(2), 0, -direction(0), -direction(1), direction(0), 0;
  
  Eigen::Vector3d base_vector1(1, 1, -(direction(0) + direction(1)) / direction(2));
  base_vector1.normalize();
  
  Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
  base_vector2.normalize();
  
  Eigen::Matrix<double, 3, 2> N;
  N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1), base_vector1(2), base_vector2(2);
  Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
  
  return direction * range_var * direction.transpose() +  A * direction_var * A.transpose();
};

#endif
