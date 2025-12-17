/*********************************************************************
*
* Software License Agreement (BSD License)
*
*  Copyright (c) 2008, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of Willow Garage, Inc. nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
* Authors: Eitan Marder-Eppstein, Sachin Chitta
*********************************************************************/
#include <angles/angles.h>
#include <carrot_planner/carrot_planner.h>
#include <pluginlib/class_list_macros.hpp>
#include <tf2/convert.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <carrot_planner/navfn.h>

#include <costmap_2d/cost_values.h>
#include <costmap_2d/costmap_2d.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#include <iostream>


using namespace std;

using namespace navfn;

//register this planner as a BaseGlobalPlanner plugin
PLUGINLIB_EXPORT_CLASS(carrot_planner::CarrotPlanner, nav_core::BaseGlobalPlanner)

namespace carrot_planner {

  CarrotPlanner::CarrotPlanner() : costmap_ros_(NULL), costmap_(NULL),  planner_(), initialized_(false), visualize_potential_(false), allow_unknown_(true)
  {
  
  }

  CarrotPlanner::CarrotPlanner(std::string name, costmap_2d::Costmap2DROS* costmap_ros) : costmap_ros_(NULL), costmap_(NULL), planner_(), initialized_(false), visualize_potential_(false), allow_unknown_(true)
  {
     initialize(name, costmap_ros);
  }

  CarrotPlanner::CarrotPlanner(std::string name, costmap_2d::Costmap2D* costmap, std::string global_frame)
    : costmap_ros_(NULL), costmap_(NULL),  planner_(), initialized_(false), visualize_potential_(false), allow_unknown_(true) {
      //initialize the planner
      initialize(name, costmap, global_frame);
  }


  CarrotPlanner::~CarrotPlanner() {

  }

  void CarrotPlanner::initialize(std::string name, costmap_2d::Costmap2DROS* costmap_ros){
    costmap_ros_ = costmap_ros;
    initialize(name, costmap_ros->getCostmap(), costmap_ros->getGlobalFrameID());
  }


  void CarrotPlanner::initialize(std::string name, costmap_2d::Costmap2D* costmap, std::string global_frame)
  {
  

    visualize_potential_ = true;

    if(!initialized_)
    {
    
      //cout<<"==============1=============="<<endl;
      costmap_ = costmap;

      //costmap_ros_ = costmap_ros;
      //costmap_ = costmap_ros_->getCostmap();

      global_frame_ = global_frame;
      
      //cout<<"==============2=============="<<endl;
      planner_ = boost::shared_ptr<NavFn>(new NavFn(costmap_->getSizeInCellsX(), costmap_->getSizeInCellsY()));

      ros::NodeHandle private_nh("~/" + name);

      //cout<<"==============3=============="<<endl;
      plan_pub_ = private_nh.advertise<nav_msgs::Path>("plan", 1);
      //if we're going to visualize the potential array we need to advertise
      if(visualize_potential_)
        potarr_pub_ = private_nh.advertise<sensor_msgs::PointCloud2>("potential", 1);


      //cout<<"==============4=============="<<endl;
      initialized_ = true;

    }
    else
    {
      ROS_WARN("This planner has already been initialized, you can't call it twice, doing nothing");
    }

  }
  




  bool CarrotPlanner::makePlan(const geometry_msgs::PoseStamped& start, const geometry_msgs::PoseStamped& goal, std::vector<geometry_msgs::PoseStamped>& plan){

    if(!initialized_){
      ROS_ERROR("The planner has not been initialized, please call initialize() to use the planner");
      return false;
    }

    
    costmap_ = costmap_ros_->getCostmap();
    std::string global_frame_ = costmap_ros_->getGlobalFrameID();
    bool allow_unknown_ = true;
    double tolerance = 0.5;

    ROS_DEBUG("Got a start: %.2f, %.2f, and a goal: %.2f, %.2f", start.pose.position.x, start.pose.position.y, goal.pose.position.x, goal.pose.position.y);

    plan.clear();


    //until tf can handle transforming things that are way in the past... we'll require the goal to be in our global frame
    if(goal.header.frame_id != global_frame_){
      ROS_ERROR("The goal pose passed to this planner must be in the %s frame.  It is instead in the %s frame.", global_frame_.c_str(), goal.header.frame_id.c_str());
      return false;
    }

    if(start.header.frame_id != global_frame_){
      ROS_ERROR("The start pose passed to this planner must be in the %s frame.  It is instead in the %s frame.", global_frame_.c_str(), start.header.frame_id.c_str());
      return false;
    }

    double world_x = start.pose.position.x;
    double world_y = start.pose.position.y;
    
    unsigned int map_x, map_y;
    if(!costmap_->worldToMap(world_x, world_y, map_x, map_y))
    {
      ROS_WARN_THROTTLE(1.0, "The robot's start position is off the global costmap. Planning will always fail, are you sure the robot has been properly localized?");
      return false;
    }

    //clear the starting cell within the costmap because we know it can't be an obstacle    
    costmap_->setCost(map_x, map_y, costmap_2d::FREE_SPACE);
    
    //make sure to resize the underlying array that Navfn uses
    planner_->setNavArr(costmap_->getSizeInCellsX(), costmap_->getSizeInCellsY());   // costmap_->getSizeInCellsX() = 800
    planner_->setCostmap(costmap_->getCharMap(), true, allow_unknown_);

    int map_start[2];
    map_start[0] = map_x;
    map_start[1] = map_y;

    world_x = goal.pose.position.x;
    world_y = goal.pose.position.y;

    if(!costmap_->worldToMap(world_x, world_y, map_x, map_y)){
      if(tolerance <= 0.0){
        ROS_WARN_THROTTLE(1.0, "The goal sent to the navfn planner is off the global costmap. Planning will always fail to this goal.");
        return false;
      }
      map_x = 0;
      map_y = 0;
    }

    int map_goal[2];
    map_goal[0] = map_x;
    map_goal[1] = map_y;

    //Why your goal looks “upside down”
    planner_->setStart(map_goal);
    planner_->setGoal(map_start);

    //bool success = planner_->calcNavFnAstar();
    planner_->calcNavFnDijkstra(true);

    double resolution = costmap_->getResolution();
    geometry_msgs::PoseStamped p, best_pose;
    p = goal;

    bool found_legal = false;
    double best_sdist = DBL_MAX;

    p.pose.position.y = goal.pose.position.y - tolerance;

    // find the best_sdist, and best_pose
    
    while(p.pose.position.y <= goal.pose.position.y + tolerance)
    {
      p.pose.position.x = goal.pose.position.x - tolerance;
      
      while(p.pose.position.x <= goal.pose.position.x + tolerance)
      {
        double potential = getPointPotential(p.pose.position);
        double sdist = sq_distance(p, goal);
        
        if(potential < POT_HIGH && sdist < best_sdist)
        {
          best_sdist = sdist;
          best_pose = p;
          found_legal = true;
        }
        
        p.pose.position.x += resolution;
        
      }
      
      p.pose.position.y += resolution;
      
    }

    if(found_legal)
    {
      //extract the plan
      if(getPlanFromPotential(best_pose, plan))
      {
        //make sure the goal we push on has the same timestamp as the rest of the plan
        geometry_msgs::PoseStamped goal_copy = best_pose;
        goal_copy.header.stamp = ros::Time::now();
        plan.push_back(goal_copy);
      }
      else
      {
        ROS_ERROR("Failed to get a plan from potential when a legal potential was found. This shouldn't happen.");
      }
    }
    
    
    if (visualize_potential_)
    {
      // Publish the potentials as a PointCloud2
      sensor_msgs::PointCloud2 cloud;
      cloud.width = 0;
      cloud.height = 0;
      cloud.header.stamp = ros::Time::now();
      cloud.header.frame_id = global_frame_;
      sensor_msgs::PointCloud2Modifier cloud_mod(cloud);
      cloud_mod.setPointCloud2Fields(4, "x", 1, sensor_msgs::PointField::FLOAT32,
                                        "y", 1, sensor_msgs::PointField::FLOAT32,
                                        "z", 1, sensor_msgs::PointField::FLOAT32,
                                        "pot", 1, sensor_msgs::PointField::FLOAT32);
      cloud_mod.resize(planner_->ny * planner_->nx);
      sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");

      PotarrPoint pt;
      float *pp = planner_->potarr;
      double pot_x, pot_y;
      
      //std::cout<<costmap_->getOriginX()<<costmap_->getOriginY()<<costmap_->getResolution()<<std::endl;
      
      for (unsigned int i = 0; i < (unsigned int)planner_->ny*planner_->nx ; i++)
      {
        if (pp[i] < 10e7)
        {
          //pot_x = costmap_->getOriginX() + (i%planner_->nx) * costmap_->getResolution();
          //pot_y = costmap_->getOriginY() + (i/planner_->nx) * costmap_->getResolution();

          pot_x = -20 + (i%planner_->nx) * 0.05;
          pot_y = -20 + (i/planner_->nx) * 0.05;
          
          //std::cout<< i<<","<< pp[i] <<std::endl;
          iter_x[0] = pot_x;
          iter_x[1] = pot_y;
          iter_x[2] = pp[i] / pp[ planner_->start[1]*planner_->nx + planner_->start[0] ] * 20;
          iter_x[3] = pp[i];
          ++iter_x;
        }
      }
      potarr_pub_.publish(cloud);
    }
    
    //publish the plan for visualization purposes
    publishPlan(plan, 1.0, 0.0, 0.0, 0.0);
    
    return !plan.empty();
  }


  void CarrotPlanner::publishPlan(const std::vector<geometry_msgs::PoseStamped>& path, double r, double g, double b, double a){
    if(!initialized_){
      ROS_ERROR("This planner has not been initialized yet, but it is being used, please call initialize() before use");
      return;
    }

    //create a message for the plan 
    nav_msgs::Path gui_path;
    gui_path.poses.resize(path.size());
    
    if(path.empty()) {
      //still set a valid frame so visualization won't hit transform issues
    	gui_path.header.frame_id = global_frame_;
      gui_path.header.stamp = ros::Time::now();
    } else { 
      gui_path.header.frame_id = path[0].header.frame_id;
      gui_path.header.stamp = path[0].header.stamp;
    }

    // Extract the plan in world co-ordinates, we assume the path is all in the same frame
    for(unsigned int i=0; i < path.size(); i++){
      gui_path.poses[i] = path[i];
    }

    plan_pub_.publish(gui_path);
  }


  double CarrotPlanner::getPointPotential(const geometry_msgs::Point& world_point)
  {
    if(!initialized_){
      ROS_ERROR("This planner has not been initialized yet, but it is being used, please call initialize() before use");
      return -1.0;
    }

    unsigned int map_x, map_y;
    if(!costmap_->worldToMap(world_point.x, world_point.y, map_x, map_y))
      return DBL_MAX;

    unsigned int index = map_y * planner_->nx + map_x;
    return planner_->potarr[index];
  }



  bool CarrotPlanner::getPlanFromPotential(const geometry_msgs::PoseStamped& goal, std::vector<geometry_msgs::PoseStamped>& plan)
  {
    if(!initialized_){
      ROS_ERROR("This planner has not been initialized yet, but it is being used, please call initialize() before use");
      return false;
    }

    //clear the plan, just in case
    plan.clear();

    std::string global_frame_ = costmap_ros_->getGlobalFrameID();

    //until tf can handle transforming things that are way in the past... we'll require the goal to be in our global frame
    if(goal.header.frame_id != global_frame_){
      ROS_ERROR("The goal pose passed to this planner must be in the %s frame.  It is instead in the %s frame.", global_frame_.c_str(), goal.header.frame_id.c_str());
      return false;
    }

    double world_x = goal.pose.position.x;
    double world_y = goal.pose.position.y;

    //the potential has already been computed, so we won't update our copy of the costmap
    unsigned int map_x, map_y;
    if( !costmap_->worldToMap(world_x, world_y, map_x, map_y) )
    {
      ROS_WARN_THROTTLE(1.0, "The goal sent to the navfn planner is off the global costmap. Planning will always fail to this goal.");
      return false;
    }

    int map_goal[2];
    map_goal[0] = map_x;
    map_goal[1] = map_y;
    planner_->setStart(map_goal);

    planner_->calcPath(costmap_->getSizeInCellsX() * 4);

    //extract the plan
    float *x = planner_->getPathX();
    float *y = planner_->getPathY();
    int len = planner_->getPathLen();
    ros::Time plan_time = ros::Time::now();

    //std::cout<<"len="<<len<<std::endl;

    for(int i = len - 1; i >= 0; --i){
      //convert the plan to world coordinates
      double world_x, world_y;
      
      //world_x = costmap_->getOriginX() + x[i] * costmap_->getResolution();
      //world_y = costmap_->getOriginY() + y[i] * costmap_->getResolution();
      
      world_x = -20 + x[i] * 0.05;
      world_y = -20 + y[i] * 0.05;     
      
      geometry_msgs::PoseStamped pose;
      pose.header.stamp = plan_time;
      pose.header.frame_id = global_frame_;
      pose.pose.position.x = world_x;
      pose.pose.position.y = world_y;
      pose.pose.position.z = 0.0;
      pose.pose.orientation.x = 0.0;
      pose.pose.orientation.y = 0.0;
      pose.pose.orientation.z = 0.0;
      pose.pose.orientation.w = 1.0;
      plan.push_back(pose);
    }

    //publish the plan for visualization purposes
    publishPlan(plan, 0.0, 1.0, 0.0, 0.0);
    
    return !plan.empty();
  }

};
