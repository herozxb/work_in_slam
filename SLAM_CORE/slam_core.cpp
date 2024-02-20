#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/correspondence_estimation.h>

// Function to perform least squares point cloud registration
void leastSquaresRegistration( const pcl::PointCloud<pcl::PointXYZ>::Ptr& destination,
                               const pcl::PointCloud<pcl::PointXYZ>::Ptr& moving_point_cloud,
                               Eigen::Matrix4f& transformation) {

    
    Eigen::Matrix3f R_result = Eigen::Matrix3f::Identity();  // Initial guess for rotation matrix
    Eigen::Vector3f t_result(0, 0, 0);  // Initial guess for translation vector


    int max_iterations = 1000;
    const float convergence_threshold = 1e-7;

    ////////////////////////////////////////normal/////////////////////////////////////////////
    // Create the normal estimation object
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(destination);

    // Create a search tree
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    ne.setSearchMethod(tree);

    // Estimate the normals
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ne.setRadiusSearch(0.5); // Adjust the search radius as needed
    ne.compute(*normals);


    // Output the normals
    for (size_t i = 0; i < normals->size(); ++i) {
        std::cout << "Normal " << i << ": "
                  << normals->points[i].normal_x << " "
                  << normals->points[i].normal_y << " "
                  << normals->points[i].normal_z << std::endl;
    }
    
    // Optimization loop
    for (int iter = 0; iter < max_iterations; ++iter) {

    	pcl::transformPointCloud(*moving_point_cloud, *moving_point_cloud, transformation);


	// Create a CorrespondenceEstimation object
	pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ> est;    
	est.setInputSource(moving_point_cloud);
	est.setInputTarget(destination);

	// Compute correspondences
	pcl::Correspondences correspondences;
	est.determineCorrespondences(correspondences);

	// Print the number of correspondences found
	//std::cout << "Number of correspondences: " << correspondences.size() << std::endl;

	// Print the indices of source and target points for each correspondence
	for (size_t i = 0; i < correspondences.size(); ++i) {
	//std::cout << "Correspondence " << i << ": ";
	//std::cout << "Source index: " << correspondences[i].index_query;
	//std::cout << ", Target index: " << correspondences[i].index_match << std::endl;
	}  
    
    
    
        // Construct the equation system using the error metric and current transformation
        Eigen::MatrixXf A(destination->size(), 6);
        Eigen::VectorXf B(destination->size());
        
        
        //------------------core_of_slam[start]---------------//
        for (size_t i = 0; i < destination->size(); ++i) {
            const pcl::PointXYZ& src_point = moving_point_cloud->points[correspondences[i].index_query];
            const pcl::PointXYZ& destination_point = destination->points[correspondences[i].index_match];
            A(i, 0) = normals->points[i].normal_z * src_point.y - normals->points[i].normal_y * src_point.z;
            A(i, 1) = normals->points[i].normal_x * src_point.z - normals->points[i].normal_z * src_point.x;
            A(i, 2) = normals->points[i].normal_y * src_point.x - normals->points[i].normal_x * src_point.y;
            A(i, 3) = normals->points[i].normal_x;
            A(i, 4) = normals->points[i].normal_y;
            A(i, 5) = normals->points[i].normal_z;
            B(i) = normals->points[i].normal_x * destination_point.x + normals->points[i].normal_y * destination_point.y + normals->points[i].normal_z * destination_point.z - normals->points[i].normal_x * src_point.x - normals->points[i].normal_y * src_point.y - normals->points[i].normal_z * src_point.z   ;
 
        }
        
        // Solve the least squares problem to obtain optimal transformation parameters
        Eigen::MatrixXf AtA = A.transpose() * A;
        Eigen::VectorXf AtB = A.transpose() * B;
        Eigen::VectorXf x = AtA.colPivHouseholderQr().solve(AtB);
     
        // Update rotation matrix and translation vector
        Eigen::Vector3f delta_t(x(3), x(4), x(5));
      
        Eigen::Vector3f delta_r(x(0), x(1), x(2));
        Eigen::AngleAxisf rotation_change(delta_r.norm(), delta_r.normalized());
        
        //------------------core_of_slam[end]---------------//


        //delta_R(0, 0) = 1;
        //delta_R(0, 1) = x(0) * x(1) - x(2);
        //delta_R(0, 2) = x(0) * x(2) + x(1);
        //delta_R(1, 0) = x(2);
        //delta_R(1, 1) = x(0) * x(1) * x(2) + 1;
        //delta_R(1, 2) = x(1) * x(2) - x(0);        
        //delta_R(2, 0) = -x(1);
        //delta_R(2, 1) = x(0);
        //delta_R(2, 2) = 1;        

        //delta_R(0, 0) = 1;
        //delta_R(0, 1) = x(2);
        //delta_R(0, 2) = x(1);
        //delta_R(1, 0) = x(2);
        //delta_R(1, 1) = 1;
        //delta_R(1, 2) = - x(0);        
        //delta_R(2, 0) = -x(1);
        //delta_R(2, 1) = x(0);
        //delta_R(2, 2) = 1;   
        
	// Update the final transformation matrix
	transformation.block<3, 3>(0, 0) = rotation_change.toRotationMatrix();
	transformation.block<3, 1>(0, 3) = delta_t;
	transformation.row(3) << 0, 0, 0, 1;
        
        t_result += delta_t;
        R_result = rotation_change.toRotationMatrix() * R_result;
        
        // Check for convergence
        if (delta_t.norm() < convergence_threshold && delta_r.norm() < convergence_threshold) {
            cout<<"iterations="<<iter<<endl;
            cout<<"convergence"<<endl;
            break;
        } 
        
    }
    
    // Update the final transformation matrix
    transformation.block<3, 3>(0, 0) = R_result;
    transformation.block<3, 1>(0, 3) = t_result;
    transformation.row(3) << 0, 0, 0, 1;
    
}



int main() {

    // Generate the first plane point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_of_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (float x = -1.0; x <= 1.0; x += 0.1) {
        for (float y = -1.0; y <= 1.0; y += 0.1) {
            pcl::PointXYZ point;
            point.x = x;
            point.y = y;
            point.z = 0.0;  // Z-coordinate is fixed for the plane
            source_of_point_cloud->push_back(point);
        }
    }

    for (float x = -1.0; x <= 1.0; x += 0.1) {
        for (float z = 0.0; z <= 1.0; z += 0.1) {
            pcl::PointXYZ point;
            point.x = x;
            point.y = 1.0;
            point.z = z;  // Z-coordinate is fixed for the plane
            source_of_point_cloud->push_back(point);
        }
    }
    
    source_of_point_cloud->width = source_of_point_cloud->size();
    source_of_point_cloud->height = 1;

    // Apply a transformation (translation + rotation) to the first point cloud
    Eigen::Affine3f transformation = Eigen::Affine3f::Identity();
    transformation.translation() << 0.0, 0.0, 10.0;  // Translate by (1.0, 0.0, 0.0)
    
    float theta_x = 0; //M_PI / 4; //M_PI / 3;  // Rotation angle around X-axis (45 degrees)
    float theta_y = 0; //M_PI / 100000;  // Rotation angle around Y-axis (30 degrees)
    float theta_z = M_PI / 6; //M_PI / 100000;  // Rotation angle around Z-axis (22.5 degrees)
    
    transformation.rotate(Eigen::AngleAxisf(theta_x, Eigen::Vector3f::UnitX()));  // Rotate around X-axis
    transformation.rotate(Eigen::AngleAxisf(theta_y, Eigen::Vector3f::UnitY()));  // Rotate around Y-axis
    transformation.rotate(Eigen::AngleAxisf(theta_z, Eigen::Vector3f::UnitZ()));  // Rotate around Z-axis

    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*source_of_point_cloud, *transformed_point_cloud, transformation);

    // Run ICP to align the transformed point cloud to the original
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(transformed_point_cloud);
    icp.setInputTarget(source_of_point_cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
    icp.align(*aligned);


    Eigen::Matrix4f transformation_result = Eigen::Matrix4f::Identity();

    //------------------------------------core-----------------------------------//
    leastSquaresRegistration(source_of_point_cloud,transformed_point_cloud,transformation_result);    
  
    cout<<"====================result========================"<<endl;
    cout<<transformation_result<<endl;

    // Print the transformation matrix
    std::cout << "Transformation matrix:" << std::endl << icp.getFinalTransformation() << std::endl;

    // Visualize the original and aligned point clouds
    pcl::visualization::PCLVisualizer viewer("ICP Demo");
    viewer.addPointCloud(source_of_point_cloud, "cloud1", 0);
    viewer.addPointCloud(transformed_point_cloud, "aligned", 0);
    viewer.spin();

    return 0;
}

