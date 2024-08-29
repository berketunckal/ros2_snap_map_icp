#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <std_srvs/srv/set_bool.hpp> // Eklenen kısım
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include <tf2/exceptions.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include <laser_geometry/laser_geometry.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <memory>
#include <mutex>
#include <chrono>

using namespace std::chrono_literals;

// Define your parameters here
const double ICP_FITNESS_THRESHOLD = 100.1;
const double DIST_THRESHOLD = 0.05;
const double ANGLE_THRESHOLD = 0.01;
const double ANGLE_UPPER_THRESHOLD = M_PI / 6; // M_PI / 6
const double AGE_THRESHOLD = 1;
const double UPDATE_AGE_THRESHOLD = 1;
const double ICP_INLIER_THRESHOLD = 0.25;
const double FIRST_POSE_ICP_INLIER_THRESHOLD = 0.1;
const double ICP_INLIER_DIST = 0.1;
const double POSE_COVARIANCE_TRANS = 1.5;
const int ICP_NUM_ITER = 250;
const double SCAN_RATE = 2;
const std::string BASE_LASER_FRAME = "laser";
const std::string ODOM_FRAME = "odom";

class ICPLocalizationNode : public rclcpp::Node
{
public:
    ICPLocalizationNode() : Node("icp_localization_node"),
                            scan_rate_(SCAN_RATE),
                            initial_position_set_(false)
    {
        RCLCPP_INFO(this->get_logger(), "ICPLocalizationNode started.");

        // Initialize QoS profiles
        rclcpp::QoS scan_qos_profile = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile();
        rclcpp::QoS map_qos_profile = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
        rclcpp::Clock::SharedPtr clock = this->get_clock();

        // Initialize publishers
        pub_output_scan_transformed = this->create_publisher<sensor_msgs::msg::PointCloud2>("output_scan_transformed_topic", 10);
        pub_pose = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("initialpose", 10);

        // Initialize service
        srv_is_initial_pose_true = this->create_service<std_srvs::srv::SetBool>(
            "is_initial_pose_true", std::bind(&ICPLocalizationNode::isInitialPoseTrue, this, std::placeholders::_1, std::placeholders::_2));

        // Initialize variables
        we_have_a_map = false;
        we_have_a_scan = false;
        we_have_a_scan_transformed = false;
        use_sim_time = false;
        lastScan = 0;
        actScan = 0;

        // Initialize tf buffer and listener
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
        listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Initialize projector
        projector_ = std::make_shared<laser_geometry::LaserProjection>();

        // Subscribe to occupancy grid map with QoS profile
        map_subscriber_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "map", map_qos_profile, std::bind(&ICPLocalizationNode::mapCallback, this, std::placeholders::_1));

        // Subscribe to laser scan with QoS profile
        laser_subscriber_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "scan", scan_qos_profile, std::bind(&ICPLocalizationNode::scanCallback, this, std::placeholders::_1));

        // Initialize the timer for ICP processing
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000 / SCAN_RATE)),
            std::bind(&ICPLocalizationNode::processScan, this));
    }

private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_output_scan_transformed;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_pose;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr srv_is_initial_pose_true;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> listener_;
    std::shared_ptr<laser_geometry::LaserProjection> projector_;
    sensor_msgs::msg::PointCloud2 cloud2;
    sensor_msgs::msg::PointCloud2 cloud2transformed;

    boost::shared_ptr<sensor_msgs::msg::PointCloud2> output_cloud = boost::shared_ptr<sensor_msgs::msg::PointCloud2>(new sensor_msgs::msg::PointCloud2());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz;
    pcl::KdTree<pcl::PointXYZ>::Ptr mapTree;
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

    bool we_have_a_map;
    bool we_have_a_scan;
    bool we_have_a_scan_transformed;
    bool use_sim_time;
    bool initial_position_set_;

    std::mutex scan_callback_mutex;
    int lastTimeSent = -1000;
    int count_sc_ = 0;
    int lastScan;
    int actScan;
    double scan_rate_;
    typedef pcl::PointCloud<pcl::PointXYZ> PointCloudT;

    rclcpp::TimerBase::SharedPtr timer_;
    sensor_msgs::msg::LaserScan::SharedPtr latest_scan_;

    pcl::KdTree<pcl::PointXYZ>::Ptr getTree(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudb)
    {
        pcl::KdTree<pcl::PointXYZ>::Ptr tree;
        tree.reset(new pcl::KdTreeFLANN<pcl::PointXYZ>);

        tree->setInputCloud(cloudb);
        return tree;
    }

    inline void matrixAsTransform(const Eigen::Matrix4f &out_mat, tf2::Transform &bt)
    {
        double mv[12];

        mv[0] = out_mat(0, 0);
        mv[4] = out_mat(0, 1);
        mv[8] = out_mat(0, 2);
        mv[1] = out_mat(1, 0);
        mv[5] = out_mat(1, 1);
        mv[9] = out_mat(1, 2);
        mv[2] = out_mat(2, 0);
        mv[6] = out_mat(2, 1);
        mv[10] = out_mat(2, 2);

        tf2::Matrix3x3 basis;
        basis.setFromOpenGLSubMatrix(mv);
        tf2::Vector3 origin(out_mat(0, 3), out_mat(1, 3), out_mat(2, 3));

        bt = tf2::Transform(basis, origin);
    }

    void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
    {
        RCLCPP_INFO(get_logger(), "Received map with frame_id: [%s]", msg->header.frame_id.c_str());

        float resolution = msg->info.resolution;
        float width = msg->info.width;
        float height = msg->info.height;

        float posx = msg->info.origin.position.x;
        float posy = msg->info.origin.position.y;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

        cloud_xyz->height = 1;
        cloud_xyz->is_dense = false;
        std_msgs::msg::Header header;
        header.stamp = msg->header.stamp; // Use message timestamp
        header.frame_id = "map";          // Use desired frame id
        cloud_xyz->header = pcl_conversions::toPCL(header);

        pcl::PointXYZ point_xyz;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (msg->data[x + y * width] == 100)
                {
                    point_xyz.x = (0.5f + x) * resolution + posx;
                    point_xyz.y = (0.5f + y) * resolution + posy;
                    point_xyz.z = 0;
                    cloud_xyz->points.push_back(point_xyz);
                }
            }
        }
        cloud_xyz->width = cloud_xyz->points.size();

        mapTree = getTree(cloud_xyz);

        pcl::toROSMsg(*cloud_xyz, *output_cloud);
        RCLCPP_INFO(get_logger(), "Publishing PointXYZ cloud with %ld",
                    cloud_xyz->points.size());

        we_have_a_map = true;
        RCLCPP_INFO(get_logger(), "Map is ready.");
    }

    void scanCallback(sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        if (!we_have_a_map)
        {
            RCLCPP_INFO(get_logger(), "SnapMapICP waiting for map to be published");
            return;
        }

        latest_scan_ = msg;
    }

    void processScan()
    {
        if (!latest_scan_)
        {
            RCLCPP_INFO(get_logger(), "Waiting for a scan to process");
            return;
        }

        rclcpp::Time scan_in_time = latest_scan_->header.stamp;
        rclcpp::Time now = this->now(); // Doğru şekilde `now` fonksiyonunu çağırın

        rclcpp::Duration scan_age = now - scan_in_time;

        if (!scan_callback_mutex.try_lock())
            return;

        if (scan_age.nanoseconds() / 1e9 > AGE_THRESHOLD)
        {
            RCLCPP_INFO(this->get_logger(), "SCAN SEEMS TOO OLD (%f seconds, %f threshold) scan time: %f , now %f",
                        scan_age.nanoseconds() / 1e9,
                        AGE_THRESHOLD,
                        scan_in_time.nanoseconds() / 1e9,
                        now.nanoseconds() / 1e9);
            scan_callback_mutex.unlock();
            return;
        }

        // Transformu al
        tf2::Stamped<tf2::Transform> base_at_laser_stamped;
        if (!getTransform(base_at_laser_stamped, ODOM_FRAME, "base_link", rclcpp::Time(0)))
        {
            RCLCPP_WARN(this->get_logger(), "Did not get base pose at laser scan time");
            scan_callback_mutex.unlock();
            return;
        }

        sensor_msgs::msg::PointCloud2 cloud;
        sensor_msgs::msg::PointCloud2 cloudInMap;

        projector_->projectLaser(*latest_scan_, cloud);

        bool gotTransform = false;
        tf2::Transform oldPose;

        while (!gotTransform && rclcpp::ok())
        {
            try
            {
                gotTransform = true;
                geometry_msgs::msg::TransformStamped transformStamped = tf_buffer_->lookupTransform("map", "base_link", rclcpp::Time(0));

                // geometry_msgs::msg::TransformStamped to tf2::Transform
                tf2::fromMsg(transformStamped.transform, oldPose);
            }
            catch (const tf2::TransformException &ex)
            {
                gotTransform = false;
                RCLCPP_WARN(this->get_logger(), "DIDNT GET TRANSFORM IN B");
            }
        }

        geometry_msgs::msg::TransformStamped mapCloud;
        try
        {
            mapCloud = tf_buffer_->lookupTransform("map", cloud.header.frame_id, rclcpp::Time(0));
        }
        catch (tf2::TransformException &ex)
        {
            RCLCPP_WARN_SKIPFIRST_THROTTLE(
                get_logger(), *get_clock(), (5000ms).count(),
                "cannot get map to base_link transform. %s", ex.what());
            scan_callback_mutex.unlock();
            return;
        }

        geometry_msgs::msg::TransformStamped mapBaseLink;
        try
        {
            mapBaseLink = tf_buffer_->lookupTransform("map", "base_link", rclcpp::Time(0));
        }
        catch (tf2::TransformException &ex)
        {
            RCLCPP_WARN_SKIPFIRST_THROTTLE(
                get_logger(), *get_clock(), (5000ms).count(),
                "cannot get map to base_link transform. %s", ex.what());
            scan_callback_mutex.unlock();
            return;
        }

        try
        {
            tf2::doTransform(cloud, cloudInMap, mapCloud);
        }
        catch (tf2::TransformException &ex)
        {
            RCLCPP_WARN(this->get_logger(), "Failed to transform cloud: %s", ex.what());
            scan_callback_mutex.unlock();
            return;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr scan_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(cloudInMap, *scan_cloud);

        if (scan_cloud->empty())
        {
            RCLCPP_WARN(this->get_logger(), "Empty scan cloud received");
            scan_callback_mutex.unlock();
            return;
        }

        icp.setTransformationEpsilon(1e-6);
        icp.setMaxCorrespondenceDistance(0.5);
        icp.setMaximumIterations(ICP_NUM_ITER);

        pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*output_cloud, *map_cloud);

        icp.setInputSource(scan_cloud);
        icp.setInputTarget(map_cloud);

        pcl::PointCloud<pcl::PointXYZ> aligned_cloud;
        int i = 0;

        icp.align(aligned_cloud);

        if (icp.hasConverged())
        {
            actScan++;
            const Eigen::Matrix4f &transf = icp.getFinalTransformation();
            tf2::Transform t;
            matrixAsTransform(transf, t);

            pcl::PointCloud<pcl::PointXYZ> transformedCloud;
            pcl::transformPointCloud(*scan_cloud, transformedCloud, icp.getFinalTransformation());

            double inlier_perc = 0.0;
            {
                std::vector<int> nn_indices(1);
                std::vector<float> nn_sqr_dists(1);

                size_t numinliers = 0;
                for (size_t k = 0; k < transformedCloud.points.size(); ++k)
                {
                    if (mapTree->radiusSearch(transformedCloud.points[k], ICP_INLIER_DIST, nn_indices, nn_sqr_dists, 1) != 0)
                        numinliers += 1;
                }
                if (transformedCloud.points.size() > 0)
                {
                    inlier_perc = (double)numinliers / (double)transformedCloud.points.size();
                }
            }

            pcl::toROSMsg(transformedCloud, cloud2transformed);
            we_have_a_scan_transformed = true;

            double dist = std::sqrt((t.getOrigin().x() * t.getOrigin().x()) + (t.getOrigin().y() * t.getOrigin().y()));
            double angleDist = t.getRotation().getAngle();
            tf2::Vector3 rotAxis = t.getRotation().getAxis();
            t = t * oldPose;

            tf2::Stamped<tf2::Transform> base_after_icp;
            if (!getTransform(base_after_icp, ODOM_FRAME, "base_link", rclcpp::Time(0)))
            {
                RCLCPP_WARN(get_logger(), "Did not get base pose at now");
                scan_callback_mutex.unlock();
                return;
            }
            else
            {
                tf2::Transform rel = base_at_laser_stamped.inverseTimes(base_after_icp);
                RCLCPP_DEBUG(get_logger(), "relative motion of robot while doing icp: %fcm %fdeg", rel.getOrigin().length(), rel.getRotation().getAngle() * 180 / M_PI);
                t = t * rel;
            }

            double cov = POSE_COVARIANCE_TRANS;

            std::cout
                << "actScan: " << actScan << "\n"
                << "--------------------------- "
                   "\n"
                << "Last Time Sent: " << lastTimeSent << "\n";
            bool cond1 = (dist > DIST_THRESHOLD);
            bool cond2 = (angleDist > ANGLE_THRESHOLD);
            bool cond3 = (inlier_perc > ICP_INLIER_THRESHOLD);
            bool cond4 = (angleDist < ANGLE_UPPER_THRESHOLD);

            std::cout << "Condition 1 (dist > DIST_THRESHOLD): " << cond1 << " distance: " << dist << "\n";
            std::cout << "Condition 2 (angleDist > ANGLE_THRESHOLD): " << cond2 << " angleDist: " << angleDist << "\n";
            std::cout << "Condition 3 (inlier_perc > ICP_INLIER_THRESHOLD): " << cond3 << " inlier_perc: " << inlier_perc << "\n";
            std::cout << "Condition 4 (angleDist < ANGLE_UPPER_THRESHOLD): " << cond4 << "\n";
            std::cout << "Inıt Pose : " << initial_position_set_ << "\n";

            if (!initial_position_set_ &&
                (inlier_perc > FIRST_POSE_ICP_INLIER_THRESHOLD) &&
                (angleDist < ANGLE_UPPER_THRESHOLD))
            {
                lastTimeSent = actScan;
                geometry_msgs::msg::PoseWithCovarianceStamped pose;
                pose.header.frame_id = "map";
                pose.pose.pose.position.x = t.getOrigin().x();
                pose.pose.pose.position.y = t.getOrigin().y();

                tf2::Quaternion quat = t.getRotation();
                pose.pose.pose.orientation = tf2::toMsg(quat);

                float factorPos = 0.03;
                float factorRot = 0.1;
                pose.pose.covariance[6 * 0 + 0] = (cov * cov) * factorPos;
                pose.pose.covariance[6 * 1 + 1] = (cov * cov) * factorPos;
                pose.pose.covariance[6 * 3 + 3] = (M_PI / 12.0 * M_PI / 12.0) * factorRot;

                pub_pose->publish(pose);

                // Mark initial position as set
                RCLCPP_INFO(this->get_logger(), "Initial position set.");
            }
            if (!initial_position_set_ && inlier_perc > 0.50)

            {
                initial_position_set_ = true;
            }

            if (initial_position_set_)
            {
                if ((actScan - lastTimeSent > UPDATE_AGE_THRESHOLD) &&
                    ((dist > DIST_THRESHOLD) || (angleDist > ANGLE_THRESHOLD)) &&
                    (inlier_perc > ICP_INLIER_THRESHOLD) &&
                    (angleDist < ANGLE_UPPER_THRESHOLD))
                {
                    lastTimeSent = actScan;
                    geometry_msgs::msg::PoseWithCovarianceStamped pose;
                    pose.header.frame_id = "map";
                    pose.pose.pose.position.x = t.getOrigin().x();
                    pose.pose.pose.position.y = t.getOrigin().y();

                    tf2::Quaternion quat = t.getRotation();
                    pose.pose.pose.orientation = tf2::toMsg(quat);

                    float factorPos = 0.03;
                    float factorRot = 0.1;
                    pose.pose.covariance[6 * 0 + 0] = (cov * cov) * factorPos;
                    pose.pose.covariance[6 * 1 + 1] = (cov * cov) * factorPos;
                    pose.pose.covariance[6 * 3 + 3] = (M_PI / 12.0 * M_PI / 12.0) * factorRot;

                    pub_pose->publish(pose);
                }
            }

            cloud2transformed.header.frame_id = "map"; // Frame ID'yi burada ayarlayın

            pub_output_scan_transformed->publish(cloud2transformed);
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "ICP did not converge.");
        }

        scan_callback_mutex.unlock();
    }

    void isInitialPoseTrue(
        const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
        std::shared_ptr<std_srvs::srv::SetBool::Response> response)
    {
        response->success = initial_position_set_;
        response->message = initial_position_set_ ? "Initial position is set." : "Initial position is not set.";
    }

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_subscriber_;

    bool getTransform(tf2::Stamped<tf2::Transform> &trans, const std::string parent_frame, const std::string child_frame, const builtin_interfaces::msg::Time &stamp)
    {
        bool gotTransform = false;

        tf2::TimePoint time_point = tf2_ros::fromMsg(stamp);

        try
        {
            geometry_msgs::msg::TransformStamped transform_stamped = tf_buffer_->lookupTransform(parent_frame, child_frame, time_point);

            tf2::Stamped<tf2::Transform> tf_stamped;
            tf2::fromMsg(transform_stamped, tf_stamped);

            trans = tf_stamped;
            gotTransform = true;
        }
        catch (tf2::TransformException &ex)
        {
            RCLCPP_INFO(get_logger(), "Failed to get transform %s to %s: %s", parent_frame.c_str(), child_frame.c_str(), ex.what());
        }

        return gotTransform;
    }

    bool waitForTransform(
        const std::string &target_frame,
        const std::string &source_frame,
        const rclcpp::Time &stamp,
        const rclcpp::Duration &timeout)
    {
        auto start = this->now();
        while (this->now() - start < timeout)
        {
            try
            {
                tf_buffer_->lookupTransform(target_frame, source_frame, stamp);
                return true;
            }
            catch (tf2::TransformException &ex)
            {
                rclcpp::sleep_for(10ms);
            }
        }
        return false;
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ICPLocalizationNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
