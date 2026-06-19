#pragma once

#include <cuda_cone_fused/ekf_odom.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <algorithm>
#include <memory>
#include <string>

/**
 * Outbound publisher / presenter: translates filter pose state ->
 * `nav_msgs/Odometry` + TF. Owned by (triggered by) FAST-LIMO, the MAIN motion
 * input. Stamps the output with the SOURCE measurement timestamp, never now()
 * (now() on a duplicate = phantom motion / bad TF extrapolation).
 */
class OdometryPublisher {
public:
  OdometryPublisher(rclcpp::Node* node, const std::string& topic,
                    std::string frame_id, std::string child_frame_id)
      : frame_id_(std::move(frame_id)),
        child_frame_id_(std::move(child_frame_id)) {
    pub_ = node->create_publisher<nav_msgs::msg::Odometry>(topic, 1);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*node);
  }

  /* Override the child frame at runtime (used to inherit it from the FAST-LIMO
     input when left empty in config). Parent frame_id stays fixed: it is the
     cone-corrected `track` frame, NOT LIMO's frame. */
  void setChildFrameId(std::string child_frame_id) {
    child_frame_id_ = std::move(child_frame_id);
  }

  void publish(EKFOdom& filter, const rclcpp::Time& stamp) {
    const Vector3f pose = filter.getState().head(3);
    const Vector3f cov = filter.getPoseCovariance();

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, pose(2));
    geometry_msgs::msg::Quaternion qmsg;
    qmsg.x = q.x(); qmsg.y = q.y(); qmsg.z = q.z(); qmsg.w = q.w();

    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = stamp;
    t.header.frame_id = frame_id_;
    t.child_frame_id = child_frame_id_;
    t.transform.translation.x = pose(0);
    t.transform.translation.y = pose(1);
    t.transform.translation.z = 0.0;
    t.transform.rotation = qmsg;

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = frame_id_;
    odom.child_frame_id = child_frame_id_;
    odom.pose.pose.position.x = pose(0);
    odom.pose.pose.position.y = pose(1);
    odom.pose.pose.position.z = 0.0;
    odom.pose.pose.orientation = qmsg;

    /* Floor each variance at a tiny positive value: the float covariance update
       P -= K(HP) can push a diagonal entry slightly negative, which makes the
       published covariance non-PSD (RViz: "Negative eigenvalue ..."). Flooring
       keeps what we publish valid without touching the filter. */
    constexpr double COV_FLOOR = 1e-9;
    odom.pose.covariance.at(0)  = std::max(static_cast<double>(cov(0)), COV_FLOOR);  /* X   */
    odom.pose.covariance.at(7)  = std::max(static_cast<double>(cov(1)), COV_FLOOR);  /* Y   */
    odom.pose.covariance.at(35) = std::max(static_cast<double>(cov(2)), COV_FLOOR);  /* Yaw */

    tf_broadcaster_->sendTransform(t);
    pub_->publish(odom);
  }

private:
  std::string frame_id_;
  std::string child_frame_id_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};
