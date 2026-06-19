#pragma once

#include <cuda_cone_fused/ekf_types.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

#include <cmath>
#include <optional>

/**
 * Inbound adapter: FAST-LIMO `nav_msgs/Odometry` -> a
 * sensor-agnostic MotionIncrement consumed by EKFOdom::predict().
 *
 * This is where ALL the LIMO-specific bookkeeping lives now (moved out of the
 * filter): the previous-frame reference, the first-frame seed, the quaternion ->
 * yaw conversion, the 0/7/35 covariance indexing, and the relative-motion +
 * covariance-increment computation. The filter consumes a ready-made increment
 * and never sees a ROS message.
 *
 * LIMO is a RELATIVE re-anchor: we emit the motion of
 * LIMO since the previous frame (expressed in the previous LIMO body frame), not
 * an absolute pose. EKFOdom rotates that into the EKF frame by its own heading
 * and composes it onto the cone-corrected mean.
 */
class LimoMotionAdapter {
public:
  /**
   * @return the increment for this frame, or std::nullopt on the very first
   *         frame (which only seeds the reference — there is no motion yet).
   */
  std::optional<MotionIncrement> convert(const nav_msgs::msg::Odometry& msg) {
    tf2::Quaternion q;
    q.setX(msg.pose.pose.orientation.x);
    q.setY(msg.pose.pose.orientation.y);
    q.setZ(msg.pose.pose.orientation.z);
    q.setW(msg.pose.pose.orientation.w);
    double r, p, y;
    tf2::Matrix3x3(q).getRPY(r, p, y);

    const Eigen::Vector3f pose(static_cast<float>(msg.pose.pose.position.x),
                               static_cast<float>(msg.pose.pose.position.y),
                               static_cast<float>(y));
    Eigen::Vector3f cov;
    cov(0) = static_cast<float>(msg.pose.covariance.at(0));   /* X   */
    cov(1) = static_cast<float>(msg.pose.covariance.at(7));   /* Y   */
    cov(2) = static_cast<float>(msg.pose.covariance.at(35));  /* Yaw */

    /* First frame: do NOT seed the EKF pose from LIMO. The EKF frame is anchored
       at the track origin; LIMO is a pure relative motion source, so we only
       record this frame as the reference for the next delta. */
    if (!initialized_) {
      prev_pose_ = pose;
      prev_cov_ = cov;
      initialized_ = true;
      return std::nullopt;
    }

    /* Relative motion since the previous frame, in the previous LIMO body frame. */
    const float dx = pose(0) - prev_pose_(0);
    const float dy = pose(1) - prev_pose_(1);
    const float prev_theta = prev_pose_(2);

    const float local_dx =  std::cos(prev_theta) * dx + std::sin(prev_theta) * dy;
    const float local_dy = -std::sin(prev_theta) * dx + std::cos(prev_theta) * dy;
    const float dtheta   =  normalizeAngle(pose(2) - prev_theta);

    /* Inflate by the INCREMENT of LIMO's own covariance over this step, not its
       absolute value (mirrors using the relative motion for the mean). Clamped
       to >= 0 since LIMO's covariance can drop (e.g. its own relocalisation),
       which must not shrink the EKF covariance. */
    Eigen::Vector3f cov_inc;
    for (int i = 0; i < 3; ++i) {
      const float d = cov(i) - prev_cov_(i);
      cov_inc(i) = (d > 0.0f) ? d : 0.0f;
    }

    prev_pose_ = pose;
    prev_cov_ = cov;

    MotionIncrement inc;
    inc.delta_pose = Eigen::Vector3f(local_dx, local_dy, dtheta);
    inc.cov_increment = cov_inc;
    inc.stamp_ns = static_cast<uint64_t>(msg.header.stamp.sec) * 1000000000ull +
                   static_cast<uint64_t>(msg.header.stamp.nanosec);
    return inc;
  }

private:
  static float normalizeAngle(float angle) {
    while (angle >  M_PI) angle -= 2.0f * static_cast<float>(M_PI);
    while (angle < -M_PI) angle += 2.0f * static_cast<float>(M_PI);
    return angle;
  }

  Eigen::Vector3f prev_pose_ = Eigen::Vector3f::Zero();
  Eigen::Vector3f prev_cov_  = Eigen::Vector3f::Zero();
  bool initialized_ = false;
};
