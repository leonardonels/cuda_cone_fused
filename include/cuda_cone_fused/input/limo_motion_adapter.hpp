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
 *
 * It also owns the FAST-LIMO IMU-CALIBRATION GATE (see convert()): the frames
 * LIMO publishes while it is still calibrating are a frozen placeholder, not
 * odometry, and are dropped here rather than fed to the filter.
 */
class LimoMotionAdapter {
public:
  /**
   * Enable/disable the FAST-LIMO IMU-calibration gate (enabled by default).
   * Disabling it declares the source calibrated from the first frame, i.e. the
   * legacy behaviour of consuming every frame LIMO publishes.
   */
  void setGateOnCalibration(bool enable) {
    gate_on_calibration_ = enable;
    if (!enable) calibrated_ = true;
  }

  /**
   * True once FAST-LIMO has finished its IMU calibration and is publishing a
   * real state (always true when the gate is disabled). The node uses this to
   * withhold the cone map for the duration of the calibration.
   */
  bool isCalibrated() const { return calibrated_; }

  /**
   * @return the increment for this frame, or std::nullopt while FAST-LIMO is
   *         still calibrating and on the first calibrated frame (which only
   *         seeds the reference — there is no motion yet).
   */
  std::optional<MotionIncrement> convert(const nav_msgs::msg::Odometry& msg) {
    /* ---- FAST-LIMO IMU-calibration gate ---------------------------------
       While FAST-LIMO calibrates the IMU (its `calibration.time`, 3 s by
       default) Localizer::getWorldState() returns a DEFAULT-CONSTRUCTED State
       and getPoseCovariance() returns 36 zeros, so what reaches this topic is a
       frozen placeholder: position exactly (0,0,0), orientation exactly
       identity, covariance exactly 0. That is not odometry, and feeding it to
       the filter breaks the start-up in two distinct ways:

         - the frozen pose makes every increment 0 while the car may already be
           rolling, so the EKF pose stays pinned at the origin while the cone
           observations sweep past it (input cones drifting one way while the
           odometry has not started moving yet);

         - at the instant calibration ends LIMO re-anchors its attitude to the
           estimated gravity vector AND sets its covariance to identity
           (init_iKFoM_state: `init_P.setIdentity()`). Read as a relative
           motion that single frame is a one-shot yaw step plus a full 1.0
           variance injection on x/y/yaw; the inflated pose covariance blows the
           Kalman gain up on the very next cone scan and the pose SNAPS.

       So drop these frames entirely and do NOT seed the reference from them:
       the first CALIBRATED frame becomes the reference, which puts the
       calibration-end discontinuity outside every delta we ever compute.

       Detection is by signature, not by a hardcoded 3 s timer, so it follows
       whatever `calibration.time` LIMO is configured with (including a runtime
       change or a LIMO restart). The signature is unambiguous: a calibrated
       LIMO reports a covariance that starts at 1.0 and never returns to
       exactly zero. The latch is one-way, mirroring LIMO's own
       `imu_calibrated_` flag. */
    if (!calibrated_) {
      if (isUncalibratedFrame(msg)) return std::nullopt;
      calibrated_ = true;
    }

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

  /* Signature of a frame published while FAST-LIMO is still calibrating: the
     default-constructed State (origin + identity attitude) carried alongside an
     all-zero covariance. The exact-zero comparisons are deliberate — these are
     literal defaults that were never computed, not values that merely round to
     zero, and LIMO's covariance is >= 1.0 from the first calibrated frame on. */
  static bool isUncalibratedFrame(const nav_msgs::msg::Odometry& msg) {
    const auto& p = msg.pose.pose.position;
    const auto& o = msg.pose.pose.orientation;
    return p.x == 0.0 && p.y == 0.0 && p.z == 0.0 &&
           o.x == 0.0 && o.y == 0.0 && o.z == 0.0 && o.w == 1.0 &&
           msg.pose.covariance.at(0)  == 0.0 &&
           msg.pose.covariance.at(7)  == 0.0 &&
           msg.pose.covariance.at(35) == 0.0;
  }

  Eigen::Vector3f prev_pose_ = Eigen::Vector3f::Zero();
  Eigen::Vector3f prev_cov_  = Eigen::Vector3f::Zero();
  bool initialized_ = false;

  /* One-way latch: set on the first frame that is not the calibration
     placeholder (or up front when the gate is disabled). */
  bool calibrated_ = false;
  bool gate_on_calibration_ = true;
};
