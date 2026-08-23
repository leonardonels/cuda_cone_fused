#pragma once

#include <cuda_cone_fused/ekf_odom.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>

/**
 * Outbound publisher / presenter: translates filter pose state ->
 * `nav_msgs/Odometry` + TF. Owned by (triggered by) FAST-LIMO, the MAIN motion
 * input. Stamps the output with the SOURCE measurement timestamp, never now()
 * (now() on a duplicate = phantom motion / bad TF extrapolation).
 *
 * ONE DELIBERATE EXCEPTION, and only for the TF: republishTf() re-stamps the
 * LAST CORRECTION with now(). See the comment on that method for why that is
 * sound here and would not be for the /Odometry topic, which is untouched by
 * it and still carries the source stamp on every sample.
 */
class OdometryPublisher {
public:
  OdometryPublisher(rclcpp::Node* node, const std::string& topic,
                    std::string frame_id, std::string child_frame_id,
                    bool tf_from_timer = false)
      : frame_id_(std::move(frame_id)),
        child_frame_id_(std::move(child_frame_id)),
        tf_from_timer_(tf_from_timer),
        logger_(node->get_logger()),
        clock_(node->get_clock()) {
    pub_ = node->create_publisher<nav_msgs::msg::Odometry>(topic, 1);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*node);
  }

  /* Override the child frame at runtime (used to inherit it from the FAST-LIMO
     input when left empty in config). Parent frame_id stays fixed: it is the
     cone-corrected `track` frame, NOT LIMO's frame. */
  void setChildFrameId(std::string child_frame_id) {
    child_frame_id_ = std::move(child_frame_id);
  }

  /* The frame FAST-LIMO calls its world, inherited from the input odometry's
     header. This is the frame we correct, i.e. the TF child. */
  void setOdomFrameId(std::string odom_frame_id) {
    odom_frame_id_ = std::move(odom_frame_id);
  }

  /**
   * @param odom_in the raw FAST-LIMO pose this correction is relative to.
   *
   * `odom_in` must be the same measurement the filter was just updated with,
   * which it is: publish() is called from fastLimoDataCallback with that very
   * message, so the two poses share a timestamp and no interpolation is needed.
   */
  void publish(EKFOdom& filter, const rclcpp::Time& stamp,
               const nav_msgs::msg::Odometry& odom_in) {
    const Vector3f pose = filter.getState().head(3);
    const Vector3f cov = filter.getPoseCovariance();

    /* Refuse to emit anything non-finite. A NaN reaches here either because the
       EKF state diverged or because the FAST-LIMO pose we are correcting against
       is already NaN -- and every field below is derived from those two, so one
       NaN poisons the transform AND the topic.

       tf2 *rejects* a transform containing NaN ("Ignoring transform ... contains
       nans") and reports nothing back to the broadcaster, so the whole subtree
       under `odom` silently detaches from `track` -- the same end state as the
       malformed empty-child transform below, reached by a different route and
       just as invisible in RViz and Foxglove.

       Dropping the topic too, rather than only the TF: a NaN pose is not a pose,
       and letting it through hands garbage to planning and lap_counter. A gap
       plus a warning is strictly more debuggable than propagated NaN. */
    if (!isFinite(pose) || !isFinite(odom_in.pose.pose)) {
      RCLCPP_WARN_THROTTLE(
          logger_, *clock_, 1000,
          "non-finite pose: EKF [%f %f %f], input odom [%f %f %f] -- skipping "
          "odometry and TF. Filter has diverged or %s is publishing NaN.",
          static_cast<double>(pose(0)), static_cast<double>(pose(1)),
          static_cast<double>(pose(2)), odom_in.pose.pose.position.x,
          odom_in.pose.pose.position.y, odom_in.pose.pose.position.z,
          odom_in.header.frame_id.c_str());
      return;
    }

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, pose(2));
    geometry_msgs::msg::Quaternion qmsg;
    qmsg.x = q.x(); qmsg.y = q.y(); qmsg.z = q.z(); qmsg.w = q.w();

    /* TF broadcast: `track -> map`, the CORRECTION -- deliberately not the
       vehicle pose, even though that is what we publish on the topic below.

       A frame may have exactly one parent, and FAST-LIMO already owns
       `map -> base_link`. Broadcasting `track -> base_link` here would give
       base_link two parents and the tree would flip between roots at the
       publish rate. The ROS answer is the map -> odom -> base_link pattern:
       the corrector publishes the edge ABOVE the drifting estimate, so the
       correction flows down the chain and base_link keeps its single owner.
       Ours is that pattern with different names -- `track` plays map,
       FAST-LIMO's `map` plays odom.

       In SE(2), with c = corrected (EKF) and l = raw (LIMO), both relative to
       base_link, we need T_track_map such that T_track_base = T_track_map * T_map_base:

           yaw = yaw_c - yaw_l
           x   = x_c - ( x_l*cos(yaw) - y_l*sin(yaw) )
           y   = y_c - ( x_l*sin(yaw) + y_l*cos(yaw) )

       With no correction applied the two poses coincide and this collapses to
       identity, which is the sanity check worth remembering. */
    const double yaw_l = yawOf(odom_in.pose.pose.orientation);
    const double yaw_corr = static_cast<double>(pose(2)) - yaw_l;
    const double cy = std::cos(yaw_corr), sy = std::sin(yaw_corr);
    const double xl = odom_in.pose.pose.position.x;
    const double yl = odom_in.pose.pose.position.y;

    tf2::Quaternion q_corr;
    q_corr.setRPY(0.0, 0.0, yaw_corr);

    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = stamp;
    t.header.frame_id = frame_id_;
    t.child_frame_id = odom_frame_id_;
    t.transform.translation.x = static_cast<double>(pose(0)) - (xl * cy - yl * sy);
    t.transform.translation.y = static_cast<double>(pose(1)) - (xl * sy + yl * cy);
    t.transform.translation.z = 0.0;
    t.transform.rotation.x = q_corr.x();
    t.transform.rotation.y = q_corr.y();
    t.transform.rotation.z = q_corr.z();
    t.transform.rotation.w = q_corr.w();

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

    /* Never broadcast into an empty child: that is the malformed transform this
       change exists to remove. Until the input odometry names its frame there
       is nothing meaningful to correct, so publish the topic and skip the TF. */
    if (!odom_frame_id_.empty()) {
      {
        std::lock_guard<std::mutex> lock(tf_mutex_);
        last_tf_ = t;
        have_tf_ = true;
      }
      /* When a timer owns the broadcast, do NOT also send from here: the edge
         would be published twice per correction at two different rates, which
         is harmless to consumers but makes `tf_monitor` unreadable and hides
         exactly the stall this split exists to remove. */
      if (!tf_from_timer_) {
        tf_broadcaster_->sendTransform(t);
      }
    }
    pub_->publish(odom);
  }

  /**
   * Re-broadcast the last computed correction with a fresh stamp.
   *
   * That rule is about POSE transforms. Re-stamping a duplicate pose asserts
   * the vehicle was at that position at the new time, so an interpolating
   * consumer reads phantom motion. This edge is not a pose -- it is the
   * CORRECTION, `P_ekf (-) P_limo`, and between two cone updates the EKF is
   * predicted from the very LIMO increments it is being differenced against,
   * so the two move together and the correction is genuinely piecewise
   * constant. Re-stamping it asserts "this correction is still current", which
   * is true. It is the same reason amcl and robot_localization republish
   * `map -> odom` on a timer rather than on measurement arrival.
   *
   * WHAT IT FIXES. `track -> odom` used to be published only at the tail of
   * fastLimoDataCallback, on the same single-threaded executor as
   * conesCallback -- so for the whole duration of an EKF update (the `exe_ms`
   * on the scan line) this edge went unpublished. A lookup of
   * `track -> base_link` resolves at the latest time COMMON to the chain, and
   * fast_LIMO's `odom -> base_link` streams from another process throughout,
   * so that common time was gated entirely by this edge and stalled in step
   * with the association loop. Only consumers of the FULL chain saw it, which
   * in practice means a viewer's follow camera and nothing else.
   *
   * Only ever publishes a correction that was computed as a MATCHED PAIR in
   * publish() -- it never pairs a fresh LIMO pose with a stale EKF pose, which
   * would reintroduce a speed-proportional error instead of removing a stall.
   */
  void republishTf(const rclcpp::Time& stamp) {
    geometry_msgs::msg::TransformStamped t;
    {
      std::lock_guard<std::mutex> lock(tf_mutex_);
      if (!have_tf_) {return;}  /* nothing computed yet: stay silent, not wrong */
      t = last_tf_;
    }
    t.header.stamp = stamp;
    tf_broadcaster_->sendTransform(t);
  }

private:
  static double yawOf(const geometry_msgs::msg::Quaternion& q) {
    return std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z));
  }

  static bool isFinite(const Vector3f& v) {
    return std::isfinite(v(0)) && std::isfinite(v(1)) && std::isfinite(v(2));
  }

  static bool isFinite(const geometry_msgs::msg::Pose& p) {
    return std::isfinite(p.position.x) && std::isfinite(p.position.y) &&
           std::isfinite(p.position.z) && std::isfinite(p.orientation.x) &&
           std::isfinite(p.orientation.y) && std::isfinite(p.orientation.z) &&
           std::isfinite(p.orientation.w);
  }

  std::string odom_frame_id_;
  std::string frame_id_;
  std::string child_frame_id_;
  bool tf_from_timer_;
  rclcpp::Logger logger_;
  rclcpp::Clock::SharedPtr clock_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  /* Last correction computed by publish(), and the lock that lets the timer
     thread read it. This mutex is the ONLY state shared across the two
     callback groups -- the filter itself is never touched from the timer. */
  std::mutex tf_mutex_;
  geometry_msgs::msg::TransformStamped last_tf_;
  bool have_tf_ = false;
};
