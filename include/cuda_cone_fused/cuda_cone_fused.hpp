#pragma once

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <cuda_cone_fused/output/cone_map_publisher.hpp>
#include <cuda_cone_fused/input/cone_obs_adapter.hpp>
#include <cuda_cone_fused/ekf_odom.hpp>
#include <cuda_cone_fused/input/filter_command.hpp>
#include <cuda_cone_fused/input/limo_motion_adapter.hpp>
#include <cuda_cone_fused/output/odometry_publisher.hpp>

#include <mmr_base/msg/race_status.hpp>

#include <cstdint>
#include <memory>
#include <string>

/**
 * THE NODE = orchestrator (Mediator / Facade). It is the only place
 * that knows both ROS and the domain exist; all it does is connect them:
 *   - composition root: build adapters, filter, publishers, queue;
 *   - timestamp-ordered drain: convert each message to a Command, enqueue, and
 *     apply the queue in stamp order;
 *   - publish policy: each MAIN input owns and triggers its output.
 * No translation or estimator logic lives here.
 */
class ConeFusion : public rclcpp::Node {
private:
  /* color logic */
  bool is_colorblind;

  /* Subscriptions topics' names */
  std::string cones_topic, input_odom_topic, race_status_topic,
      mapped_cones_topic, output_odom_topic,
      output_frame_id, output_child_frame_id;

#ifdef CONE_FUSED_DEBUG
  /* Debug topic name and parameters — only compiled in a -DDEBUG=ON build. */
  std::string input_cones_debug_topic;

  /* If true, keep republishing the live map even after the corrected map is
     frozen (raw vs. corrected map comparison). */
  bool cones_pub_for_debug;

  /* If true, publish the raw input cones projected into the map frame via the
     current EKF pose (red markers) for an input-vs-map check */
  bool pub_input_cones_debug = false;
#endif

  std::string cuda_module_loading;

  /* Enable logging parameter */
  bool enable_logging;

  /* Actual Race status (current lap drives the lap-freeze latch). */
  mmr_base::msg::RaceStatus race_status;

  /* Min number of time a cone has to be seen in order to map it */
  uint32_t cone_time_seen_th;

  /* If true (default), lap 2+ freezes the map (rigid pose-only localization); if
     false, lap 2+ keeps refining pose AND landmarks (continuous SLAM, legacy) */
  bool freeze_map = true;

  /* Chi-square (2 DOF) gate for lap-2+ data association by Mahalanobis distance */
  double assoc_maha_gate = 9.21;

  /* If true (default), hold the filter back until FAST-LIMO has finished its IMU
     calibration: its motion frames are dropped by the adapter and cone scans are
     dropped here, so no cone map is published for the duration. */
  bool gate_on_limo_calibration = true;

  /* Eigen/OpenMP thread count for the CPU linear algebra (default 1). */
  int eigen_threads = 1;

  /* Late-arrival window [ms] for the timestamp-ordered drain. 0 = apply
     immediately in stamp order (no added latency); raise to absorb out-of-order
     measurements at the cost of that much publish latency on the main path. */
  double late_arrival_window_ms = 0.0;
  uint64_t late_arrival_window_ns = 0;

  /* EKF Parameters */
  /* Measurement noise R: (range, bearing) variances [m^2, rad^2]. */
  Vector2f meas_noise;
  double min_new_cone_distance;

  /* Process noise Q: [x,y per metre travelled, theta per radian turned]. */
  Vector2f proc_noise;

  /* ---- domain core + edges (composition root) ------------------------- */

  /* EKF SLAM filter object (ROS-free core). */
  std::shared_ptr<EKFOdom> ekf_odom;

  /* Inbound adapters: ROS msg -> domain command. */
  LimoMotionAdapter limo_adapter_;
  std::unique_ptr<ConeObsAdapter> cone_adapter_;

  /* Timestamp-ordered command queue (Command pattern). */
  CommandQueue cmd_queue_;

  /* Set once the (empty-config) child frame has been inherited from the first
     FAST-LIMO message, so the resolution happens exactly once. */
  bool child_frame_resolved_ = false;

  /* Outbound publishers: domain state -> ROS msg. Each MAIN input owns one. */
  std::unique_ptr<OdometryPublisher> odom_out_;   /* triggered by FAST-LIMO */
  std::unique_ptr<ConeMapPublisher> cone_map_out_; /* triggered by cones    */

#ifdef CONE_FUSED_DEBUG
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr inputConesDebugPub;
#endif

  /* ROS 2 Subscribers */
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr fast_limo_odom_sub;
  rclcpp::Subscription<visualization_msgs::msg::Marker>::SharedPtr cones_sub;
  rclcpp::Subscription<mmr_base::msg::RaceStatus>::SharedPtr race_status_sub;

  /* Subscriptions Callbacks */
  void conesCallback(const visualization_msgs::msg::Marker::SharedPtr cones_data);
  void fastLimoDataCallback(const nav_msgs::msg::Odometry::SharedPtr fast_limo_data);
  void raceStatusCallback(const mmr_base::msg::RaceStatus::SharedPtr race_status_data);

  /* Load node parameters */
  void loadParameters();

  /* Apply the buffered commands in timestamp order, up to the late-arrival
     horizon, and report what ran (for diagnostics). */
  CommandQueue::DrainResult drainQueue();

#ifdef CONE_FUSED_DEBUG
  /* Debug: project raw input cones into the map frame (current EKF pose) and
     publish them as red markers for an input-vs-map visual comparison */
  void pubInputConesDebug(const visualization_msgs::msg::Marker::SharedPtr &cones_data);
#endif

public:
  /* Constructor */
  ConeFusion();
};
