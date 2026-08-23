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

#ifdef LATENCY_TESTING
#include <chrono>
#include <mmr_base/msg/latency_sample.hpp>
#endif

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
      output_frame_id, output_child_frame_id, input_odom_frame_id;

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

  /* Rate [Hz] at which the `output_frame_id -> input_odom_frame_id` correction
     is re-broadcast from its own timer and callback group. 0 disables the timer
     entirely and restores the old behaviour EXACTLY -- the edge then goes out
     only from the tail of fastLimoDataCallback, which is the baseline arm.

     Why it exists: that callback shares a mutually-exclusive callback group
     with conesCallback, so the edge stops being published for the whole
     duration of an EKF update (`exe_ms` on the scan line). `track -> base_link`
     resolves at the latest time COMMON to the chain, and fast_LIMO's
     `odom -> base_link` keeps streaming from its own process, so the common
     time was gated by this edge alone and stalled with the association loop.
     Only a full-chain consumer sees that -- a viewer's follow camera lags while
     every topic, which needs a shorter path, looks fine. */
  double tf_republish_hz = 50.0;

  /* Late-arrival window [ms] for the timestamp-ordered drain. 0 = apply
     immediately in stamp order (no added latency); raise to absorb out-of-order
     measurements at the cost of that much publish latency on the main path. */
  double late_arrival_window_ms = 0.0;
  uint64_t late_arrival_window_ns = 0;

  /* EKF Parameters */
  /* Measurement noise R: (range, bearing) variances [m^2, rad^2]. */
  Vector2f meas_noise;
  double min_new_cone_distance;

  /* LAP-1 new-vs-existing gate (see EKFOdom::new_cone_sigma_scale_). The lap-1
     radius is min_new_cone_distance widened by this many pose sigmas and capped
     at max_new_cone_distance, so it adapts to the pose uncertainty like the
     lap-2+ Mahalanobis gate does instead of staying fixed on the lap that
     builds the map. sigma_scale 0 restores the old fixed radius. */
  double new_cone_sigma_scale = 3.0;
  double max_new_cone_distance = 4.0;

  /* Latch for the one-shot "map near capacity" warning (the full-map warning is
     throttled instead, since that state persists). */
  bool map_capacity_warned_ = false;

  /* ---- stream-pairing diagnostic ---------------------------------------
     The two inputs do NOT carry the same clock, and this measures it.
     /clusters is stamped with the SOURCE CLOUD'S ACQUISITION TIME
     (controller_node.cu:614, deliberately). /fast_limo/state is stamped with
     get_clock()->now() AT PUBLISH (LimoWrapper.cpp:501) -- the state's own
     measurement time, State::time, is available there and discarded. So the
     CommandQueue, whose whole purpose is to apply commands in TIMESTAMP order,
     is ordering two different epochs; with late_arrival_window_ms 0 it drains
     everything on every callback and degenerates to ARRIVAL order, which is why
     nothing has visibly broken. Arrival order is honest in bag mode (one player
     emits both streams) and is a scheduling artefact in sensor mode (pcap
     pacing vs rosbag2's clock, independent).
     What is logged: the predicts fused per cone scan, the cone-stamp interval
     (true sensor time), and the implied IMU rate. If the odometry stream runs
     ahead of the cloud stream, cone_fused integrates more motion between two
     cone scans than the elapsed sensor time accounts for and the implied rate
     rises above the real IMU rate -- the observation is then fused against a
     pose that has already moved past it. */
  uint64_t last_cone_stamp_ns_ = 0;
  uint64_t last_odom_stamp_ns_ = 0;

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
  bool odom_frame_resolved_ = false;

  /* Outbound publishers: domain state -> ROS msg. Each MAIN input owns one. */
  std::unique_ptr<OdometryPublisher> odom_out_;   /* triggered by FAST-LIMO */
  std::unique_ptr<ConeMapPublisher> cone_map_out_; /* triggered by cones    */

#ifdef CONE_FUSED_DEBUG
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr inputConesDebugPub;
#endif

  /* Timer that re-broadcasts the correction, and the callback group that keeps
     it off the estimator's thread. The group is MutuallyExclusive and holds
     ONLY this timer; every subscription stays in the node's default group, so
     the cones/odom/race_status callbacks remain serialised against each other
     exactly as they are today and no new race is introduced into the filter. */
  rclcpp::CallbackGroup::SharedPtr tf_cb_group_;
  rclcpp::TimerBase::SharedPtr tf_timer_;

  /* ROS 2 Subscribers */
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr fast_limo_odom_sub;
  rclcpp::Subscription<visualization_msgs::msg::Marker>::SharedPtr cones_sub;
  rclcpp::Subscription<mmr_base::msg::RaceStatus>::SharedPtr race_status_sub;

  /* Subscriptions Callbacks */
  void conesCallback(const visualization_msgs::msg::Marker::SharedPtr cones_data);

#ifdef LATENCY_TESTING
  /* Per-frame latency instrumentation. Reports this
     node's own two instants and nothing derived -- as_demo's monitor joins them
     against cone_rush's sample for the same frame to get delivery, and
     subtracts them from each other to get compute.

     Sampled on the CONE PATH only, i.e. conesCallback. /Odometry is published
     from fastLimoDataCallback, driven by FAST-LIMO rather than by the frame, so
     an interval measured across it would be a wait and not this node's work. */
  rclcpp::Publisher<mmr_base::msg::LatencySample>::SharedPtr latency_sample_pub_;
  uint32_t latency_seq_ = 0;
  int64_t latency_t_in_ = 0;

  static int64_t monotonicNs()
  {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  void publishLatencySample(const builtin_interfaces::msg::Time & frame_stamp, int64_t t_out);
#endif
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
