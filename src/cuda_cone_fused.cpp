#include <cuda_cone_fused/cuda_cone_fused.hpp>

#include <cmath>

ConeFusion::ConeFusion() : rclcpp::Node("cuda_cone_fused_node") {
  /* Load node parameters */
  this->loadParameters();

  /* EKF SLAM filter object */
  this->ekf_odom = std::make_shared<EKFOdom>(this->meas_noise, this->proc_noise, this->min_new_cone_distance, this->eigen_threads);
  this->ekf_odom->setFreezeMap(this->freeze_map);
  this->ekf_odom->setAssocMahaGate(static_cast<float>(this->assoc_maha_gate));

  /* Inbound adapter for cones (LIMO adapter is a value member). The colour
     classification Strategy is chosen once here by the factory from params. */
  this->cone_adapter_ =
      std::make_unique<ConeObsAdapter>(makeColorClassifier(this->is_colorblind));

  /* Outbound publishers (input-owned, event-driven). */
  this->odom_out_ = std::make_unique<OdometryPublisher>(
      this, this->output_odom_topic, this->output_frame_id, this->output_child_frame_id);

  bool republish_live_for_debug = false;
#ifdef CONE_FUSED_DEBUG
  republish_live_for_debug = this->cones_pub_for_debug;
  this->inputConesDebugPub = this->create_publisher<visualization_msgs::msg::Marker>(this->input_cones_debug_topic, 1);
#endif
  this->cone_map_out_ = std::make_unique<ConeMapPublisher>(
      this, this->mapped_cones_topic, this->output_frame_id,
      this->cone_time_seen_th, republish_live_for_debug);

  /* Create subscriptions */
  this->cones_sub = this->create_subscription<visualization_msgs::msg::Marker>(this->cones_topic, 100,std::bind(&ConeFusion::conesCallback, this, std::placeholders::_1));
  this->fast_limo_odom_sub = this->create_subscription<nav_msgs::msg::Odometry>(this->input_odom_topic, 1,std::bind(&ConeFusion::fastLimoDataCallback, this, std::placeholders::_1));
  this->race_status_sub = this->create_subscription<mmr_base::msg::RaceStatus>(this->race_status_topic, 1, std::bind(&ConeFusion::raceStatusCallback, this, std::placeholders::_1));

  /* Init race status */
  this->race_status = mmr_base::msg::RaceStatus();
  this->race_status.current_lap = 0;
}

void ConeFusion::loadParameters() {
  std::vector<double> tmp_meas_noise(2), tmp_proc_noise(2);
  declare_parameter("is_colorblind", true);

  declare_parameter("generic.output_frame_id", "track");
  declare_parameter("generic.output_child_frame_id", "imu_link");

  declare_parameter("generic.imu_topic", "/imu/data");
  declare_parameter("generic.cones_topic", "/clusters");
  declare_parameter("generic.input_odom_topic", "/fast_limo/state");
  declare_parameter("generic.race_status_topic", "/planning/race_status");

  declare_parameter("generic.mapped_cones_topic", "/slam/cones_positions");
  declare_parameter("generic.output_odom_topic", "/Odometry");

  declare_parameter("generic.enable_logging", false);
  declare_parameter("generic.cone_time_seen_th", 10);
  declare_parameter("generic.is_skidpad_mission", false);

#ifdef CONE_FUSED_DEBUG
  /* Debug-only parameters (compiled in only with -DDEBUG=ON). */
  declare_parameter("generic.cones_pub_for_debug", false);
  declare_parameter("generic.input_cones_debug_topic", "/slam/input_cones_debug");
  declare_parameter("generic.pub_input_cones_debug", false);
#endif

  /* Freeze the map from lap 2 (rigid pose-only localization) vs. continuous SLAM */
  declare_parameter("generic.freeze_map", true);

  /* Chi-square (2 DOF) gate for lap-2+ Mahalanobis data association */
  declare_parameter("generic.assoc_maha_gate", 9.21);

  /* Eigen/OpenMP thread count for the CPU linear algebra (default 1). */
  declare_parameter("generic.eigen_threads", 1);

  /* Late-arrival window [ms] for the timestamp-ordered drain (default 0). */
  declare_parameter("generic.late_arrival_window_ms", 0.0);

  /* Declare Sensor Noise parameters */
  declare_parameter<std::vector<double>>("noises.meas_noise", std::vector<double>{0.0, 0.0});
  declare_parameter<std::vector<double>>("noises.proc_noise", std::vector<double>{0.0, 0.0});
  declare_parameter("noises.min_new_cone_distance", 2.0);

  /* Get Parameters */
  get_parameter("is_colorblind", this->is_colorblind);

  get_parameter("generic.output_frame_id", this->output_frame_id);
  get_parameter("generic.output_child_frame_id", this->output_child_frame_id);

  get_parameter("generic.imu_topic", this->imu_topic);
  get_parameter("generic.cones_topic", this->cones_topic);
  get_parameter("generic.input_odom_topic", this->input_odom_topic);
  get_parameter("generic.race_status_topic", this->race_status_topic);

  get_parameter("generic.mapped_cones_topic", this->mapped_cones_topic);
  get_parameter("generic.output_odom_topic", this->output_odom_topic);

  get_parameter("generic.enable_logging", this->enable_logging);
  get_parameter("generic.is_skidpad_mission", this->is_skidpad_mission);

#ifdef CONE_FUSED_DEBUG
  get_parameter("generic.cones_pub_for_debug", this->cones_pub_for_debug);
  get_parameter("generic.input_cones_debug_topic", this->input_cones_debug_topic);
  get_parameter("generic.pub_input_cones_debug", this->pub_input_cones_debug);
#endif

  get_parameter("generic.freeze_map", this->freeze_map);
  get_parameter("generic.assoc_maha_gate", this->assoc_maha_gate);
  get_parameter("generic.eigen_threads", this->eigen_threads);
  get_parameter("generic.late_arrival_window_ms", this->late_arrival_window_ms);
  this->late_arrival_window_ns =
      static_cast<uint64_t>(this->late_arrival_window_ms * 1e6);

  std::cout << "IS_SKIDPAD: " << this->is_skidpad_mission << "\n";

  /* Get Sensor Noise parameters */
  get_parameter("noises.meas_noise", tmp_meas_noise);
  get_parameter("noises.proc_noise", tmp_proc_noise);
  get_parameter("noises.min_new_cone_distance", this->min_new_cone_distance);
  get_parameter("generic.cone_time_seen_th", this->cone_time_seen_th);

  /* Copy noise parameters */
  for (size_t i = 0; i < 2; i++) {
    this->meas_noise(i) = (float)tmp_meas_noise[i];
    this->proc_noise(i) = (float)tmp_proc_noise[i];
  }
}

/* Apply buffered commands in timestamp order up to the late-arrival horizon. */
CommandQueue::DrainResult ConeFusion::drainQueue() {
  const uint64_t newest = this->cmd_queue_.newestStamp();
  const uint64_t horizon =
      (newest > this->late_arrival_window_ns) ? (newest - this->late_arrival_window_ns) : 0;
  return this->cmd_queue_.drainUpTo(horizon, *this->ekf_odom);
}

/* Callbacks */

void ConeFusion::conesCallback(const visualization_msgs::msg::Marker::SharedPtr cones_data)
{
  /* Translate the transient vehicle-frame observations into a domain Command and
     enqueue it (cones = MAIN correction input). */
  const size_t detected_cones = cones_data->points.size();
  const uint64_t stamp_ns =
      static_cast<uint64_t>(cones_data->header.stamp.sec) * 1000000000ull +
      static_cast<uint64_t>(cones_data->header.stamp.nanosec);
  this->cmd_queue_.push(
      FilterCommand::makeUpdate(this->cone_adapter_->convert(*cones_data), stamp_ns));

  /* Drain in timestamp order, then publish the map (cones own this output). */
  Vector3f pose_pre = this->ekf_odom->getState().head(3);
  rclcpp::Time start = this->now();

  CommandQueue::DrainResult res = this->drainQueue();

  rclcpp::Time end = this->now();
  rclcpp::Duration exe_time = end - start;

  if (this->enable_logging) {
    /* Correction-health diagnostic: how many cones passed association this drain
       (corrected), the resulting pose move (|dpos|, |dyaw|), and the current pose
       covariance. If "corrected" trends to 0 while P climbs and |dpos| shrinks
       over the final laps, the corrections are dropping out (drift cascade). */
    Vector3f pose_post = this->ekf_odom->getState().head(3);
    const double dpos = std::hypot((double)(pose_post(0) - pose_pre(0)),
                                   (double)(pose_post(1) - pose_pre(1)));
    const double dyaw = std::fabs(this->ekf_odom->normalizeAngle(pose_post(2) - pose_pre(2)));
    Vector3f pcov = this->ekf_odom->getPoseCovariance();
    RCLCPP_INFO(this->get_logger(),
                "scan: detected=%zu corrected=%zu |dpos|=%.4f |dyaw|=%.4f "
                "Pxx=%.5f Pyy=%.5f Pyaw=%.5f exe_ms=%.3f",
                detected_cones, res.associated, dpos, dyaw,
                pcov(0), pcov(1), pcov(2), exe_time.nanoseconds() * 1e-6);
  }

#ifdef CONE_FUSED_DEBUG
  /* Debug: project the raw input cones into the map frame via the current EKF
     pose and publish them as red markers (input vs. map check). */
  if (this->pub_input_cones_debug)
    this->pubInputConesDebug(cones_data);
#endif

  this->cone_map_out_->publish(*this->ekf_odom, cones_data->header.stamp,
                               this->race_status.current_lap);
}

void ConeFusion::fastLimoDataCallback(const nav_msgs::msg::Odometry::SharedPtr fast_limo_data)
{
  /* Translate FAST-LIMO odometry into a motion Command and enqueue it (LIMO =
     MAIN motion input). The adapter returns nullopt on the very first frame (it
     only seeds the relative-motion reference). */
  /* Inherit the output child frame from FAST-LIMO when left empty in config
     (the body the pose describes is the same one LIMO tracks). Resolved once.
     The parent frame_id is left as configured: it is the cone-corrected `track`
     frame, distinct from LIMO's frame. */
  if (this->output_child_frame_id.empty() && !this->child_frame_resolved_) {
    this->odom_out_->setChildFrameId(fast_limo_data->child_frame_id);
    this->child_frame_resolved_ = true;
    RCLCPP_INFO(this->get_logger(),
                "output_child_frame_id empty in config; inherited '%s' from %s",
                fast_limo_data->child_frame_id.c_str(), this->input_odom_topic.c_str());
  }

  std::optional<MotionIncrement> inc = this->limo_adapter_.convert(*fast_limo_data);
  if (inc) {
    this->cmd_queue_.push(FilterCommand::makePredict(*inc));
  }

  /* Drain in timestamp order, then publish odom (FAST-LIMO owns this output).
     We publish the EKF state (corrected by the cones), NOT the raw FAST-LIMO
     pose — this is what lets the cone corrections survive on the wire. */
  this->drainQueue();
  this->odom_out_->publish(*this->ekf_odom, fast_limo_data->header.stamp);
}

#ifdef CONE_FUSED_DEBUG
void ConeFusion::pubInputConesDebug(
    const visualization_msgs::msg::Marker::SharedPtr &cones_data) {
  /* Current EKF pose: project the raw (vehicle-frame) input cones into the map
     frame, so they can be overlaid on the mapped cones. global = pose_xy +
     R(theta) * point. If the pose is good, red lands on the yellow map; if it
     has drifted, red peels away — directly visualising input vs. map. */
  Vector3f pose = this->ekf_odom->getState().head(3);
  const double ct = cos(pose(2));
  const double st = sin(pose(2));

  visualization_msgs::msg::Marker dbg;
  dbg.header.frame_id = this->output_frame_id;
  dbg.header.stamp = cones_data->header.stamp;
  dbg.ns = "InputConesDebug";
  dbg.id = 0;
  dbg.type = visualization_msgs::msg::Marker::SPHERE_LIST;
  dbg.action = visualization_msgs::msg::Marker::ADD;
  dbg.scale.x = dbg.scale.y = dbg.scale.z = 0.3;
  dbg.pose.orientation.w = 1.0;
  dbg.color.r = 1.0;
  dbg.color.g = 0.0;
  dbg.color.b = 0.0;
  dbg.color.a = 1.0;
  dbg.points.reserve(cones_data->points.size());

  for (const auto &pt : cones_data->points) {
    geometry_msgs::msg::Point p;
    p.x = pose(0) + ct * pt.x - st * pt.y;
    p.y = pose(1) + st * pt.x + ct * pt.y;
    p.z = 0.0;
    dbg.points.push_back(p);
  }

  this->inputConesDebugPub->publish(dbg);
}
#endif  /* CONE_FUSED_DEBUG */

void ConeFusion::raceStatusCallback(
    const mmr_base::msg::RaceStatus::SharedPtr race_status_data) {
  /* Update race status */
  this->race_status = *race_status_data;
}
