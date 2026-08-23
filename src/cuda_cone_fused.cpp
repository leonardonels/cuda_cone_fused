#include <cuda_cone_fused/cuda_cone_fused.hpp>

#include <chrono>
#include <cmath>

ConeFusion::ConeFusion() : rclcpp::Node("cuda_cone_fused_node") {
  /* Load node parameters */
  this->loadParameters();

  setenv("CUDA_MODULE_LOADING", static_cast<const char*>(this->cuda_module_loading.c_str()), /*overwrite=*/0);  // 0 = let an explicit env var still win

  /* EKF SLAM filter object */
  this->ekf_odom = std::make_shared<EKFOdom>(this->meas_noise, this->proc_noise, this->min_new_cone_distance, this->eigen_threads);
  this->ekf_odom->setFreezeMap(this->freeze_map);
  this->ekf_odom->setAssocMahaGate(static_cast<float>(this->assoc_maha_gate));
  this->ekf_odom->setNewConeGate(static_cast<float>(this->new_cone_sigma_scale),
                                 static_cast<float>(this->max_new_cone_distance));

  /* Arm (or disable) the FAST-LIMO IMU-calibration gate. The adapter detects the
     calibration by the signature of the frames LIMO publishes, so no timer here
     has to agree with LIMO's `calibration.time`. */
  this->limo_adapter_.setGateOnCalibration(this->gate_on_limo_calibration);

  /* Inbound adapter for cones (LIMO adapter is a value member). The colour
     classification Strategy is chosen once here by the factory from params. */
  this->cone_adapter_ =
      std::make_unique<ConeObsAdapter>(makeColorClassifier(this->is_colorblind));

  /* Outbound publishers (input-owned, event-driven). */
  this->odom_out_ = std::make_unique<OdometryPublisher>(
      this, this->output_odom_topic, this->output_frame_id, this->output_child_frame_id,
      /*tf_from_timer=*/this->tf_republish_hz > 0.0);

  /* Resolve the TF parent from config up front so the frame graph is defined
     before a single message arrives. Only an empty value falls back to
     inheriting it from the input odometry (see fastLimoDataCallback). */
  if (!this->input_odom_frame_id.empty()) {
    this->odom_out_->setOdomFrameId(this->input_odom_frame_id);
    this->odom_frame_resolved_ = true;
    RCLCPP_INFO(this->get_logger(), "broadcasting TF %s -> %s (cone correction)",
                this->output_frame_id.c_str(), this->input_odom_frame_id.c_str());
  }

  bool republish_live_for_debug = false;
#ifdef CONE_FUSED_DEBUG
  republish_live_for_debug = this->cones_pub_for_debug;
  this->inputConesDebugPub = this->create_publisher<visualization_msgs::msg::Marker>(this->input_cones_debug_topic, 1);
#endif
  this->cone_map_out_ = std::make_unique<ConeMapPublisher>(
      this, this->mapped_cones_topic, this->output_frame_id,
      this->cone_time_seen_th, republish_live_for_debug);

#ifdef LATENCY_TESTING
  this->latency_sample_pub_ = this->create_publisher<mmr_base::msg::LatencySample>(
      "/latency/sample/cone_fused", rclcpp::QoS(rclcpp::KeepLast(100)).reliable());
  RCLCPP_INFO(this->get_logger(),
      "LATENCY_TESTING: per-frame samples on /latency/sample/cone_fused");
#endif

  /* Create subscriptions */
  this->cones_sub = this->create_subscription<visualization_msgs::msg::Marker>(this->cones_topic, 100,std::bind(&ConeFusion::conesCallback, this, std::placeholders::_1));
  this->fast_limo_odom_sub = this->create_subscription<nav_msgs::msg::Odometry>(this->input_odom_topic, 10,std::bind(&ConeFusion::fastLimoDataCallback, this, std::placeholders::_1));
  this->race_status_sub = this->create_subscription<mmr_base::msg::RaceStatus>(this->race_status_topic, 1, std::bind(&ConeFusion::raceStatusCallback, this, std::placeholders::_1));

  /* Re-broadcast the correction on its own thread. Created AFTER the
     subscriptions so the default (MutuallyExclusive) group already owns them:
     this timer is the only member of tf_cb_group_, and the only state it
     touches is the last-correction snapshot inside OdometryPublisher, behind
     that class's own mutex. It never reads the filter.

     Needs a MultiThreadedExecutor to be worth anything -- see
     cuda_cone_fused_node.cpp. Under a single-threaded executor it would simply
     queue behind the EKF update and change nothing. */
  if (this->tf_republish_hz > 0.0) {
    this->tf_cb_group_ =
        this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    const auto period = std::chrono::duration<double>(1.0 / this->tf_republish_hz);
    this->tf_timer_ = this->create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(period),
        [this]() { this->odom_out_->republishTf(this->now()); },
        this->tf_cb_group_);
    RCLCPP_INFO(this->get_logger(),
                "re-broadcasting %s -> %s at %.1f Hz from a dedicated callback group "
                "(the EKF update no longer stalls it)",
                this->output_frame_id.c_str(), this->input_odom_frame_id.c_str(),
                this->tf_republish_hz);
  } else {
    RCLCPP_INFO(this->get_logger(),
                "generic.tf_republish_hz = 0: the correction goes out only from the "
                "FAST-LIMO callback, so it stalls for the duration of each EKF update "
                "(baseline behaviour)");
  }

  /* Init race status */
  this->race_status = mmr_base::msg::RaceStatus();
  this->race_status.current_lap = 0;
}

void ConeFusion::loadParameters() {
  std::vector<double> tmp_meas_noise(2), tmp_proc_noise(2);
  declare_parameter("is_colorblind", true);

  declare_parameter("generic.output_frame_id", "track");
  declare_parameter("generic.output_child_frame_id", "base_link");
  /* Parent of the TF edge we broadcast: `output_frame_id -> input_odom_frame_id`.
     Declaring it makes the whole tree readable from config; left empty it is
     inherited from the input odometry's header, which works but means the frame
     graph can only be discovered by running the stack. */
  declare_parameter("generic.input_odom_frame_id", "odom");

  declare_parameter("generic.cones_topic", "/clusters");
  declare_parameter("generic.input_odom_topic", "/fast_limo/state");
  declare_parameter("generic.race_status_topic", "/planning/race_status");

  declare_parameter("generic.mapped_cones_topic", "/slam/cones_positions");
  declare_parameter("generic.output_odom_topic", "/Odometry");

  declare_parameter("generic.enable_logging", false);
  declare_parameter("generic.cone_time_seen_th", 10);

#ifdef CONE_FUSED_DEBUG
  /* Debug-only parameters (compiled in only with -DDEBUG=ON). */
  declare_parameter("generic.cones_pub_for_debug", false);
  declare_parameter("generic.input_cones_debug_topic", "/slam/input_cones_debug");
  declare_parameter("generic.pub_input_cones_debug", false);
#endif

  declare_parameter("generic.CUDA_MODULE_LOADING", "LAZY");

  /* Freeze the map from lap 2 (rigid pose-only localization) vs. continuous SLAM */
  declare_parameter("generic.freeze_map", true);

  /* Chi-square (2 DOF) gate for lap-2+ Mahalanobis data association */
  declare_parameter("generic.assoc_maha_gate", 9.21);

  /* Hold the filter back for the duration of FAST-LIMO's IMU calibration */
  declare_parameter("generic.gate_on_limo_calibration", true);

  /* Eigen/OpenMP thread count for the CPU linear algebra (default 1). */
  declare_parameter("generic.eigen_threads", 1);

  /* Late-arrival window [ms] for the timestamp-ordered drain (default 0). */
  declare_parameter("generic.late_arrival_window_ms", 0.0);

  /* Re-broadcast rate [Hz] for the cone correction; 0 restores the old
     publish-from-the-callback-only behaviour exactly. */
  declare_parameter("generic.tf_republish_hz", 50.0);

  /* Declare Sensor Noise parameters */
  declare_parameter<std::vector<double>>("noises.meas_noise", std::vector<double>{0.7, 0.3});
  declare_parameter<std::vector<double>>("noises.proc_noise", std::vector<double>{0.05, 0.02});
  declare_parameter("noises.min_new_cone_distance", 2.0);
  declare_parameter("noises.new_cone_sigma_scale", 3.0);
  declare_parameter("noises.max_new_cone_distance", 4.0);

  /* Get Parameters */
  get_parameter("is_colorblind", this->is_colorblind);

  get_parameter("generic.output_frame_id", this->output_frame_id);
  get_parameter("generic.output_child_frame_id", this->output_child_frame_id);
  get_parameter("generic.input_odom_frame_id", this->input_odom_frame_id);

  get_parameter("generic.cones_topic", this->cones_topic);
  get_parameter("generic.input_odom_topic", this->input_odom_topic);
  get_parameter("generic.race_status_topic", this->race_status_topic);

  get_parameter("generic.mapped_cones_topic", this->mapped_cones_topic);
  get_parameter("generic.output_odom_topic", this->output_odom_topic);

  get_parameter("generic.enable_logging", this->enable_logging);

#ifdef CONE_FUSED_DEBUG
  get_parameter("generic.cones_pub_for_debug", this->cones_pub_for_debug);
  get_parameter("generic.input_cones_debug_topic", this->input_cones_debug_topic);
  get_parameter("generic.pub_input_cones_debug", this->pub_input_cones_debug);
#endif

  get_parameter("generic.CUDA_MODULE_LOADING", this->cuda_module_loading);

  get_parameter("generic.freeze_map", this->freeze_map);
  get_parameter("generic.assoc_maha_gate", this->assoc_maha_gate);
  get_parameter("generic.gate_on_limo_calibration", this->gate_on_limo_calibration);
  get_parameter("generic.eigen_threads", this->eigen_threads);
  get_parameter("generic.late_arrival_window_ms", this->late_arrival_window_ms);
  get_parameter("generic.tf_republish_hz", this->tf_republish_hz);
  this->late_arrival_window_ns =
      static_cast<uint64_t>(this->late_arrival_window_ms * 1e6);

  /* Get Sensor Noise parameters */
  get_parameter("noises.meas_noise", tmp_meas_noise);
  get_parameter("noises.proc_noise", tmp_proc_noise);
  get_parameter("noises.min_new_cone_distance", this->min_new_cone_distance);
  get_parameter("noises.new_cone_sigma_scale", this->new_cone_sigma_scale);
  get_parameter("noises.max_new_cone_distance", this->max_new_cone_distance);
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

#ifdef LATENCY_TESTING
void ConeFusion::publishLatencySample(const builtin_interfaces::msg::Time & frame_stamp,
                                      int64_t t_out)
{
  if (!latency_sample_pub_) return;
  mmr_base::msg::LatencySample m;
  m.frame_stamp = frame_stamp;
  m.t_in = latency_t_in_;
  m.t_out = t_out;
  m.seq = latency_seq_++;
  latency_sample_pub_->publish(m);
}
#endif

void ConeFusion::conesCallback(const visualization_msgs::msg::Marker::SharedPtr cones_data)
{
#ifdef LATENCY_TESTING
  /* First statement: after the middleware delivered and deserialised the
     marker, which is where the delivery interval has to end. */
  latency_t_in_ = monotonicNs();
#endif
  /* FAST-LIMO IMU-calibration gate. Until LIMO is calibrated its pose is a
     frozen placeholder (see LimoMotionAdapter::convert), so the scan is DROPPED
     outright rather than merely withheld from the output: fusing it would seed
     the landmark means, the colour votes and the cone_time_seen_th counters from
     a pose that is meaningless and about to be re-anchored to gravity, leaving a
     doubled/smeared map once the real pose arrives.

     Publishing nothing is also the point, not a side effect: with no cone map
     the planner cannot produce a trajectory, so the car stays put — which is
     exactly the standstill LIMO's IMU calibration assumes. */
  if (this->gate_on_limo_calibration && !this->limo_adapter_.isCalibrated()) {
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                         "FAST-LIMO still calibrating: dropping cone scans and "
                         "withholding the cone map");
    return;
  }

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

  /* ---- map-capacity cliff ----------------------------------------------
     Once landmark_count hits N_CONES the association loop drops EVERY further
     observation, good ones included, and the filter is permanently blind — it
     does not degrade, it stops. That state used to announce itself with
     nothing. Warn on the first drop and keep warning while it persists; also
     warn once when the map crosses a high-water fraction of capacity, which is
     the early sign that the map is filling with duplicates. */
  {
    const EKFOdom::AssocStats& as = this->ekf_odom->getAssocStats();
    const size_t mapped = this->ekf_odom->getActMappedLandmarks();
    const size_t capacity = EKFOdom::landmarkCapacity();
    if (as.dropped_full > 0) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                           "cone map FULL (%zu/%zu landmarks): dropped %zu observation(s) "
                           "this scan and every further one — the filter is BLIND, not "
                           "degraded. The map is almost certainly full of duplicates.",
                           mapped, capacity, as.dropped_full);
    } else if (!this->map_capacity_warned_ &&
               mapped >= (capacity * 4) / 5) {
      RCLCPP_WARN(this->get_logger(),
                  "cone map at %zu/%zu landmarks (>=80%% of capacity). A real track is "
                  "~130 cones: this is duplicate growth, and at %zu the filter goes blind.",
                  mapped, capacity, capacity);
      this->map_capacity_warned_ = true;
    }

    if (this->enable_logging) {
      /* Correction-health diagnostic: how many cones passed association this drain
         (corrected), the resulting pose move (|dpos|, |dyaw|), and the current pose
         covariance. If "corrected" trends to 0 while P climbs and |dpos| shrinks
         over the final laps, the corrections are dropping out (drift cascade).

         mapped/nn_max/nn_mean/gate are the association-runaway measurement:
         nn_* is the offset between the observations projected through the
         current pose and the map (the red-vs-yellow offset on
         /slam/input_cones_debug). If its p95 climbs toward `gate` while
         `mapped` climbs toward the capacity, the duplicate loop is confirmed.
         The p95 is taken offline from these per-scan lines. */
      Vector3f pose_post = this->ekf_odom->getState().head(3);
      const double dpos = std::hypot((double)(pose_post(0) - pose_pre(0)),
                                     (double)(pose_post(1) - pose_pre(1)));
      const double dyaw = std::fabs(this->ekf_odom->normalizeAngle(pose_post(2) - pose_pre(2)));
      Vector3f pcov = this->ekf_odom->getPoseCovariance();
      /* Stream pairing (see last_cone_stamp_ns_): dt_cone is the interval
         between consecutive cone scans in TRUE SENSOR TIME; predicts is how
         many odometry frames were integrated across it; imu_hz is the implied
         rate. epoch is the raw gap between the two streams' stamps, which is
         the different-clocks fact itself. */
      const double dt_cone_ms =
          (this->last_cone_stamp_ns_ != 0 && stamp_ns > this->last_cone_stamp_ns_)
              ? (stamp_ns - this->last_cone_stamp_ns_) * 1e-6 : 0.0;
      const double imu_hz = (dt_cone_ms > 0.0) ? (res.predicts * 1000.0 / dt_cone_ms) : 0.0;
      const double epoch_s =
          (static_cast<double>(this->last_odom_stamp_ns_) - static_cast<double>(stamp_ns)) * 1e-9;
      this->last_cone_stamp_ns_ = stamp_ns;

      RCLCPP_INFO(this->get_logger(),
                  "scan: predicts=%u dt_cone=%.2f imu_hz=%.1f epoch=%.3f "
                  "detected=%zu corrected=%zu mapped=%zu/%zu new=%zu "
                  "new_nn=[%.3f,%.3f] "
                  "nn_max=%.3f nn_mean=%.3f gate=%.3f drop_gate=%zu drop_full=%zu "
                  "|dpos|=%.4f |dyaw|=%.4f "
                  "Pxx=%.5f Pyy=%.5f Pyaw=%.5f exe_ms=%.3f",
                  res.predicts, dt_cone_ms, imu_hz, epoch_s,
                  detected_cones, res.associated, mapped, capacity, as.created,
                  as.new_nn_min, as.new_nn_max,
                  as.nn_max, as.nn_mean, as.gate_radius, as.dropped_gate, as.dropped_full,
                  dpos, dyaw,
                  pcov(0), pcov(1), pcov(2), exe_time.nanoseconds() * 1e-6);
    }
  }

#ifdef CONE_FUSED_DEBUG
  /* Debug: project the raw input cones into the map frame via the current EKF
     pose and publish them as red markers (input vs. map check). */
  if (this->pub_input_cones_debug)
    this->pubInputConesDebug(cones_data);
#endif

#ifdef LATENCY_TESTING
  /* Before the publish, never after -- so this node's serialisation lands in
     the NEXT hop's delivery. The early returns above are deliberately NOT
     sampled: a dropped scan published nothing, so it has no t_out. */
  const int64_t latency_t_out = monotonicNs();
#endif
  this->cone_map_out_->publish(*this->ekf_odom, cones_data->header.stamp,
                               this->race_status.current_lap);
#ifdef LATENCY_TESTING
  publishLatencySample(cones_data->header.stamp, latency_t_out);
#endif
}

void ConeFusion::fastLimoDataCallback(const nav_msgs::msg::Odometry::SharedPtr fast_limo_data)
{
  /* Translate FAST-LIMO odometry into a motion Command and enqueue it (LIMO =
     MAIN motion input). The adapter returns nullopt on the very first frame (it
     only seeds the relative-motion reference). */
  /* Both output frames are normally set from config (see loadParameters and the
     constructor). The two blocks below are the fallback for an empty value:
     inherit from the input odometry, once, and say so at WARN -- an inherited
     frame means the frame graph cannot be read from the YAML, which the frame
     convention (docs/FRAMES.md, rule 4) exists to prevent. */
  if (this->output_child_frame_id.empty() && !this->child_frame_resolved_) {
    this->odom_out_->setChildFrameId(fast_limo_data->child_frame_id);
    this->child_frame_resolved_ = true;
    RCLCPP_WARN(this->get_logger(),
                "generic.output_child_frame_id empty in config; inherited '%s' "
                "from %s. Set it explicitly.",
                fast_limo_data->child_frame_id.c_str(), this->input_odom_topic.c_str());
  }

  /* The TF we broadcast is `output_frame_id -> input_odom_frame_id`, the
     correction: the edge ABOVE the estimate we are correcting, never alongside
     it (FRAMES.md rule 2). Falling back to the input's own header keeps a
     misconfigured stack running, but the value belongs in config. */
  if (!this->odom_frame_resolved_ && !fast_limo_data->header.frame_id.empty()) {
    this->odom_out_->setOdomFrameId(fast_limo_data->header.frame_id);
    this->odom_frame_resolved_ = true;
    RCLCPP_WARN(this->get_logger(),
                "generic.input_odom_frame_id empty in config; inherited '%s' "
                "from %s. Broadcasting TF %s -> %s. Set it explicitly.",
                fast_limo_data->header.frame_id.c_str(), this->input_odom_topic.c_str(),
                this->output_frame_id.c_str(), fast_limo_data->header.frame_id.c_str());
  }

  /* The adapter returns nullopt for every frame published during LIMO's IMU
     calibration and latches `isCalibrated()` on the first real one; announce
     that transition once, since it is what releases the cone map. */
  const bool was_calibrated = this->limo_adapter_.isCalibrated();
  std::optional<MotionIncrement> inc = this->limo_adapter_.convert(*fast_limo_data);
  if (!was_calibrated && this->limo_adapter_.isCalibrated()) {
    RCLCPP_INFO(this->get_logger(),
                "FAST-LIMO calibration complete: consuming odometry from %s and "
                "publishing the cone map",
                this->input_odom_topic.c_str());
  }
  if (inc) {
    this->cmd_queue_.push(FilterCommand::makePredict(*inc));
  }
  this->last_odom_stamp_ns_ =
      static_cast<uint64_t>(fast_limo_data->header.stamp.sec) * 1000000000ull +
      static_cast<uint64_t>(fast_limo_data->header.stamp.nanosec);

  /* Drain in timestamp order, then publish odom (FAST-LIMO owns this output).
     We publish the EKF state (corrected by the cones), NOT the raw FAST-LIMO
     pose — this is what lets the cone corrections survive on the wire. */
  this->drainQueue();
  this->odom_out_->publish(*this->ekf_odom, fast_limo_data->header.stamp, *fast_limo_data);
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
