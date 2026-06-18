#include <cuda_cone_fused/cuda_cone_fused.hpp>

ConeFusion::ConeFusion() : rclcpp::Node("cuda_cone_fused_node") {
  /* Load node parameters */
  this->loadParameters();

  /* Create publisher */
  this->odom_pub = this->create_publisher<nav_msgs::msg::Odometry>(this->output_odom_topic, 1);
  this->conesPositionsMarkerPub = this->create_publisher<visualization_msgs::msg::Marker>(this->mapped_cones_topic, 1);
#ifdef CONE_FUSED_DEBUG
  this->inputConesDebugPub = this->create_publisher<visualization_msgs::msg::Marker>(this->input_cones_debug_topic, 1);
#endif

  /* Create subscriptions */
  this->cones_sub = this->create_subscription<visualization_msgs::msg::Marker>(this->cones_topic, 100,std::bind(&ConeFusion::conesCallback, this, std::placeholders::_1));
  this->fast_limo_odom_sub = this->create_subscription<nav_msgs::msg::Odometry>(this->input_odom_topic, 1,std::bind(&ConeFusion::fastLimoDataCallback, this, std::placeholders::_1));
  this->race_status_sub = this->create_subscription<mmr_base::msg::RaceStatus>(this->race_status_topic, 1, std::bind(&ConeFusion::raceStatusCallback, this, std::placeholders::_1));

  /* Initialize the transform broadcaster */
  this->tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  /* Initialize vehicle pose */
  // 0 for now, this can change if a relocalisations is implemented using fast limo pose as global reference
  this->act_position << 0.0, 0.0, 0.0;
  this->act_orientation.set__w(1.0);
  this->act_orientation.set__x(0.0);
  this->act_orientation.set__y(0.0);
  this->act_orientation.set__z(0.0);

  /* Create EKF SLAM filter object */
  this->ekf_odom = std::make_shared<EKFOdom>(this->proc_noise, this->motion_noise, this->min_new_cone_distance, this->eigen_threads);
  this->ekf_odom->setFreezeMap(this->freeze_map);
  this->ekf_odom->setAssocMahaGate(static_cast<float>(this->assoc_maha_gate));

  /* Init mapped cones markers */
  this->initConesMarker(this->conesMarker);
  this->initConesMarker(this->correctedConesMarker);

  /* Init race status */
  this->race_status = mmr_base::msg::RaceStatus();
  this->race_status.current_lap = 0;
}

void ConeFusion::loadParameters() {
  std::vector<double> tmp_proc_noise(2), tmp_motion_noise(2);
  declare_parameter("is_colorblind", true);

  declare_parameter("generic.output_frame_id", "track");
  declare_parameter("generic.output_child_frame_id", "imu_link");

  declare_parameter("generic.imu_topic", "/imu/data");
  declare_parameter("generic.cones_topic", "/clusters");
  declare_parameter("generic.input_odom_topic", "/fast_limo/state");
  declare_parameter("generic.race_status_topic", "/planning/race_status");
  // declare_parameter("generic.gps_speed_topic", "/speed/gps");
  // declare_parameter("generic.gps_data_topic", "/gps/data");
  
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

  /* Declare Sensor Noise parameters */
  declare_parameter<std::vector<double>>("noises.proc_noise", std::vector<double>{0.0, 0.0});
  declare_parameter<std::vector<double>>("noises.motion_noise", std::vector<double>{0.0, 0.0});
  declare_parameter("noises.min_new_cone_distance", 2.0);

  /* Get Parameters */
  get_parameter("is_colorblind", this->is_colorblind);

  get_parameter("generic.output_frame_id", this->output_frame_id);
  get_parameter("generic.output_child_frame_id", this->output_child_frame_id);

  get_parameter("generic.imu_topic", this->imu_topic);
  get_parameter("generic.cones_topic", this->cones_topic);
  get_parameter("generic.input_odom_topic", this->input_odom_topic);
  get_parameter("generic.race_status_topic", this->race_status_topic);
  // get_parameter("generic.gps_speed_topic", this->gps_speed_topic);
  // get_parameter("generic.gps_data_topic", this->gps_data_topic);
  
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

  std::cout << "IS_SKIDPAD: " << this->is_skidpad_mission << "\n";

  /* Get Sensor Noise parameters */
  get_parameter("noises.proc_noise", tmp_proc_noise);
  get_parameter("noises.motion_noise", tmp_motion_noise);
  get_parameter("noises.min_new_cone_distance", this->min_new_cone_distance);
  get_parameter("generic.cone_time_seen_th", this->cone_time_seen_th);


  /* Copy noise parameters */
  for (size_t i = 0; i < 2; i++) {
    this->proc_noise(i) = (float)tmp_proc_noise[i];
    this->motion_noise(i) = (float)tmp_motion_noise[i];
  }
}

void ConeFusion::initConesMarker(visualization_msgs::msg::Marker &cones) {
  cones.header.frame_id = this->output_frame_id;
  cones.ns = "ConesAbsolutePos";
  cones.id = 0;
  cones.type = visualization_msgs::msg::Marker::SPHERE_LIST;
  cones.action = visualization_msgs::msg::Marker::ADD;
  cones.scale.x = 0.35;
  cones.scale.y = 0.35;
  cones.scale.z = 0.35;
  cones.pose.position.x = 0.0;
  cones.pose.position.y = 0.0;
  cones.pose.position.z = 0.0;
  // Initialize with identity quaternion (no rotation)
  cones.pose.orientation.w = 1.0;
  cones.pose.orientation.x = 0.0;
  cones.pose.orientation.y = 0.0;
  cones.pose.orientation.z = 0.0;
}

/* Callbacks */

void ConeFusion::conesCallback(const visualization_msgs::msg::Marker::SharedPtr cones_data) 
{
  /* If corrected cons are created -> Skip callback */
  // if(this->corrected_cones_created)
  // {
  //     return;
  // }
  /* By refusing to updates the cones (the center line from the local planner 
    won't be updated anyway) the ekf should work as an achor for possible fast limo drift */
  size_t detected_cones = cones_data->points.size();

  Vector3f *z = (Vector3f *)malloc(detected_cones * sizeof(Vector3f));
  for (size_t i = 0; i < detected_cones; i++) {
    z[i](0) = sqrt(pow(cones_data->points[i].x, 2) +
                   pow(cones_data->points[i].y, 2)); // + coneRadius;
    z[i](1) = atan2(cones_data->points[i].y, cones_data->points[i].x);

    z[i](2) = this->is_colorblind ? yellowCone : this->getConeColor(cones_data->colors[i]);
    /* Normalize angle */
    z[i](1) = this->ekf_odom->normalizeAngle(z[i](1));
  }

  rclcpp::Time start, end;
  Vector3f pose_pre = this->ekf_odom->getState().head(3);
  start = this->now();

  size_t corrected = this->ekf_odom->correct(z, detected_cones);

  end = this->now();
  rclcpp::Duration exe_time = end - start;

  if (this->enable_logging) {
    /* Correction-health diagnostic: how many cones actually passed association
       and got applied (corrected), the resulting pose move (|dpos|, |dyaw|), and
       the current pose covariance. If "corrected" trends to 0 while P climbs and
       |dpos| shrinks over the final laps, the corrections are dropping out (drift
       cascade) rather than the covariance merely ratcheting. */
    Vector3f pose_post = this->ekf_odom->getState().head(3);
    const double dpos = std::hypot((double)(pose_post(0) - pose_pre(0)),
                                   (double)(pose_post(1) - pose_pre(1)));
    const double dyaw = std::fabs(this->ekf_odom->normalizeAngle(pose_post(2) - pose_pre(2)));
    Vector3f pcov = this->ekf_odom->getPoseCovariance();
    RCLCPP_INFO(this->get_logger(),
                "scan: detected=%zu corrected=%zu |dpos|=%.4f |dyaw|=%.4f "
                "Pxx=%.5f Pyy=%.5f Pyaw=%.5f exe_ms=%.3f",
                detected_cones, corrected, dpos, dyaw,
                pcov(0), pcov(1), pcov(2), exe_time.nanoseconds() * 1e-6);
  }

  free(z);

  /* In a DEBUG build, cones_pub_for_debug keeps republishing the live (lap-1)
     map even after the corrected map is frozen, for visual comparison. In a
     normal build it is always false (compiled out). */
  bool republish_for_debug = false;
#ifdef CONE_FUSED_DEBUG
  /* Debug: project the raw input cones into the map frame via the current EKF
     pose and publish them as red markers (input vs. map check). */
  if (this->pub_input_cones_debug)
    this->pubInputConesDebug(cones_data);

  republish_for_debug = this->cones_pub_for_debug;
#endif

  /* Publish cones */
  if (!this->corrected_cones_created || republish_for_debug) {
    size_t mapped_cones = this->ekf_odom->getActMappedLandmarks();
    conesMarker.points.reserve(mapped_cones);
    conesMarker.colors.reserve(mapped_cones);

    /* Snapshot state and signatures once (getters return by value). */
    SignatureVector sigs = this->ekf_odom->getSignatures();
    VectorXf state = this->ekf_odom->getState();

    for (size_t i = 0; i < mapped_cones; i++) {
      /* Get the most-voted color and how many times the i-th cone was seen */
      std::pair<ColorId, uint32_t> cone_info = sigs(i).getConeColorAndCount();

      /* Only map cones seen at least cone_time_seen_th times */
      if (cone_info.second < cone_time_seen_th)
        continue;

      Vector2f cone = state.segment(3 + (i * 2), 2);
      geometry_msgs::msg::Point p;
      std_msgs::msg::ColorRGBA c;
      this->setConeColor(c, static_cast<uint8_t>(cone_info.first));
      p.x = cone(0);
      p.y = cone(1);
      p.z = 0.0;
      conesMarker.points.push_back(p);
      conesMarker.colors.push_back(c);
    }

    if (this->race_status.current_lap > 1 && !this->corrected_cones_created) {
      this->ekf_odom->setFirstLapCompleted(true);
      RCLCPP_INFO(this->get_logger(), "Creating corrected cones markers with %zu cones", conesMarker.points.size());
      this->corrected_cones_created = true;
      this->corrected_cones_size = mapped_cones;
      this->correctedConesMarker = conesMarker;
    }
  }

  /* Publish vehicle pose and cones position */
  if (!this->corrected_cones_created || republish_for_debug) {
    this->pubConesMarkers(conesMarker);
    conesMarker.points.clear();
    conesMarker.colors.clear();
  } else {
    this->pubConesMarkers(correctedConesMarker);
  }
}

void ConeFusion::fastLimoDataCallback(const nav_msgs::msg::Odometry::SharedPtr fast_limo_data) 
{
  // RCLCPP_INFO(this->get_logger(), "FAST LIMO CB");
  tf2::Quaternion q;
  q.setX(fast_limo_data->pose.pose.orientation.x);
  q.setY(fast_limo_data->pose.pose.orientation.y);
  q.setZ(fast_limo_data->pose.pose.orientation.z);
  q.setW(fast_limo_data->pose.pose.orientation.w);
  tf2::Matrix3x3 m(q);

  double r, p, y;
  m.getRPY(r, p, y);

  Vector3f pose;
  pose << fast_limo_data->pose.pose.position.x,
      fast_limo_data->pose.pose.position.y, y;
  Vector3f pose_cov;

  /* Retrieve FAST-LIMO Pose covariance. The covariance of FAST-LIMO is a 6x6. The
     necessary pose covariance is in the main diagonal at indexes 0 (X), 7 (Y),
     35 (Yaw).
  */
  pose_cov(0) = fast_limo_data->pose.covariance.at(0);  /* X covariance */
  pose_cov(1) = fast_limo_data->pose.covariance.at(7);  /* Y covariance */
  pose_cov(2) = fast_limo_data->pose.covariance.at(35); /* Yaw covariance */

  this->ekf_odom->setPose(pose);
  this->ekf_odom->setPoseCovariance(pose_cov);

  /* Publish the EKF state (corrected by the cones), NOT the raw FAST-LIMO pose.
     This is what lets the cone corrections survive on the wire. */
  Vector3f ekf_pose = this->ekf_odom->getState().head(3);

  /* If this is the second lap, retrieve ONLY vehicle position, and publish
   * already mapped cones. */
  if (this->corrected_cones_created || this->is_skidpad_mission) {
    /* Get vehicle pose */
    this->act_position << ekf_pose(0), ekf_pose(1), 0.0;
    this->act_yaw = ekf_pose(2);

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, this->act_yaw);
    this->act_orientation.set__x(q.getX());
    this->act_orientation.set__y(q.getY());
    this->act_orientation.set__z(q.getZ());
    this->act_orientation.set__w(q.getW());

    this->updatePose();
  }else{
    /* If this is the first lap, just publish FAST-LIMO pose and cones position if
     * seen. */
    this->act_position << ekf_pose(0), ekf_pose(1), 0.0;
    this->act_yaw = ekf_pose(2);

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, this->act_yaw);
    this->act_orientation.set__x(q.getX());
    this->act_orientation.set__y(q.getY());
    this->act_orientation.set__z(q.getZ());
    this->act_orientation.set__w(q.getW());

    this->updatePose();
  }
}

void ConeFusion::pubConesMarkers(visualization_msgs::msg::Marker &cones) {
  cones.header.stamp = this->now();
  this->conesPositionsMarkerPub->publish(cones);
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
  dbg.header.stamp = this->now();
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

void ConeFusion::updatePose() {
  geometry_msgs::msg::TransformStamped t;
  nav_msgs::msg::Odometry _odom;

  /* Update header */
  t.header.stamp = this->now();
  t.header.frame_id = this->output_frame_id;
  t.child_frame_id = this->output_child_frame_id;

  /* Update TF location */
  t.transform.translation.x = act_position[0];
  t.transform.translation.y = act_position[1];
  t.transform.translation.z = 0.0f;

  /* Update TF Orientation */
  t.transform.rotation = this->act_orientation;

  _odom.header.stamp = this->now();
  _odom.header.frame_id = this->output_frame_id;
  _odom.child_frame_id = this->output_child_frame_id;

  /* Update Odom position */
  _odom.pose.pose.position.x = act_position[0];
  _odom.pose.pose.position.y = act_position[1];
  _odom.pose.pose.position.z = 0.0f;

  /* Update Odom orientation */
  _odom.pose.pose.orientation = this->act_orientation;

  /* Update Odom covariance (just the main diagonal). Floor each variance at a
     tiny positive value: the EKF covariance update P -= K(HP) is done in float,
     so when strong cone corrections shrink the pose covariance, round-off can
     push a diagonal entry slightly negative. A negative variance makes the
     published covariance non-PSD (RViz: "Negative eigenvalue found for
     position"). Flooring keeps what we publish valid without touching the
     filter. */
  constexpr double COV_FLOOR = 1e-9;
  Vector3f pose_cov = this->ekf_odom->getPoseCovariance();

  _odom.pose.covariance.at(0) = std::max((double)pose_cov(0), COV_FLOOR);  /* X Covariance */
  _odom.pose.covariance.at(7) = std::max((double)pose_cov(1), COV_FLOOR);  /* Y Covariance */
  _odom.pose.covariance.at(35) = std::max((double)pose_cov(2), COV_FLOOR); /* Yaw Covariance */

  this->tf_broadcaster_->sendTransform(t);
  this->odom_pub->publish(_odom);
}

void ConeFusion::raceStatusCallback(
    const mmr_base::msg::RaceStatus::SharedPtr race_status_data) {
  /* Update race status */
  this->race_status = *race_status_data;
}