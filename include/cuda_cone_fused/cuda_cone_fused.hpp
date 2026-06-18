#pragma once

#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker.hpp>

#include <cuda_cone_fused/ekf_odom.hpp>

#include <mmr_base/msg/race_status.hpp>

#include <memory>
#include <string>

class ConeFusion : public rclcpp::Node {
private:
  /* color logic */
  bool is_colorblind;

  /* Subscriptions topics' names */
  std::string cones_topic, imu_topic, input_odom_topic, race_status_topic,
      mapped_cones_topic, output_odom_topic,
      output_frame_id, output_child_frame_id;
      // gps_speed_topic, gps_data_topic,

#ifdef CONE_FUSED_DEBUG
  /* Debug topic name and parameters — only compiled in a -DDEBUG=ON build. */
  std::string input_cones_debug_topic;

  /* If true, keep republishing the live map even after the corrected map is
     frozen (raw vs. corrected map comparison). */
  bool cones_pub_for_debug;

  /* If true, publish the raw input cones projected into the map frame via the
     current EKF pose (red markers) for an input-vs-map visual check */
  bool pub_input_cones_debug = false;
#endif

  /* Enable logging parameter */
  bool enable_logging;

  /* If SKIDPAD mission is selected, the EKF can publish FAST_LIMO odometry even
   * if no cones are seen */
  bool is_skidpad_mission = false;

  /* Bool to understand if this is the first topic [Used for initializing the
   * filter]*/
  bool is_first_gps_topic = true;

  /* Actual Odometry position and rotation */
  Vector3d act_position;
  double act_yaw =
      0.0; /* Defined for avoiding to access vector state to much times. */
  geometry_msgs::msg::Quaternion act_orientation;

  /* Mapped cones */
  visualization_msgs::msg::Marker conesMarker;

  /* Final and corrected cones */
  visualization_msgs::msg::Marker correctedConesMarker;
  size_t corrected_cones_size = 0;
  bool corrected_cones_created = false;

  Vector2d act_ll, prev_ll;

  /* Actual Race status */
  mmr_base::msg::RaceStatus race_status;

  /* Total number of cones mapped after the first lap */
  uint16_t tot_mapped_cones = 0;

  /* Min number of time a cone has to be seen in order to map it */
  uint32_t cone_time_seen_th;

  /* If true (default), lap 2+ freezes the map (rigid pose-only localization); if
     false, lap 2+ keeps refining pose AND landmarks (continuous SLAM, legacy) */
  bool freeze_map = true;

  /* Chi-square (2 DOF) gate for lap-2+ data association by Mahalanobis distance */
  double assoc_maha_gate = 9.21;

  /* Eigen/OpenMP thread count for the CPU linear algebra (default 1). */
  int eigen_threads = 1;

  /* EKF Parameters */
  Vector2f proc_noise;
  double min_new_cone_distance;

  /* Motion process noise: [x,y per metre travelled, theta per radian turned].
     Keeps the pose covariance from collapsing so the assoc gate stays honest. */
  Vector2f motion_noise;

  /* EKF SLAM filter object */
  std::shared_ptr<EKFOdom> ekf_odom;

  /* TF2 broadcaster */
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  /* ROS 2 Publishers */
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr conesPositionsMarkerPub;
#ifdef CONE_FUSED_DEBUG
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr inputConesDebugPub;
#endif

  /* ROS 2 Subscribers */
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr fast_limo_odom_sub;
  // rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_odom_sub;  // lidar_imu??
  rclcpp::Subscription<visualization_msgs::msg::Marker>::SharedPtr cones_sub;
  rclcpp::Subscription<mmr_base::msg::RaceStatus>::SharedPtr race_status_sub;

  /* Subscriptions Callbacks */
  void conesCallback(const visualization_msgs::msg::Marker::SharedPtr cones_data);
  // void imuDataCallback(const sensor_msgs::msg::Imu::SharedPtr imu_data);             // lidar_imu??
  void fastLimoDataCallback(const nav_msgs::msg::Odometry::SharedPtr fast_limo_data);
  void raceStatusCallback(const mmr_base::msg::RaceStatus::SharedPtr race_status_data);

  /* Load node parameters */
  void loadParameters();

  /* Init mapped cones */
  void initConesMarker(visualization_msgs::msg::Marker &cones);
  /* Method for publishing cones markers */
  void pubConesMarkers(visualization_msgs::msg::Marker &cones);

#ifdef CONE_FUSED_DEBUG
  /* Debug: project raw input cones into the map frame (current EKF pose) and
     publish them as red markers for an input-vs-map visual comparison */
  void pubInputConesDebug(const visualization_msgs::msg::Marker::SharedPtr &cones_data);
#endif

  /* Method for updating vehicle pose */
  void updatePose();

  uint8_t getConeColor(const std_msgs::msg::ColorRGBA &color) {
    if (color.r < 0.1f && color.g < 0.1f && color.b > 0.9f) {
      // Blue cone recognized
      return blueCone;
    } else if (color.r > 0.9f && color.g > 0.9f && color.b < 0.1f) {
      // Yellow cone recognized
      return yellowCone;
    } else if (color.r > 0.9f && color.g > 0.3f && color.b < 0.1f) {
      // Orange small cone recognized
      // green: 0.31 small orange
      return orangeCone;
    } else if (color.r > 0.9f && color.g > 0.6f && color.b < 0.1f) {
      // Orange Big cone recognized
      // green: 0.63 big orange
      return orangeBigCone;
    }
    return 255;
  }

  void setConeColor(std_msgs::msg::ColorRGBA &color, const uint8_t color_id) {
    switch (color_id) {
    case blueCone:
      color.r = 0.0f;
      color.g = 0.0f;
      color.b = 1.0f;
      color.a = 1.0f;
      break;
    case yellowCone:
      color.r = 1.0f;
      color.g = 1.0f;
      color.b = 0.0f;
      color.a = 1.0f;
      break;
    case orangeCone:
      color.r = 1.0f;
      color.g = 0.3f;
      color.b = 0.0f;
      color.a = 1.0f;
      break;
    case orangeBigCone:
      color.r = 1.0f;
      color.g = 0.63f;
      color.b = 0.1f;
      color.a = 1.0f;
      break;
    }
  }

public:
  /* Constructor */
  ConeFusion();
};