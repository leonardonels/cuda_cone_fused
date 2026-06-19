#pragma once

#include <cuda_cone_fused/ekf_odom.hpp>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <cstdint>
#include <string>

/**
 * Outbound publisher / presenter: translates the accumulated
 * global cone map (landmark means + voted signatures) -> a
 * `visualization_msgs/Marker`. Owned by (triggered by) the cones input, the MAIN
 * correction source.
 *
 * This is the OUTPUT side of "cones": the accumulated map. It shares only the
 * word "cones" with ConeObsAdapter (the transient input observations).
 *
 * The lap-freeze / corrected-cones latch is a PUBLISHING policy and lives here,
 * off the node and out of the filter: once the first lap completes it snapshots
 * the map, flips the filter into rigid localization (setFirstLapCompleted), and
 * from then on republishes the frozen map.
 *
 * Stamps output with the SOURCE measurement timestamp, never now().
 */
class ConeMapPublisher {
public:
  ConeMapPublisher(rclcpp::Node* node, const std::string& topic,
                   std::string frame_id, uint32_t cone_time_seen_th,
                   bool republish_live_for_debug = false);

  /**
   * @param filter        the EKF (read state/signatures; freeze on lap rollover).
   * @param stamp         source measurement timestamp for the marker header.
   * @param current_lap   from race status; drives the freeze latch.
   */
  void publish(EKFOdom& filter, const rclcpp::Time& stamp, uint32_t current_lap);

private:
  void initMarker(visualization_msgs::msg::Marker& m) const;
  static void setConeColor(std_msgs::msg::ColorRGBA& color, uint8_t color_id);

  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_;
  std::string frame_id_;
  uint32_t cone_time_seen_th_;
  bool republish_live_for_debug_;

  visualization_msgs::msg::Marker live_marker_;       /* rebuilt each scan during mapping */
  visualization_msgs::msg::Marker corrected_marker_;  /* latched snapshot, republished after freeze */
  bool corrected_created_ = false;
  size_t corrected_size_ = 0;
};
