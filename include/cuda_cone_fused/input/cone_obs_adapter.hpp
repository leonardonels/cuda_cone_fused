#pragma once

#include <cuda_cone_fused/color_logic.hpp>
#include <cuda_cone_fused/ekf_types.hpp>
#include <cuda_cone_fused/input/color_classifier.hpp>

#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

/**
 * Inbound adapter: cones `visualization_msgs/Marker` (transient
 * vehicle-frame observations) -> sensor-agnostic range/bearing/signature
 * Observations consumed by EKFOdom::update().
 *
 * This is the input side of "cones": transient observations. It shares only the
 * word "cones" with the OUTPUT side (ConeMapPublisher, the accumulated global
 * map) — they never touch. ROS-msg translation lives here (vehicle-frame ->
 * polar), while the colour -> signature mapping is delegated to an injected
 * IColorClassifier Strategy (chosen once by the factory).
 */
class ConeObsAdapter {
public:
  explicit ConeObsAdapter(std::unique_ptr<IColorClassifier> classifier)
      : classifier_(std::move(classifier)) {}

  std::vector<Observation> convert(const visualization_msgs::msg::Marker& msg) const {
    const uint64_t stamp_ns =
        static_cast<uint64_t>(msg.header.stamp.sec) * 1000000000ull +
        static_cast<uint64_t>(msg.header.stamp.nanosec);

    std::vector<Observation> obs;
    obs.reserve(msg.points.size());
    for (size_t i = 0; i < msg.points.size(); ++i) {
      Observation o;
      o.z(0) = std::sqrt(msg.points[i].x * msg.points[i].x +
                         msg.points[i].y * msg.points[i].y);  // + coneRadius;
      o.z(1) = normalizeAngle(std::atan2(msg.points[i].y, msg.points[i].x));
      /* Guard the parallel colours array: a colour-agnostic classifier ignores
         it, so a scan may legitimately omit colours. */
      const std_msgs::msg::ColorRGBA color =
          (i < msg.colors.size()) ? msg.colors[i] : std_msgs::msg::ColorRGBA();
      o.z(2) = static_cast<float>(classifier_->classify(color));
      o.stamp_ns = stamp_ns;
      obs.push_back(o);
    }
    return obs;
  }

private:
  static float normalizeAngle(float angle) {
    while (angle >  M_PI) angle -= 2.0f * static_cast<float>(M_PI);
    while (angle < -M_PI) angle += 2.0f * static_cast<float>(M_PI);
    return angle;
  }

  std::unique_ptr<IColorClassifier> classifier_;
};
