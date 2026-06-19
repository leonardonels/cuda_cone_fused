#include <cuda_cone_fused/output/cone_map_publisher.hpp>

#include <geometry_msgs/msg/point.hpp>

ConeMapPublisher::ConeMapPublisher(rclcpp::Node* node, const std::string& topic,
                                   std::string frame_id, uint32_t cone_time_seen_th,
                                   bool republish_live_for_debug)
    : frame_id_(std::move(frame_id)),
      cone_time_seen_th_(cone_time_seen_th),
      republish_live_for_debug_(republish_live_for_debug) {
  pub_ = node->create_publisher<visualization_msgs::msg::Marker>(topic, 1);
  initMarker(live_marker_);
  initMarker(corrected_marker_);
}

void ConeMapPublisher::initMarker(visualization_msgs::msg::Marker& m) const {
  m.header.frame_id = frame_id_;
  m.ns = "ConesAbsolutePos";
  m.id = 0;
  m.type = visualization_msgs::msg::Marker::SPHERE_LIST;
  m.action = visualization_msgs::msg::Marker::ADD;
  m.scale.x = 0.35;
  m.scale.y = 0.35;
  m.scale.z = 0.35;
  m.pose.position.x = 0.0;
  m.pose.position.y = 0.0;
  m.pose.position.z = 0.0;
  m.pose.orientation.w = 1.0;
  m.pose.orientation.x = 0.0;
  m.pose.orientation.y = 0.0;
  m.pose.orientation.z = 0.0;
}

void ConeMapPublisher::setConeColor(std_msgs::msg::ColorRGBA& color, uint8_t color_id) {
  switch (color_id) {
  case blueCone:
    color.r = 0.0f; color.g = 0.0f; color.b = 1.0f; color.a = 1.0f;
    break;
  case yellowCone:
    color.r = 1.0f; color.g = 1.0f; color.b = 0.0f; color.a = 1.0f;
    break;
  case orangeCone:
    color.r = 1.0f; color.g = 0.3f; color.b = 0.0f; color.a = 1.0f;
    break;
  case orangeBigCone:
    color.r = 1.0f; color.g = 0.63f; color.b = 0.1f; color.a = 1.0f;
    break;
  }
}

void ConeMapPublisher::publish(EKFOdom& filter, const rclcpp::Time& stamp,
                               uint32_t current_lap) {
  /* While still mapping (or when the debug live-republish is on), rebuild the
     live marker from the current filter state + voted signatures. */
  if (!corrected_created_ || republish_live_for_debug_) {
    const size_t mapped_cones = filter.getActMappedLandmarks();

    live_marker_.points.clear();
    live_marker_.colors.clear();
    live_marker_.points.reserve(mapped_cones);
    live_marker_.colors.reserve(mapped_cones);

    /* Snapshot state and signatures once (getters return by value). */
    SignatureVector sigs = filter.getSignatures();
    VectorXf state = filter.getState();

    for (size_t i = 0; i < mapped_cones; i++) {
      /* Most-voted colour + how many times the i-th cone was seen. */
      std::pair<ColorId, uint32_t> cone_info = sigs(i).getConeColorAndCount();

      /* Only map cones seen at least cone_time_seen_th_ times. */
      if (cone_info.second < cone_time_seen_th_)
        continue;

      Vector2f cone = state.segment(3 + (i * 2), 2);
      geometry_msgs::msg::Point p;
      std_msgs::msg::ColorRGBA c;
      setConeColor(c, static_cast<uint8_t>(cone_info.first));
      p.x = cone(0);
      p.y = cone(1);
      p.z = 0.0;
      live_marker_.points.push_back(p);
      live_marker_.colors.push_back(c);
    }

    /* Lap-freeze latch: once past lap 1, flip the filter into rigid
       localization and snapshot the map to republish from now on. */
    if (current_lap > 1 && !corrected_created_) {
      filter.setFirstLapCompleted(true);
      RCLCPP_INFO(rclcpp::get_logger("cone_map_publisher"),
                  "Creating corrected cones markers with %zu cones",
                  live_marker_.points.size());
      corrected_created_ = true;
      corrected_size_ = mapped_cones;
      corrected_marker_ = live_marker_;
    }
  }

  /* Publish the live map while mapping (or for debug), else the frozen map. */
  visualization_msgs::msg::Marker& out =
      (!corrected_created_ || republish_live_for_debug_) ? live_marker_
                                                         : corrected_marker_;
  out.header.stamp = stamp;
  pub_->publish(out);
}
