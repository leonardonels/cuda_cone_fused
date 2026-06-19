#pragma once

#include <Eigen/Dense>
#include <cstdint>

/**
 * Sensor-agnostic filter I/O types (ROS-free — invariant 1).
 *
 * These are the only things crossing the EKFOdom boundary. An inbound adapter
 * (e.g. LimoMotionAdapter, ConeObsAdapter) translates a ROS message into one of
 * these; the filter never sees a ROS header. The stamp is a plain nanosecond
 * count so the (future) timestamp-ordered drain stays ROS-free too.
 */
struct MotionIncrement {              /* control input -> predict() */
  Eigen::Vector3f delta_pose = Eigen::Vector3f::Zero();      /* local dx, dy, dtheta */
  Eigen::Vector3f cov_increment = Eigen::Vector3f::Zero();   /* pose-cov increment, MUST be >= 0 and grow with motion */
  uint64_t stamp_ns = 0;
};

struct Observation {                  /* measurement -> update() */
  Eigen::Vector3f z = Eigen::Vector3f::Zero();               /* range, bearing, signature */
  uint64_t stamp_ns = 0;
};
