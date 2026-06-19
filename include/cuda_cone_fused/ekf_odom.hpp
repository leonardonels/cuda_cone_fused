#pragma once

#include <Eigen/Dense>

#include "color_logic.hpp"
#include <cuda_cone_fused/backend/ekf_backend.hpp>
#include <cuda_cone_fused/ekf_types.hpp>

#include <iostream>
#include <memory>
#include <vector>

#define N_CONES 400           
#define INF 1e9       

using namespace Eigen;

/**
 *  type defining the signature vector. A signature is way of indexing a particular landmark. In our use case
 * the landmark (cones) are identified by the INDEX in the state vector. This further vector is used to identify
 * the COLOR of the cone which is extremely important. Each element in this vector refers to an already mapped cone in the
 * state vector with the same index
*/
typedef Eigen::Matrix<ColorLogic, N_CONES, 1> SignatureVector;

/**
 * EKF-SLAM filter core. ROS-free on BOTH boundaries: it consumes
 * sensor-agnostic MotionIncrement / Observation and exposes plain state getters.
 *
 * The authoritative state (full x and dim x dim P) lives in an IEkfBackend
 * (CPU or CUDA peer, chosen once at startup by makeEkfBackend). The filter keeps
 * only small host MIRRORS — the pose, the active landmark means (x_), and the
 * 3x3 pose covariance block (P_pose_) — which it uses for data association,
 * Jacobian assembly and publishing, refreshed via syncFromBackend() after each
 * state-mutating backend call.
 */
class EKFOdom
{
private:
    /* ---- host mirrors of the backend-owned authoritative state ---------- */
    Eigen::VectorXf x_;       /* mirror of [x, y, theta | m_0x, m_0y, ...]; active head (3 + 2*landmark_count) kept synced */
    Eigen::Matrix3f P_pose_;  /* mirror of the 3x3 pose covariance block (used by the Mahalanobis gate + publishing) */

    Eigen::Matrix2f R_;     /* (range, bearing) measurement-noise covariance, added to the innovation covariance S in update() */
    SignatureVector s_;     /* Vector containing the signature for each landmark. In our case the signature is the COLOR of the cone. The ID for each color are defined in hedaer file: "color_logic.hpp" . */

    size_t landmark_count = 0;  /* Counter of mapped landmarks */

    float max_new_cone_dist;   /* Max distance for creating a new cone */

    bool is_first_lap_completed = false; /* Bool to understand if the first lap is completed. In that case, do not map any new cone */

    bool freeze_map_ = true; /* If true (default), lap 2+ does a rigid-map pose-only update (landmarks fixed, gauge locked). If false, lap 2+ keeps doing a full-state SLAM update (pose AND landmarks corrected continuously). */

    float assoc_maha_gate_ = 9.21f; /* chi-square (2 DOF) gate for lap-2+ data association by Mahalanobis distance */

    float q_motion_pos_ = 0.0f; /* Additive process noise on x,y per metre travelled [m^2/m]. Keeps P from collapsing so the assoc gate stays honest. 0 = off. */
    float q_motion_yaw_ = 0.0f; /* Additive process noise on theta per radian turned [rad^2/rad]. 0 = off. */

    /* Backend Bridge: owns the authoritative x and P (resident on the GPU for
       the CUDA peer, host Eigen for the CPU peer). The heavy joint update is
       dispatched here; there is a single code path regardless of platform. */
    std::unique_ptr<IEkfBackend> backend_;

    /* Refresh the host mirrors (x_ active head + P_pose_) from the backend. */
    void syncFromBackend();

public:
    EKFOdom(Vector2f meas_noise, Vector2f proc_noise, const float alpha, int eigen_threads = 1);
    virtual ~EKFOdom();

    /**
     * Predict step: absorb a sensor-agnostic motion increment.
     * Rotates the local increment into the EKF frame by the current heading,
     * propagates the pose covariance through the motion Jacobian, injects the
     * motion process noise (scaled with travelled distance / turn) and the
     * adapter-supplied covariance increment, and composes the mean. Replaces the
     * former setPose() + setPoseCovariance().
     */
    void predict(const MotionIncrement& inc);

    /**
     * Correction step: fuse a scan of sensor-agnostic observations.
     * Each Observation carries (range, bearing, signature). Returns the number
     * of observations that passed data association (and thus contributed to a
     * correction) this scan — diagnostic. Replaces the former correct().
     */
    size_t update(const std::vector<Observation>& obs);

    VectorXf getState() const;
    Vector3f getPoseCovariance() const;
    size_t getActMappedLandmarks() const;
    SignatureVector getSignatures() const;

    void setFirstLapCompleted(const bool first_lap_completed);
    void setFreezeMap(const bool enable);
    void setAssocMahaGate(const float gate);


    inline float euclideanDistance(float x1, float x2, float y1, float y2) {
      return sqrt(((x2 - x1)*(x2 - x1)) + ((y2 - y1)*(y2 - y1)));
    }

    inline float normalizeAngle(float angle) {
      while(angle > M_PI)
        angle -= 2*M_PI;
      while(angle < -M_PI)
        angle += 2*M_PI;

      return angle;
    }

    inline float normalizeYaw(float yaw) {
      while(yaw > (2*M_PI))
      {
        yaw -= (2*M_PI);
      }
      while(yaw < -(2*M_PI))
      {
        yaw += (2*M_PI);
      }

      return yaw;
    }



};
