#pragma once

#include <Eigen/Dense>

#include "color_logic.hpp"
#include <cuda_cone_fused/backend/ekf_backend.hpp>
#include <cuda_cone_fused/ekf_types.hpp>

#include <iostream>
#include <memory>
#include <vector>

#define N_CONES 400           
/* New-landmark prior variance ("uninformative" init).
   The float32 covariance update P -= K(HP) computes INF - (~INF) for a fresh
   landmark, so a huge INF (e.g. 1e9, ULP ~128 in float32) is swallowed by
   catastrophic cancellation -> negative/garbage variance -> indefinite P ->
   NaN in the resident GPU state on the very first scan. 1e1 (~3.2 m std) is
   uninformative enough yet numerically safe. */
#define INF 1e1

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

    /* LAP-1 new-vs-existing gate, P-SCALED. The lap-1 radius used to be the
       fixed max_new_cone_dist, which is the one gate in the filter that does NOT
       widen with uncertainty — and lap 1 is the lap that BUILDS the map. Once
       the pose error exceeded that radius the filter concluded "new cone"
       instead of "I am uncertain": the duplicate went in with INF covariance
       (contributing no correction) and the real landmark that should have
       corrected the pose was never used, so the correction was withheld exactly
       when it was needed most and the drift ran away on its own. The effective
       radius is now
           r = clamp(max_new_cone_dist + k * sigma_pose, ..., new_cone_dist_cap_)
       with sigma_pose = sqrt(max(Pxx, Pyy)), so it widens with the pose
       uncertainty the same way the lap-2+ Mahalanobis gate does. The cap keeps
       it below the physical cone spacing so a diverged P can never merge two
       distinct cones. (A full Mahalanobis gate is not used here: a fresh
       landmark carries INF covariance and the host mirror only holds the pose
       block, so the pose-only S would be optimistic exactly where the map is
       still immature.) */
    float new_cone_sigma_scale_ = 3.0f; /* k: pose-sigma multiplier added to the lap-1 radius. 0 = fixed radius (old behaviour). */
    float new_cone_dist_cap_ = 4.0f;    /* absolute cap [m] on the widened lap-1 radius; must stay below the real cone spacing. */

    float q_motion_pos_ = 0.0f; /* Additive process noise on x,y per metre travelled [m^2/m]. Keeps P from collapsing so the assoc gate stays honest. 0 = off. */
    float q_motion_yaw_ = 0.0f; /* Additive process noise on theta per radian turned [rad^2/rad]. 0 = off. */

public:
    /* Per-scan data-association diagnostics, refreshed by every update().
       Read by the ROS node for the correction-health log; the filter itself is
       ROS-free, so it counts here and logs nowhere. */
    struct AssocStats
    {
        size_t observed = 0;      /* observations considered (after the orange-cone filter) */
        size_t associated = 0;    /* passed association and entered the batch update */
        size_t created = 0;       /* new landmarks allocated this scan (lap 1 only) */
        size_t dropped_gate = 0;  /* lap 2+: failed the Mahalanobis gate */
        size_t dropped_full = 0;  /* dropped because the map hit N_CONES (the cliff) */
        float nn_max = 0.0f;      /* max nearest-neighbour distance [m] over this scan */
        float nn_mean = 0.0f;     /* mean nearest-neighbour distance [m] over this scan */
        float gate_radius = 0.0f; /* lap-1 effective new-cone radius [m] used this scan */

        /* Nearest-neighbour distance AT THE MOMENT OF CREATION, over the
           landmarks created this scan (only counted once the map is non-empty,
           so the very first cones do not register a meaningless FLT_MAX).
           This is what separates a duplicate from a genuinely new cone: a real
           new cone appears at the physical cone spacing from anything already
           mapped (>= ~3 m), a duplicate appears just outside the gate radius.
           The histogram of these over a run is what sets the right radius. */
        float new_nn_min = 0.0f;
        float new_nn_max = 0.0f;
    };

private:
    AssocStats assoc_stats_;

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
    void setNewConeGate(const float sigma_scale, const float dist_cap);

    /* Data-association diagnostics from the most recent update(). */
    const AssocStats& getAssocStats() const { return this->assoc_stats_; }

    /* Landmark capacity, so callers can report headroom against the cliff. */
    static constexpr size_t landmarkCapacity() { return N_CONES; }


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
