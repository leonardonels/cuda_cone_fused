#pragma once

#include <Eigen/Dense>

#include "color_logic.hpp"

#include <iostream>

#ifdef USE_CUDA
#include <cuda_cone_fused/ekf_cuda.hpp>
#include <memory>
#endif

#define N_CONES 400           /* This constant is the initial cones number. */
#define INF 1e1       /* 1e10 Constant value to represent Infinite */

using namespace Eigen;

/**
 *  type defining the signature vector. A signature is way of indexing a particular landmark. In our use case
 * the landmark (cones) are identified by the INDEX in the state vector. This further vector is used to identify
 * the COLOR of the cone which is extremely important. Each element in this vector refers to an already mapped cone in the 
 * state vector with the same index
*/
typedef Eigen::Matrix<ColorLogic, N_CONES, 1> SignatureVector;

class EKFOdom
{
private:
    Eigen::VectorXf x_;     /* Robot state vector: [x, y, theta | m_0x, m_0y, ...] */
    Eigen::MatrixXf P_;     /* State covariance matrix */
    Eigen::Matrix2f Q_;     /* (range, bearing) measurement-noise covariance, added to the innovation covariance S in correct() */
    SignatureVector s_;     /* Vector containing the signature for each landmark. In our case the signature is the COLOR of the cone. The ID for each color are defined in hedaer file: "color_logic.hpp" . */

    size_t landmark_count = 0;  /* Counter of mapped landmarks */

    float max_new_cone_dist;   /* Max distance for creating a new cone */

    bool is_pose_initialized = false;  /* Bool to check if an initial pose has been set */

    Eigen::Vector3f prev_limo_pose_ = Eigen::Vector3f::Zero();  /* Last FAST-LIMO pose, used to compute the relative motion increment in setPose */

    Eigen::Vector3f prev_limo_cov_ = Eigen::Vector3f::Zero();   /* Last FAST-LIMO pose covariance, used to compute the covariance increment in setPoseCovariance */
    bool is_cov_initialized = false;  /* Bool to check if the first FAST-LIMO covariance has been recorded */

    bool is_first_lap_completed = false; /* Bool to understand if the first lap is completed. In that case, do not map any new cone */

    /* Cone-update regime is fixed at build time: a GPU build (USE_CUDA) always
       fuses ALL cones of a scan in one joint EKF update (and the single-cone
       path is compiled out); a CPU-only build always does the single-cone
       update. Lap 1 always runs full EKF-SLAM (cones correct the pose too) —
       FAST-LIMO is never trusted blindly while mapping. */

    bool freeze_map_ = true; /* If true (default), lap 2+ does a rigid-map pose-only update (landmarks fixed, gauge locked). If false, lap 2+ keeps doing a full-state SLAM update (pose AND landmarks corrected continuously). */

    float assoc_maha_gate_ = 9.21f; /* chi-square (2 DOF) gate for lap-2+ data association by Mahalanobis distance */

    float q_motion_pos_ = 0.0f; /* Additive process noise on x,y per metre travelled [m^2/m]. Keeps P from collapsing so the assoc gate stays honest. 0 = off. */
    float q_motion_yaw_ = 0.0f; /* Additive process noise on theta per radian turned [rad^2/rad]. 0 = off. */

#ifdef USE_CUDA
    /* Resident-state GPU backend. When active, the device holds the
       authoritative P_ and x_; the host x_ keeps a synced mirror of the active
       head (pose + active landmarks) and P_ keeps a synced 3x3 pose block — the
       only parts the CPU touches (data association, markers, publishing). */
    std::unique_ptr<EkfCudaBackend> gpu_;
    bool use_gpu_ = false;

    /* Refresh the host mirrors (x_ head(na) and P_ 3x3) from the device. */
    void syncFromDevice();
#endif

public:
    EKFOdom(Vector2f process_noise, Vector2f motion_noise, const float alpha, int eigen_threads = 1);
    virtual ~EKFOdom();

    /**
     * Correct filter state using measurement(s) z:
     * 
     * @param z: Measurement(s) of actual cones. This vector contains:
     *  - actual range from vehicle to the landmark;
     *  - actual bearing from vehicle to the landmark;
     *  - signature of the landmark;
     *  @param act_cones_detected: Number of actually observed landmarks
     *  @return number of observations that passed data association (and thus
     *          contributed to a correction) this scan — diagnostic.
    */
    size_t correct(const Vector3f *z, const size_t act_cones_detected);
    
    VectorXf getState() const;
    Vector3f getPoseCovariance() const;
    size_t getActMappedLandmarks() const;
    SignatureVector getSignatures() const;

    void setFirstLapCompleted(const bool first_lap_completed);
    void setFreezeMap(const bool enable);
    void setAssocMahaGate(const float gate);
    void setPose(const Vector3f pose);
    void setPoseCovariance(const Vector3f pos_cov);


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