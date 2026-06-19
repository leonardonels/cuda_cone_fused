#pragma once

#include <cstddef>
#include <memory>

/**
 * EKF backend Bridge.
 *
 * Decouples the EKFOdom abstraction from the platform that runs the heavy linear
 * algebra. Two peers implement it:
 *   - CudaBackend: resident GPU state, the device holds the
 *     authoritative dim x dim P and dim x_; only tiny slices cross the bus.
 *   - CpuBackend: the former host fallback promoted to a first-class peer; holds
 *     P and x in Eigen on the host.
 *
 * In BOTH cases the backend owns the authoritative state. EKFOdom keeps only
 * small host mirrors (pose, active landmark means, the 3x3 pose block) refreshed
 * via the download* methods, and runs ONE update code path that dispatches the
 * O(na^2 m) joint solve to batchUpdateFullState() — no more #ifdef braiding.
 *
 * Buffer conventions (identical for both peers, so the same host buffers feed
 * either): matrices are COLUMN-MAJOR contiguous (Eigen default); G3x3 / P33 are
 * ROW-MAJOR 3x3.
 */
class IEkfBackend {
public:
  virtual ~IEkfBackend() = default;

  /** True if the backend constructed and all setup succeeded. */
  virtual bool ok() const = 0;
  /** Human-readable last-error string (empty if none). */
  virtual const char* lastError() const = 0;

  /* ---- hot path -------------------------------------------------------- */

  /**
   * Full-state joint EKF update on the active na x na block of P and the active
   * head of x: HP = H P; S = HP H^T + R; solve S K^T = HP; x += K nu; P -= K HP.
   * Does NOT wrap yaw (EKFOdom wraps after refreshing its mirror).
   *
   * @param na     active dimension (3 + 2*landmark_count).
   * @param H      [in] host, (2m)*na col-major stacked Jacobian.
   * @param R      [in] host, (2m)*(2m) col-major block-diag measurement noise.
   * @param nu     [in] host, 2m stacked (wrapped) innovation.
   * @param two_m  2*m.
   * @return false on backend failure (state left unchanged).
   */
  virtual bool batchUpdateFullState(int na, const float* H, const float* R,
                                    const float* nu, int two_m) = 0;

  /* ---- predict / motion ----------------------------------------------- */

  /**
   * Predict-step covariance propagation + pose-mean composition:
   *   P[0:3,0:na] = G P[0:3,0:na];  P[0:na,0:3] = P[0:na,0:3] G^T
   *   P(0,0)+=add_xx; P(1,1)+=add_yy; P(2,2)+=add_yaw
   *   x(0)+=dwx; x(1)+=dwy; x(2)=wrap_2pi(x(2)+dtheta)
   * @param G3x3 row-major 3x3 motion Jacobian.
   */
  virtual void motionPropagate(int na, const float G3x3[9],
                               float dwx, float dwy, float dtheta,
                               float add_xx, float add_yy, float add_yaw) = 0;

  /** Add increments to the pose covariance diagonal (setPoseCovariance). */
  virtual void addPoseCovDiag(float dxx, float dyy, float dyaw) = 0;

  /* ---- structure / IO -------------------------------------------------- */

  /** Lap-1 new cone: write landmark k mean (x[3+2k], x[3+2k+1]). */
  virtual void insertLandmark(int k, float mx, float my) = 0;

  virtual void downloadPose3(float pose3[3]) const = 0;
  virtual void downloadPoseCov3(float cov3[3]) const = 0;
  virtual void downloadPoseBlock3x3(float P33[9]) const = 0;
  virtual void downloadLandmarks(float* xy, int count) const = 0;
  virtual void uploadPose3(const float pose3[3]) = 0;
  virtual void uploadPoseBlock3x3(const float P33[9]) = 0;

  /* ---- debug / validation (full dim x dim transfers) ------------------- */
  virtual void debugSetState(const float* P, const float* x) = 0;
  virtual void debugGetState(float* P, float* x) const = 0;
};

/**
 * Factory: builds the concrete backend chosen once at startup.
 * Prefers the CUDA peer when the build enables it and the device comes up;
 * otherwise (or on CUDA init failure) returns the CPU peer. This is the SINGLE
 * place where USE_CUDA is still consulted.
 *
 * @param dim       full state dimension (= 3 + 2*N_CONES).
 * @param max_two_m largest 2*m expected per scan (GPU scratch hint; ignored CPU).
 * @param inf_init  initial variance on the inactive landmark diagonal.
 */
std::unique_ptr<IEkfBackend> makeEkfBackend(int dim, int max_two_m, float inf_init);
