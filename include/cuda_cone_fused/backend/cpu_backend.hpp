#pragma once

#include <cuda_cone_fused/backend/ekf_backend.hpp>

#include <Eigen/Dense>
#include <string>

/**
 * Host (Eigen) EKF backend — the former CPU fallback promoted to a first-class
 * peer of the CUDA backend. Holds the authoritative full state:
 *   x_ : dim                    (pose | landmark means)
 *   P_ : dim x dim covariance
 * and runs every operation with Eigen. Same joint-update math the GPU peer runs
 * on the device, so the two are interchangeable behind IEkfBackend and the CPU
 * build is no longer a separate single-cone code path.
 */
class CpuBackend : public IEkfBackend {
public:
  /**
   * @param dim      full state dimension (= 3 + 2*N_CONES).
   * @param inf_init initial variance on the inactive landmark diagonal.
   * Initialises P = blockdiag(I_3, inf_init * I_2N) and x = 0, exactly like the
   * CUDA peer's constructor.
   */
  CpuBackend(int dim, float inf_init);
  ~CpuBackend() override = default;

  bool ok() const override { return good_; }
  const char* lastError() const override { return err_.c_str(); }

  bool batchUpdateFullState(int na, const float* H, const float* R,
                            const float* nu, int two_m) override;

  void motionPropagate(int na, const float G3x3[9],
                       float dwx, float dwy, float dtheta,
                       float add_xx, float add_yy, float add_yaw) override;
  void addPoseCovDiag(float dxx, float dyy, float dyaw) override;

  void insertLandmark(int k, float mx, float my) override;

  void downloadPose3(float pose3[3]) const override;
  void downloadPoseCov3(float cov3[3]) const override;
  void downloadPoseBlock3x3(float P33[9]) const override;
  void downloadLandmarks(float* xy, int count) const override;
  void uploadPose3(const float pose3[3]) override;
  void uploadPoseBlock3x3(const float P33[9]) override;

  void debugSetState(const float* P, const float* x) override;
  void debugGetState(float* P, float* x) const override;

private:
  int dim_ = 0;
  bool good_ = false;
  std::string err_;

  Eigen::VectorXf x_;   /* dim */
  Eigen::MatrixXf P_;   /* dim x dim */
};
