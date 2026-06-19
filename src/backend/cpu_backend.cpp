#include <cuda_cone_fused/backend/cpu_backend.hpp>

using namespace Eigen;

CpuBackend::CpuBackend(int dim, float inf_init) : dim_(dim) {
  /* Mirror the CUDA peer / former host init: P = blockdiag(I3, inf*I), x = 0. */
  x_ = VectorXf::Zero(dim);
  P_ = MatrixXf::Zero(dim, dim);
  P_.block(0, 0, 3, 3) = Matrix3f::Identity();
  P_.bottomRightCorner(dim - 3, dim - 3) =
      inf_init * MatrixXf::Identity(dim - 3, dim - 3);
  good_ = true;
}

bool CpuBackend::batchUpdateFullState(int na, const float* H, const float* R,
                                      const float* nu, int two_m) {
  /* Host buffers are column-major (Eigen default) -> map without copies. */
  Map<const MatrixXf> Hm(H, two_m, na);
  Map<const MatrixXf> Rm(R, two_m, two_m);
  Map<const VectorXf> num(nu, two_m);

  auto P_a = P_.topLeftCorner(na, na);
  MatrixXf HP = Hm * P_a;                            /* (2m x na)  */
  MatrixXf S  = (HP * Hm.transpose()) + Rm;          /* (2m x 2m)  */
  MatrixXf K  = P_a * Hm.transpose() * S.inverse();  /* (na x 2m)  */

  /* No yaw wrap here — EKFOdom wraps after refreshing its mirror, matching the
     GPU peer (which does not wrap on the device). */
  x_.head(na).noalias() += (K * num);
  P_a.noalias() -= K * HP;
  return true;
}

void CpuBackend::motionPropagate(int na, const float G3x3[9],
                                 float dwx, float dwy, float dtheta,
                                 float add_xx, float add_yy, float add_yaw) {
  Matrix3f G;
  G << G3x3[0], G3x3[1], G3x3[2],
       G3x3[3], G3x3[4], G3x3[5],
       G3x3[6], G3x3[7], G3x3[8];

  auto Pa = P_.topLeftCorner(na, na);
  Pa.topRows(3)  = (G * Pa.topRows(3)).eval();               /* rows: G * P    */
  Pa.leftCols(3) = (Pa.leftCols(3) * G.transpose()).eval();  /* cols: P * G^T  */

  P_(0, 0) += add_xx;
  P_(1, 1) += add_yy;
  P_(2, 2) += add_yaw;

  x_(0) += dwx;
  x_(1) += dwy;
  /* wrap to (-2pi, 2pi), matching the GPU k_pose_compose / EKFOdom::normalizeYaw */
  float y = x_(2) + dtheta;
  const float TWO_PI = 6.28318530717958647692f;
  while (y >  TWO_PI) y -= TWO_PI;
  while (y < -TWO_PI) y += TWO_PI;
  x_(2) = y;
}

void CpuBackend::addPoseCovDiag(float dxx, float dyy, float dyaw) {
  P_(0, 0) += dxx;
  P_(1, 1) += dyy;
  P_(2, 2) += dyaw;
}

void CpuBackend::insertLandmark(int k, float mx, float my) {
  x_(3 + 2 * k)     = mx;
  x_(3 + 2 * k + 1) = my;
}

void CpuBackend::downloadPose3(float pose3[3]) const {
  pose3[0] = x_(0); pose3[1] = x_(1); pose3[2] = x_(2);
}

void CpuBackend::downloadPoseCov3(float cov3[3]) const {
  cov3[0] = P_(0, 0); cov3[1] = P_(1, 1); cov3[2] = P_(2, 2);
}

void CpuBackend::downloadPoseBlock3x3(float P33[9]) const {
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      P33[r * 3 + c] = P_(r, c);   /* row-major out */
}

void CpuBackend::downloadLandmarks(float* xy, int count) const {
  if (count <= 0) return;
  for (int i = 0; i < 2 * count; ++i) xy[i] = x_(3 + i);
}

void CpuBackend::uploadPose3(const float pose3[3]) {
  x_(0) = pose3[0]; x_(1) = pose3[1]; x_(2) = pose3[2];
}

void CpuBackend::uploadPoseBlock3x3(const float P33[9]) {
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      P_(r, c) = P33[r * 3 + c];   /* row-major in */
}

void CpuBackend::debugSetState(const float* P, const float* x) {
  P_ = Map<const MatrixXf>(P, dim_, dim_);   /* col-major in */
  x_ = Map<const VectorXf>(x, dim_);
}

void CpuBackend::debugGetState(float* P, float* x) const {
  Map<MatrixXf>(P, dim_, dim_) = P_;         /* col-major out */
  Map<VectorXf>(x, dim_) = x_;
}
