#pragma once

#include <cstddef>

#include <cuda_cone_fused/backend/ekf_backend.hpp>

/**
 * Resident-state GPU backend for the EKF-SLAM filter.
 *
 * Goal: keep execution under a few ms (target < 1 ms) by holding the LARGE
 * state on the device permanently and never streaming it across the bus per
 * scan. The covariance P_ (dim x dim) and the state x_ (dim) are allocated and
 * initialised ONCE here and live on the GPU for the whole run. Every operation
 * that touches them is executed on the device:
 *
 *   - batchUpdateFullState : the joint full-state measurement update
 *                            (HP = H P, S = HP H^T + R, solve S K^T = HP,
 *                             x += K nu, P -= K HP) — the O(na^2 m) hot path.
 *   - motionPropagate      : the setPose() predict step (P pose strips through
 *                            the motion Jacobian G, additive motion noise, and
 *                            the pose-mean composition).
 *   - addPoseCovDiag       : the setPoseCovariance() diagonal increments.
 *   - insertLandmark       : lap-1 new-cone mean write.
 *
 * Only TINY slices ever cross the bus: the per-scan measurement matrices in
 * (H, R, nu — a few hundred KB at most), and out the 3-vector pose, the 3x3
 * pose-covariance block (for data association + publishing) and, on demand, the
 * active landmark means (for markers). The dim x dim covariance never moves.
 *
 * All host buffers are COLUMN-MAJOR contiguous (Eigen default) so they map onto
 * cuBLAS without transposes. The device P uses leading dimension = dim.
 */
class CudaBackend : public IEkfBackend {
public:
    /**
     * @param dim       full state dimension (= 3 + 2*N_CONES).
     * @param max_two_m largest 2*m (stacked innovation rows) expected per scan;
     *                  measurement scratch grows automatically if exceeded.
     * @param inf_init  initial variance on the (inactive) landmark diagonal,
     *                  matching the CPU constructor's INF block.
     *
     * Initialises device P = blockdiag(I_3, inf_init * I_2N) and x = 0, exactly
     * like the CPU EKFOdom constructor.
     */
    CudaBackend(int dim, int max_two_m, float inf_init);
    ~CudaBackend() override;

    CudaBackend(const CudaBackend&) = delete;
    CudaBackend& operator=(const CudaBackend&) = delete;

    /** True if construction and all CUDA setup succeeded. */
    bool ok() const override;
    /** Human-readable last-error string (empty if none). */
    const char* lastError() const override;

    /* ---- hot path -------------------------------------------------------- */

    /**
     * Full-state joint EKF update on the active na x na block of the resident P
     * and the active head of the resident x. Mirrors the CPU batch path.
     *
     * @param na          active dimension (3 + 2*landmark_count).
     * @param H           [in] host, (2m)*na col-major. Stacked Jacobian.
     * @param R           [in] host, (2m)*(2m) col-major. Block-diag meas. noise.
     * @param nu          [in] host, 2m. Stacked (wrapped) innovation.
     * @param two_m       2*m.
     * @return false on any CUDA/cuBLAS/cuSOLVER failure (state left unchanged).
     */
    bool batchUpdateFullState(int na, const float* H, const float* R,
                              const float* nu, int two_m) override;

    /* ---- predict / motion (setPose) ------------------------------------- */

    /**
     * Apply the predict-step covariance propagation and pose-mean composition
     * to the resident state, exactly as EKFOdom::setPose:
     *   P[0:3, 0:na] = G * P[0:3, 0:na];  P[0:na, 0:3] = P[0:na, 0:3] * G^T
     *   P(0,0)+=add_xx; P(1,1)+=add_yy; P(2,2)+=add_yaw
     *   x(0)+=dwx; x(1)+=dwy; x(2)=wrapped(x(2)+dtheta)
     * @param G3x3 row-major 3x3 motion Jacobian (only G(0,2),G(1,2) differ from I).
     */
    void motionPropagate(int na, const float G3x3[9],
                         float dwx, float dwy, float dtheta,
                         float add_xx, float add_yy, float add_yaw) override;

    /** setPoseCovariance: add increments to the pose covariance diagonal. */
    void addPoseCovDiag(float dxx, float dyy, float dyaw) override;

    /* ---- structure / IO -------------------------------------------------- */

    /** Lap-1 new cone: write landmark k mean (x[3+2k], x[3+2k+1]). */
    void insertLandmark(int k, float mx, float my) override;

    /** Download x(0..2). */
    void downloadPose3(float pose3[3]) const override;
    /** Download P(0,0), P(1,1), P(2,2). */
    void downloadPoseCov3(float cov3[3]) const override;
    /** Download the 3x3 top-left pose covariance block (row-major). */
    void downloadPoseBlock3x3(float P33[9]) const override;
    /** Download the first `count` landmark means: xy[2i],xy[2i+1] = x[3+2i..]. */
    void downloadLandmarks(float* xy, int count) const override;
    /** Overwrite the pose mean x(0..2) on the device (host-side scalar fixups). */
    void uploadPose3(const float pose3[3]) override;
    /** Overwrite the 3x3 top-left pose covariance block (row-major in). Used by
     *  the rigid pose-only update, which is computed host-side on the 3x3 mirror
     *  and written back (it touches only the pose block). */
    void uploadPoseBlock3x3(const float P33[9]) override;

    /* ---- debug / validation (full dim x dim transfers) ------------------- */
    /** Overwrite the resident P (dim*dim col-major) and x (dim). Test/seed use. */
    void debugSetState(const float* P, const float* x) override;
    /** Read back the full resident P (dim*dim col-major) and x (dim). */
    void debugGetState(float* P, float* x) const override;

private:
    struct Impl;
    Impl* p_;
};
