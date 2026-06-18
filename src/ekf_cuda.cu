#include <cuda_cone_fused/ekf_cuda.hpp>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <thrust/device_vector.h>

#include <string>
#include <cmath>
#include <exception>

using thrust::raw_pointer_cast;

/* ----------------------------------------------------------------------------
 * Error plumbing
 * ------------------------------------------------------------------------- */
#define CK(call, what)                                                        \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            p_->err = std::string(what) + ": " + cudaGetErrorString(_e);      \
            return false;                                                     \
        }                                                                     \
    } while (0)

#define BK(call, what)                                                        \
    do {                                                                      \
        cublasStatus_t _s = (call);                                           \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                    \
            p_->err = std::string(what) + ": cublas status " +                \
                      std::to_string((int)_s);                                \
            return false;                                                     \
        }                                                                     \
    } while (0)

#define SK(call, what)                                                        \
    do {                                                                      \
        cusolverStatus_t _s = (call);                                         \
        if (_s != CUSOLVER_STATUS_SUCCESS) {                                  \
            p_->err = std::string(what) + ": cusolver status " +              \
                      std::to_string((int)_s);                                \
            return false;                                                     \
        }                                                                     \
    } while (0)

/* ----------------------------------------------------------------------------
 * Kernels
 * ------------------------------------------------------------------------- */

/* set P diagonal: first 3 -> 1, landmarks -> inf_init (P already zeroed). */
__global__ void k_init_diag(float* P, int dim, float inf_init) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) P[(size_t)i * dim + i] = (i < 3) ? 1.0f : inf_init;
}

__global__ void k_add_pose_diag(float* P, int dim, float a, float b, float c) {
    if (threadIdx.x == 0) {
        P[0] += a;                      /* (0,0) */
        P[(size_t)dim + 1] += b;        /* (1,1) */
        P[(size_t)2 * dim + 2] += c;    /* (2,2) */
    }
}

/* x0+=dwx; x1+=dwy; x2 = wrap_2pi(x2 + dtheta)  (matches EKFOdom::normalizeYaw) */
__global__ void k_pose_compose(float* x, float dwx, float dwy, float dtheta) {
    if (threadIdx.x == 0) {
        x[0] += dwx;
        x[1] += dwy;
        float y = x[2] + dtheta;
        const float TWO_PI = 6.28318530717958647692f;
        while (y >  TWO_PI) y -= TWO_PI;
        while (y < -TWO_PI) y += TWO_PI;
        x[2] = y;
    }
}

__global__ void k_set_vec2(float* x, int idx, float a, float b) {
    if (threadIdx.x == 0) { x[idx] = a; x[idx + 1] = b; }
}

/* ----------------------------------------------------------------------------
 * Impl — all device storage is RAII via thrust::device_vector (allocated once,
 * grown only when a scan needs more measurement rows). raw_pointer_cast hands
 * the underlying device pointer to cuBLAS / cuSOLVER.
 * ------------------------------------------------------------------------- */
struct EkfCudaBackend::Impl {
    int dim = 0;
    int cap_two_m = 0;     /* current measurement capacity (rows) */
    bool good = false;
    std::string err;

    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t solver = nullptr;

    /* resident state */
    thrust::device_vector<float> P;   /* dim x dim, col-major, ld = dim */
    thrust::device_vector<float> x;   /* dim */

    /* per-scan scratch */
    thrust::device_vector<float> H;   /* 2m x dim */
    thrust::device_vector<float> S;   /* 2m x 2m (holds R, then S) */
    thrust::device_vector<float> nu;  /* 2m */
    thrust::device_vector<float> HP;  /* 2m x dim */
    thrust::device_vector<float> Kt;  /* 2m x dim (= K^T) */

    /* motion scratch */
    thrust::device_vector<float> G;     /* 3x3 */
    thrust::device_vector<float> tmp;   /* 3 x dim */
    thrust::device_vector<float> tmp2;  /* dim x 3 */

    /* solver scratch */
    thrust::device_vector<int>   ipiv;
    thrust::device_vector<int>   info;
    thrust::device_vector<float> work;

    float* dP()  { return raw_pointer_cast(P.data()); }
    float* dx()  { return raw_pointer_cast(x.data()); }
    float* dH()  { return raw_pointer_cast(H.data()); }
    float* dS()  { return raw_pointer_cast(S.data()); }
    float* dnu() { return raw_pointer_cast(nu.data()); }
    float* dHP() { return raw_pointer_cast(HP.data()); }
    float* dKt() { return raw_pointer_cast(Kt.data()); }
    float* dG()  { return raw_pointer_cast(G.data()); }
    float* dtmp(){ return raw_pointer_cast(tmp.data()); }
    float* dtmp2(){return raw_pointer_cast(tmp2.data()); }
    int*   dipiv(){return raw_pointer_cast(ipiv.data()); }
    int*   dinfo(){return raw_pointer_cast(info.data()); }
    float* dwork(){return raw_pointer_cast(work.data()); }

    bool alloc_scratch(int two_m);
};

/* (Re)allocate measurement / solver scratch for a given two_m capacity. */
bool EkfCudaBackend::Impl::alloc_scratch(int two_m) {
    auto p_ = this;  /* for SK macro */
    try {
        H.resize((size_t)two_m * dim);
        S.resize((size_t)two_m * two_m);
        nu.resize((size_t)two_m);
        HP.resize((size_t)two_m * dim);
        Kt.resize((size_t)two_m * dim);
        ipiv.resize((size_t)two_m);
    } catch (const std::exception& e) {
        err = std::string("scratch alloc: ") + e.what();
        return false;
    }
    int lw = 0;
    SK(cusolverDnSgetrf_bufferSize(solver, two_m, two_m, dS(), two_m, &lw),
       "getrf_bufferSize");
    try {
        work.resize((size_t)lw);
    } catch (const std::exception& e) {
        err = std::string("work alloc: ") + e.what();
        return false;
    }
    cap_two_m = two_m;
    return true;
}

const char* EkfCudaBackend::lastError() const { return p_->err.c_str(); }
bool EkfCudaBackend::ok() const { return p_->good; }

EkfCudaBackend::EkfCudaBackend(int dim, int max_two_m, float inf_init)
    : p_(new Impl) {
    p_->dim = dim;

    if (cudaSetDevice(0) != cudaSuccess) { p_->err = "cudaSetDevice failed"; return; }
    if (cublasCreate(&p_->blas) != CUBLAS_STATUS_SUCCESS) { p_->err = "cublasCreate failed"; return; }
    if (cusolverDnCreate(&p_->solver) != CUSOLVER_STATUS_SUCCESS) { p_->err = "cusolverDnCreate failed"; return; }

    try {
        /* device_vector::resize value-initialises to 0, giving the memset for free */
        p_->P.resize((size_t)dim * dim);
        p_->x.resize((size_t)dim);
        p_->info.resize(1);
        p_->G.resize(9);
        p_->tmp.resize((size_t)3 * dim);
        p_->tmp2.resize((size_t)dim * 3);
    } catch (const std::exception& e) {
        p_->err = std::string("state alloc: ") + e.what();
        return;
    }

    /* init resident P = blockdiag(I3, inf*I); x already 0 */
    {
        int t = 256, b = (dim + t - 1) / t;
        k_init_diag<<<b, t>>>(p_->dP(), dim, inf_init);
        if (cudaGetLastError() != cudaSuccess) { p_->err = "k_init_diag launch"; return; }
    }

    if (!p_->alloc_scratch(max_two_m > 0 ? max_two_m : 8)) return;

    if (cudaDeviceSynchronize() != cudaSuccess) { p_->err = "init sync"; return; }
    p_->good = true;
}

EkfCudaBackend::~EkfCudaBackend() {
    if (!p_) return;
    /* device buffers free themselves (RAII); just the library handles remain */
    if (p_->solver) cusolverDnDestroy(p_->solver);
    if (p_->blas) cublasDestroy(p_->blas);
    delete p_;
}

/* ----------------------------------------------------------------------------
 * Hot path: full-state batch update
 *
 *   HP = H P            (2m x na)
 *   S  = HP H^T + R     (2m x 2m)
 *   solve S K^T = HP    (K^T : 2m x na)   [getrf/getrs; HP copied into Kt]
 *   x_head += K nu      (gemv, op(K^T)=T)
 *   P     -= K HP       (gemm, op(K^T)=T)
 * ------------------------------------------------------------------------- */
bool EkfCudaBackend::batchUpdateFullState(int na, const float* H, const float* R,
                                          const float* nu, int two_m) {
    if (!p_->good) return false;
    if (two_m > p_->cap_two_m) {
        if (!p_->alloc_scratch(two_m)) return false;
    }
    const int dim = p_->dim;
    const float one = 1.0f, zero = 0.0f, neg = -1.0f;

    /* upload measurement matrices (small) */
    CK(cudaMemcpy(p_->dH(), H, (size_t)two_m * na * sizeof(float), cudaMemcpyHostToDevice), "up H");
    CK(cudaMemcpy(p_->dS(), R, (size_t)two_m * two_m * sizeof(float), cudaMemcpyHostToDevice), "up R");
    CK(cudaMemcpy(p_->dnu(), nu, (size_t)two_m * sizeof(float), cudaMemcpyHostToDevice), "up nu");

    /* HP = H * P_active     C(2m x na) = A(2m x na) * B(na x na) */
    BK(cublasSgemm(p_->blas, CUBLAS_OP_N, CUBLAS_OP_N,
                   two_m, na, na, &one,
                   p_->dH(), two_m, p_->dP(), dim, &zero, p_->dHP(), two_m),
       "gemm HP");

    /* S = HP * H^T + R     (S preloaded with R, beta=1) */
    BK(cublasSgemm(p_->blas, CUBLAS_OP_N, CUBLAS_OP_T,
                   two_m, two_m, na, &one,
                   p_->dHP(), two_m, p_->dH(), two_m, &one, p_->dS(), two_m),
       "gemm S");

    /* Kt <- HP, then solve S * Kt = HP  ->  Kt = K^T  (2m x na) */
    CK(cudaMemcpy(p_->dKt(), p_->dHP(), (size_t)two_m * na * sizeof(float),
                  cudaMemcpyDeviceToDevice), "copy HP->Kt");
    SK(cusolverDnSgetrf(p_->solver, two_m, two_m, p_->dS(), two_m,
                        p_->dwork(), p_->dipiv(), p_->dinfo()), "getrf S");
    SK(cusolverDnSgetrs(p_->solver, CUBLAS_OP_N, two_m, na, p_->dS(), two_m,
                        p_->dipiv(), p_->dKt(), two_m, p_->dinfo()), "getrs S");

    /* x_head += K * nu       y(na) = (K^T)^T * nu */
    BK(cublasSgemv(p_->blas, CUBLAS_OP_T, two_m, na, &one,
                   p_->dKt(), two_m, p_->dnu(), 1, &one, p_->dx(), 1),
       "gemv x");

    /* P_active -= K * HP      C(na x na) = (K^T)^T (na x 2m) * HP (2m x na) */
    BK(cublasSgemm(p_->blas, CUBLAS_OP_T, CUBLAS_OP_N,
                   na, na, two_m, &neg,
                   p_->dKt(), two_m, p_->dHP(), two_m, &one, p_->dP(), dim),
       "gemm P update");

    CK(cudaDeviceSynchronize(), "batch sync");
    return true;
}

/* ----------------------------------------------------------------------------
 * Motion propagation (setPose predict step)
 * ------------------------------------------------------------------------- */
void EkfCudaBackend::motionPropagate(int na, const float G3x3[9],
                                     float dwx, float dwy, float dtheta,
                                     float add_xx, float add_yy, float add_yaw) {
    if (!p_->good) return;
    const int dim = p_->dim;
    const float one = 1.0f, zero = 0.0f;

    /* G arrives row-major; cuBLAS wants column-major 3x3 -> transpose on upload. */
    float Gcm[9];
    Gcm[0] = G3x3[0]; Gcm[1] = G3x3[3]; Gcm[2] = G3x3[6];
    Gcm[3] = G3x3[1]; Gcm[4] = G3x3[4]; Gcm[5] = G3x3[7];
    Gcm[6] = G3x3[2]; Gcm[7] = G3x3[5]; Gcm[8] = G3x3[8];
    cudaMemcpy(p_->dG(), Gcm, 9 * sizeof(float), cudaMemcpyHostToDevice);

    /* topRows(3): tmp(3 x na) = G(3x3) * P[0:3,0:na]; write back into strided P. */
    cublasSgemm(p_->blas, CUBLAS_OP_N, CUBLAS_OP_N,
                3, na, 3, &one, p_->dG(), 3, p_->dP(), dim, &zero, p_->dtmp(), 3);
    cudaMemcpy2D(p_->dP(), (size_t)dim * sizeof(float),
                 p_->dtmp(), (size_t)3 * sizeof(float),
                 3 * sizeof(float), na, cudaMemcpyDeviceToDevice);

    /* leftCols(3): tmp2(na x 3) = P[0:na,0:3] * G^T; write back (reads updated P). */
    cublasSgemm(p_->blas, CUBLAS_OP_N, CUBLAS_OP_T,
                na, 3, 3, &one, p_->dP(), dim, p_->dG(), 3, &zero, p_->dtmp2(), na);
    cudaMemcpy2D(p_->dP(), (size_t)dim * sizeof(float),
                 p_->dtmp2(), (size_t)na * sizeof(float),
                 (size_t)na * sizeof(float), 3, cudaMemcpyDeviceToDevice);

    /* additive motion noise on pose diagonal + pose-mean composition */
    k_add_pose_diag<<<1, 1>>>(p_->dP(), dim, add_xx, add_yy, add_yaw);
    k_pose_compose<<<1, 1>>>(p_->dx(), dwx, dwy, dtheta);
    cudaDeviceSynchronize();
}

void EkfCudaBackend::addPoseCovDiag(float dxx, float dyy, float dyaw) {
    if (!p_->good) return;
    k_add_pose_diag<<<1, 1>>>(p_->dP(), p_->dim, dxx, dyy, dyaw);
    cudaDeviceSynchronize();
}

void EkfCudaBackend::insertLandmark(int k, float mx, float my) {
    if (!p_->good) return;
    k_set_vec2<<<1, 1>>>(p_->dx(), 3 + 2 * k, mx, my);
    cudaDeviceSynchronize();
}

void EkfCudaBackend::downloadPose3(float pose3[3]) const {
    cudaMemcpy(pose3, p_->dx(), 3 * sizeof(float), cudaMemcpyDeviceToHost);
}

void EkfCudaBackend::downloadPoseCov3(float cov3[3]) const {
    const int dim = p_->dim;
    cudaMemcpy(&cov3[0], p_->dP() + 0,                 sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&cov3[1], p_->dP() + (size_t)dim + 1,   sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&cov3[2], p_->dP() + (size_t)2*dim + 2, sizeof(float), cudaMemcpyDeviceToHost);
}

void EkfCudaBackend::downloadPoseBlock3x3(float P33[9]) const {
    const int dim = p_->dim;
    /* 3x3 top-left, strided columns -> row-major out */
    float col[3];
    for (int c = 0; c < 3; ++c) {
        cudaMemcpy(col, p_->dP() + (size_t)c * dim, 3 * sizeof(float), cudaMemcpyDeviceToHost);
        for (int r = 0; r < 3; ++r) P33[r * 3 + c] = col[r];
    }
}

void EkfCudaBackend::downloadLandmarks(float* xy, int count) const {
    if (count <= 0) return;
    cudaMemcpy(xy, p_->dx() + 3, (size_t)2 * count * sizeof(float), cudaMemcpyDeviceToHost);
}

void EkfCudaBackend::uploadPose3(const float pose3[3]) {
    cudaMemcpy(p_->dx(), pose3, 3 * sizeof(float), cudaMemcpyHostToDevice);
}

void EkfCudaBackend::uploadPoseBlock3x3(const float P33[9]) {
    const int dim = p_->dim;
    /* row-major in -> column-major 3x3, then strided write into P[0:3,0:3] */
    float cm[9];
    cm[0]=P33[0]; cm[1]=P33[3]; cm[2]=P33[6];
    cm[3]=P33[1]; cm[4]=P33[4]; cm[5]=P33[7];
    cm[6]=P33[2]; cm[7]=P33[5]; cm[8]=P33[8];
    cudaMemcpy2D(p_->dP(), (size_t)dim * sizeof(float),
                 cm, (size_t)3 * sizeof(float),
                 3 * sizeof(float), 3, cudaMemcpyHostToDevice);
}

void EkfCudaBackend::debugSetState(const float* P, const float* x) {
    const int dim = p_->dim;
    cudaMemcpy(p_->dP(), P, (size_t)dim * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(p_->dx(), x, (size_t)dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void EkfCudaBackend::debugGetState(float* P, float* x) const {
    const int dim = p_->dim;
    cudaMemcpy(P, p_->dP(), (size_t)dim * dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(x, p_->dx(), (size_t)dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}
