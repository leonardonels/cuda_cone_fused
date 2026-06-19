#include <cuda_cone_fused/backend/ekf_backend.hpp>
#include <cuda_cone_fused/backend/cpu_backend.hpp>

#ifdef USE_CUDA
#include <cuda_cone_fused/backend/cuda_backend.hpp>
#endif

#include <iostream>

/* The SINGLE remaining USE_CUDA decision point (Factory, Bridge).
   Everything downstream talks to IEkfBackend and is platform-agnostic. */
std::unique_ptr<IEkfBackend> makeEkfBackend(int dim, int max_two_m, float inf_init) {
#ifdef USE_CUDA
  auto gpu = std::make_unique<CudaBackend>(dim, max_two_m, inf_init);
  if (gpu->ok()) {
    std::cerr << "EKF backend: CUDA ACTIVE (resident GPU state)\n";
    return gpu;
  }
  std::cerr << "EKF backend: CUDA init FAILED (" << gpu->lastError()
            << ") -> CPU fallback\n";
#else
  (void)max_two_m;
#endif
  std::cerr << "EKF backend: CPU (host Eigen)\n";
  return std::make_unique<CpuBackend>(dim, inf_init);
}
