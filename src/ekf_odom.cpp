#include <cuda_cone_fused/ekf_odom.hpp>

#include <omp.h>
#include <chrono>
#include <vector>
#include <utility>

/* Constructor */
EKFOdom::EKFOdom(Vector2f process_noise, Vector2f motion_noise, const float alpha, int eigen_threads)
{
    /* Eigen/OpenMP thread count. Default 1: at the matrix sizes here multithreaded
       GEMM mostly adds fork/join jitter (p99 latency blows up at 6 threads on the
       Orin's 6-core SoC) — and with the GPU backend the heavy work is offloaded
       anyway. Configurable via the `eigen_threads` node parameter for CPU tuning. */
    if (eigen_threads < 1) eigen_threads = 1;
    omp_set_num_threads(eigen_threads);
    Eigen::setNbThreads(eigen_threads);

    std::cerr << "Eigen threads: " << Eigen::nbThreads() << "\n";
    
    /* Initialize state vector */
    this->x_ = VectorXf::Zero(2 * N_CONES + 3);

    /* Initialize covariance matrix */
    this->P_ = MatrixXf::Zero(2 * N_CONES + 3, 2 * N_CONES + 3);
    // this->P_.setConstant(INF);
    this->P_.block(0, 0, 3, 3) = Matrix3f::Identity();
    this->P_.bottomRightCorner(2 * N_CONES, 2 * N_CONES) = INF * Eigen::MatrixXf::Identity(2 * N_CONES, 2 * N_CONES);

    /* Initialize the (range, bearing) measurement-noise covariance Q_. It is the
       additive term in the innovation covariance S = H P H^T + Q_ used by every
       correct() update. Diagonal entries come from the `process_noise` parameter
       (sigma_range [m], sigma_bearing [rad]). */
    this->Q_ = Matrix2f::Identity();
    for (uint8_t i=0;i < process_noise.size(); i++)
    {
        this->Q_(i,i) = process_noise(i);
    }

    std::cerr << "Q_ : \n" << this->Q_ << "\n";

    /* Initialize motion (predict-step) process noise: additive pose covariance
       per unit of travelled distance / turned angle (see setPose). */
    this->q_motion_pos_ = motion_noise(0);
    this->q_motion_yaw_ = motion_noise(1);

    std::cerr << "M_:\n" << this->q_motion_pos_ << "    0\n   0 " << this->q_motion_yaw_ << "\n";

    /* Initialize max new cone dist */
    this->max_new_cone_dist = alpha;

#ifdef USE_CUDA
    /* Bring up the resident-state GPU backend. It initialises the device P and x
       identically to the CPU constructor above (P = blockdiag(I3, INF*I), x=0),
       so the host mirrors (this->x_, this->P_) are already consistent with the
       device. max_two_m = 256 -> up to 128 cones/scan before an (automatic)
       scratch grow. On any failure we fall back to the CPU path. */
    this->gpu_ = std::make_unique<EkfCudaBackend>(2 * N_CONES + 3, 256, INF);
    this->use_gpu_ = this->gpu_->ok();
    if (this->use_gpu_)
        std::cerr << "EKFOdom: CUDA backend ACTIVE (resident GPU state)\n";
    else
        std::cerr << "EKFOdom: CUDA backend init FAILED (" << this->gpu_->lastError()
                  << ") -> CPU fallback\n";
#endif
}

#ifdef USE_CUDA
void EKFOdom::syncFromDevice() {
    /* Pull the small host mirrors back from the authoritative device state. */
    float pose3[3];
    this->gpu_->downloadPose3(pose3);
    this->x_(0) = pose3[0];
    this->x_(1) = pose3[1];
    this->x_(2) = pose3[2];
    if (this->landmark_count > 0)
        this->gpu_->downloadLandmarks(this->x_.data() + 3,
                                      static_cast<int>(this->landmark_count));
    float cov3[3];
    this->gpu_->downloadPoseCov3(cov3);
    this->P_(0, 0) = cov3[0];
    this->P_(1, 1) = cov3[1];
    this->P_(2, 2) = cov3[2];
}
#endif


/* EKF Correct step function.
   Note: there is no separate predict() entry point. The motion (predict) step is
   driven by FAST-LIMO odometry and applied in setPose()/setPoseCovariance(), which
   propagate the pose mean and covariance before the next correction. */
size_t EKFOdom::correct(const Vector3f *z, const size_t act_cones_detected) {
    /* Diagnostic: # of observations that passed data association this scan. */
    size_t associated = 0;
    /* Vector that stores the temp parameters for a hypothetical new cone */
    Vector2f tmp_cone;
#ifndef USE_CUDA
    /* Vector that stores the delta X,Y between a cone and the vehicle (single-cone path) */
    Vector2f delta_k;
#endif

    size_t i,k, j = 0;
    float min_dist;

#ifdef USE_CUDA
    /* Associations collected this scan for the joint (batch) update path:
       (landmark index k, observed (range, bearing)). GPU build only — the
       CPU-only build does the single-cone update inline. */
    std::vector<std::pair<size_t, Vector2f>> batch_obs;
#endif

    // VectorXf new_state_update = VectorXf::Zero(2*N_CONES+3);
    // MatrixXf new_covariance_update = MatrixXf::Zero(2*N_CONES+3, 2*N_CONES+3);

    for(i = 0; i < act_cones_detected; i++)
    {
        /* Discard orange cones */
        if ((z[i][2] == 2) || (z[i][2] == 3))
        {
            // std::cerr << "Skipping orange cone.\n";
            continue;   
        }

        // bool is_new_cone = false;
        /* Reset min distance and k index */
        min_dist = __FLT_MAX__;
        k = 0;

        /* Compute expected NEW cone position */
        tmp_cone << this->x_(0), this->x_(1);
        tmp_cone = tmp_cone + (z[i](0) * Vector2f(cos(z[i](1) + this->x_(2)), sin(z[i](1) + this->x_(2))));

        // std::cerr << "EKF INDEX: " << i << " TMP CONE X: " << tmp_cone(0) << " TMP CONE Y: " << tmp_cone(1) << "\n";

        /* Check if this cone has been seen before based on nearest neightbour (min distance) */

        for (j = 0; j < this->landmark_count; j++)
        {
            float tempDistance = euclideanDistance(this->x_(j * 2 + 3), tmp_cone(0), this->x_(j * 2 + 4), tmp_cone(1));
            // // std::cerr << "Distance between cone n° "<<j<<" ("<<this->x_(j * 2 + 3)<<","<<this->x_(j * 2 + 4)<<
            //     ") and new cone ("<<tmp_cone(0)<<","<<tmp_cone(1)<<") is: " << tempDistance <<"\n";
            if (tempDistance < min_dist)
            {
                min_dist = tempDistance;
                k = j;
            }
        }

        if (this->is_first_lap_completed)
        {
            /* LAP 2+: associate by MAHALANOBIS distance (adaptive to the pose
               uncertainty via S), not by a fixed Euclidean radius. A fixed metric
               radius drops far cones once heading drift displaces them past it,
               which stalls the correction and lets the drift run away; the
               Mahalanobis gate scales with uncertainty so those cones keep
               correcting. The map is frozen here, so a cone that fails the gate is
               simply discarded (no new landmarks after lap 1). This is the single
               gating point for lap 2+ — the update steps below no longer re-gate. */
            Vector2f dlt = this->x_.segment(3 + 2*k, 2) - Vector2f(this->x_(0), this->x_(1));
            float qd = dlt.transpose() * dlt;
            qd = (qd == 0.0f) ? __FLT_MIN__ : qd;
            const float sqd = sqrt(qd);

            Vector2f expz(sqd, normalizeAngle(atan2(dlt(1), dlt(0)) - this->x_(2)));
            Vector2f nu = Vector2f(z[i](0), z[i](1)) - expz;
            nu(1) = normalizeAngle(nu(1));

            /* Pose-only innovation covariance S = H_p P_pp H_p^T + Q (2x2). */
            Matrix<float,2,3> Hp;
            Hp << -sqd*dlt(0), -sqd*dlt(1),  0.0f,
                      dlt(1),     -dlt(0),    -qd;
            Hp /= qd;
            Matrix3f P_pp = this->P_.topLeftCorner(3,3);
            Matrix2f S = (Hp * P_pp * Hp.transpose()) + this->Q_;

            const float d2 = nu.transpose() * S.inverse() * nu;
            if (d2 > this->assoc_maha_gate_)
            {
                continue;   /* not consistent with the nearest landmark -> drop */
            }
            this->s_(k).setColor(static_cast<uint32_t>(z[i][2]));
        }
        else
        {
            /* LAP 1: build the map with Euclidean nearest-neighbour + a fixed
               radius (the map/pose covariance is still immature, so a metric gate
               is simpler and robust for the new-vs-existing decision). */
            if (this->max_new_cone_dist < min_dist)
            {
                /* New cone */
                k = this->landmark_count;
                this->landmark_count++;
                this->x_.segment(3 + (2*k), 2) = tmp_cone;
#ifdef USE_CUDA
                /* Mirror the new landmark mean onto the resident device state.
                   Its covariance is already INF on the device (pre-initialised),
                   matching the host P_. */
                if (this->use_gpu_)
                    this->gpu_->insertLandmark(static_cast<int>(k), tmp_cone(0), tmp_cone(1));
#endif
                ColorLogic c;
                c.setColor(static_cast<uint32_t>(z[i][2]));
                this->s_(k) = c;
            }
            else
            {
                /* Existing cone: update color counter */
                this->s_(k).setColor(static_cast<uint32_t>(z[i][2]));
            }
        }

        // std::cerr << "EKF INDEX: " << i << " k is: " << k << "\n";
        associated++;   /* reached only by observations that passed association */

#ifdef USE_CUDA
        /* GPU build: defer EVERY association to the joint update applied once
           after the loop. The device holds the authoritative P_, so all
           covariance-modifying updates must go through the offloaded joint path
           (a single-cone CPU update would read/write a stale host P_); the
           single-cone path below is therefore compiled out entirely. */
        batch_obs.emplace_back(k, Vector2f(z[i](0), z[i](1)));
#else
        /* CPU-only (debug) build: single-cone-per-scan update on the last cone. */
        if (i == (act_cones_detected-1))
        {
            delta_k = this->x_.segment((3 + (2*k)), 2) - Vector2f(this->x_(0), this->x_(1)); /* This is equivalent to ((cone_x - vehicle_x), (cone_y - vehicle_y)) */

            float q_k = delta_k.transpose() * delta_k; /* Compute euclidean distance ^2 */

            q_k = (q_k == 0.0f) ? (__FLT_MIN__) : (q_k);

            /* Compute predicted measurement */
            Vector2f exp_z_k
            (
                sqrt(q_k),
                atan2(delta_k(1), delta_k(0)) - this->x_(2)
            );

            /* Normalize exp z angle */
            exp_z_k(1) = normalizeAngle(exp_z_k(1));

            // if (!is_new_cone)
            // {
                // std::cerr << "EKF INDEX: " << i << " MEAS z is: \n" << z[i] << "\n";
                // std::cerr << "EKF INDEX: " << i << " EXPC z is: \n" << exp_z_k << "\n";
            // }

            /* Low-Jacobian block (2x5) over [pose | landmark k], pre-scaled by
               1/q, and the (wrapped) measurement innovation. */
            MatrixXf ht_k = MatrixXf::Zero(2,5);
            ht_k(0,0) = -sqrt(q_k) * delta_k(0);
            ht_k(0,1) = -sqrt(q_k) * delta_k(1);
            ht_k(0,2) = 0.0;
            ht_k(0,3) = sqrt(q_k) * delta_k(0);
            ht_k(0,4) = sqrt(q_k) * delta_k(1);

            ht_k(1,0) = delta_k(1);
            ht_k(1,1) = -delta_k(0);
            ht_k(1,2) = -q_k;
            ht_k(1,3) = -delta_k(1);
            ht_k(1,4) = delta_k(0);
            ht_k /= q_k;

            Vector2f meas_diff = Vector2f(z[i](0), z[i](1)) - exp_z_k;
            /* Wrap the bearing innovation to [-pi, pi]: both terms are normalized
               individually but their difference can straddle +/-pi (e.g. +3.0 -
               (-3.0) = 6.0 rad instead of -0.28), which would otherwise snap the
               pose when a cone is re-observed across the wrap. */
            meas_diff(1) = normalizeAngle(meas_diff(1));

            if (!(this->is_first_lap_completed && this->freeze_map_))
            {
                /* FULL-STATE EKF-SLAM update on the active sub-block (pose +
                   mapped landmarks; inactive landmarks are decoupled, so they are
                   excluded from the products). Used in lap 1 (mapping) and, when
                   freeze_map_ is false, in lap 2+ as well (continuous SLAM: pose
                   AND landmarks keep being corrected — the legacy behavior, which
                   refines the map but can let the global-rotation gauge drift).
                   The pose is corrected in every lap, including lap 1: the cones
                   anchor FAST-LIMO drift while the map is built. */
                const size_t na = 3 + 2 * this->landmark_count;

                MatrixXf Fx_k_a = MatrixXf::Zero(5, na);
                Fx_k_a.block(0,0,3,3) = Matrix3f::Identity();
                Fx_k_a.block(3, 3 + 2*k, 2, 2) = Matrix2f::Identity();

                MatrixXf Ht_k = ht_k * Fx_k_a;                  /* (2 x na) */
                auto P_a = this->P_.topLeftCorner(na, na);

                MatrixXf HtP = Ht_k * P_a;                      /* (2 x na) */
                Matrix2f S = (HtP * Ht_k.transpose()) + this->Q_;
                MatrixXf K = P_a * Ht_k.transpose() * S.inverse();

                this->x_.head(na).noalias() += (K * meas_diff);
                this->x_(2) = this->normalizeYaw(this->x_(2));
                P_a.noalias() -= K * HtP;
            }
            else
            {
                /* LAP 2+ (freeze_map_): RIGID-MAP localization update. The map is
                   now a fixed, known reference: treat landmark k as a constant and
                   update ONLY the 3-DoF pose. A clean pose-only update (rather than
                   zeroing the landmark rows of a full-state gain) keeps P
                   consistent — no asymmetric gain truncation — and fixes the
                   global-rotation gauge, so the map neither drifts nor rotates over
                   laps and the pose is not snapped by a contaminated gain. */
                Matrix<float,2,3> H_p = ht_k.leftCols(3);
                auto P_pp = this->P_.topLeftCorner(3,3);

                Matrix<float,2,3> HpP = H_p * P_pp;             /* (2x3) */
                Matrix2f S = (HpP * H_p.transpose()) + this->Q_;

                /* No gate here: the Mahalanobis association gate above already
                   validated this cone for lap 2+. */
                Matrix<float,3,2> K_p = P_pp * H_p.transpose() * S.inverse();  /* (3x2) */

                this->x_.head(3).noalias() += (K_p * meas_diff);
                this->x_(2) = this->normalizeYaw(this->x_(2));
                P_pp.noalias() -= (K_p * HpP);
            }

        }
#endif  /* !USE_CUDA: single-cone path */
    }

    /* ---- Batch (joint) measurement update --------------------------------
       Fuse ALL cones associated this scan in a single joint EKF update instead
       of only the last one. Applying M independent SEQUENTIAL updates per scan
       shrinks P ~M-fold and drives the filter overconfident until it diverges;
       a joint update instead linearises every cone at the SAME state and does
       exactly ONE covariance reduction, so it uses all the measurement
       information while staying consistent. Only the GPU build reaches this
       path (the CPU-only build uses the single-cone update above). */
#ifdef USE_CUDA
    if (!batch_obs.empty())
    {
        const size_t na = 3 + 2 * this->landmark_count;

        /* Per-cone Jacobian block (2x5: [pose | landmark k]), innovation, and an
           individual Mahalanobis gate. Survivors are stacked below. */
        struct ConeUpdate { size_t k; Matrix<float,2,5> Hb; Vector2f nu; };
        std::vector<ConeUpdate> ups;
        ups.reserve(batch_obs.size());

        for (const auto &obs : batch_obs)
        {
            const size_t kk = obs.first;

            Vector2f d = this->x_.segment(3 + 2*kk, 2) - Vector2f(this->x_(0), this->x_(1));
            float q = d.transpose() * d;
            q = (q == 0.0f) ? __FLT_MIN__ : q;
            const float sq = sqrt(q);

            Vector2f exp_z(sq, normalizeAngle(atan2(d(1), d(0)) - this->x_(2)));

            /* (1/q) * low-Jacobian, columns = [x, y, theta | m_kx, m_ky]. */
            Matrix<float,2,5> Hb;
            Hb << -sq*d(0), -sq*d(1),  0.0f,  sq*d(0),  sq*d(1),
                      d(1),    -d(0),    -q,    -d(1),     d(0);
            Hb /= q;

            Vector2f nu = obs.second - exp_z;
            nu(1) = normalizeAngle(nu(1));

            /* No gate here: lap-2+ associations were already validated by the
               Mahalanobis gate at association time (see correct() loop). */
            ups.push_back({kk, Hb, nu});
        }

        const size_t m = ups.size();
        if (m > 0)
        {
            /* Stacked innovation nu_all (2m) and block-diagonal measurement noise
               R (2m x 2m), common to both regimes. */
            VectorXf nu_all = VectorXf::Zero(2*m);
            MatrixXf R      = MatrixXf::Zero(2*m, 2*m);
            for (size_t r = 0; r < m; r++)
            {
                nu_all.segment(2*r, 2)  = ups[r].nu;
                R.block(2*r, 2*r, 2, 2) = this->Q_;
            }

            if (!(this->is_first_lap_completed && this->freeze_map_))
            {
                /* FULL-STATE joint update — build/refine the map. Used in lap 1
                   (mapping) and, when freeze_map_ is false, in lap 2+ too
                   (continuous SLAM). H is sparse: each cone touches pose + its
                   landmark. The pose is corrected in every lap, including lap 1
                   (the cones anchor FAST-LIMO drift while mapping). */
                auto P_a = this->P_.topLeftCorner(na, na);

                MatrixXf H = MatrixXf::Zero(2*m, na);
                for (size_t r = 0; r < m; r++)
                {
                    H.block(2*r, 0, 2, 3)              = ups[r].Hb.leftCols(3);
                    H.block(2*r, 3 + 2*ups[r].k, 2, 2) = ups[r].Hb.rightCols(2);
                }

                if (this->use_gpu_)
                {
                    /* Offload the O(na^2 m) update onto the resident device state.
                       H, R, nu_all are column-major contiguous (Eigen default),
                       so they map straight onto cuBLAS. */
                    if (!this->gpu_->batchUpdateFullState(
                            static_cast<int>(na), H.data(), R.data(),
                            nu_all.data(), static_cast<int>(2*m)))
                    {
                        std::cerr << "EKFOdom: GPU batch update failed: "
                                  << this->gpu_->lastError() << "\n";
                    }
                    /* device updated x_head and P (no yaw wrap on device): refresh
                       host mirrors, wrap yaw on host, push the pose back. */
                    this->syncFromDevice();
                    this->x_(2) = this->normalizeYaw(this->x_(2));
                    float p3[3] = { this->x_(0), this->x_(1), this->x_(2) };
                    this->gpu_->uploadPose3(p3);
                }
                else
                {
                    /* CPU fallback (GPU init failed): same joint update on the host. */
                    MatrixXf HP = H * P_a;                           /* (2m x na)  */
                    MatrixXf S  = (HP * H.transpose()) + R;          /* (2m x 2m)  */
                    MatrixXf K  = P_a * H.transpose() * S.inverse(); /* (na x 2m)  */

                    this->x_.head(na).noalias() += (K * nu_all);
                    this->x_(2) = this->normalizeYaw(this->x_(2));
                    P_a.noalias() -= K * HP;
                }
            }
            else
            {
                /* LAP 2+ (freeze_map_): RIGID-MAP joint localization. Treat the map
                   as fixed and update ONLY the pose, using just the pose columns of
                   each Jacobian. Consistent P (3x3 block), no gauge drift, no snap
                   from a truncated gain. */
                auto P_pp = this->P_.topLeftCorner(3,3);

                MatrixXf Hp = MatrixXf::Zero(2*m, 3);
                for (size_t r = 0; r < m; r++)
                {
                    Hp.block(2*r, 0, 2, 3) = ups[r].Hb.leftCols(3);
                }

                MatrixXf HpP = Hp * P_pp;                        /* (2m x 3)   */
                MatrixXf S   = (HpP * Hp.transpose()) + R;       /* (2m x 2m)  */
                MatrixXf K_p = P_pp * Hp.transpose() * S.inverse(); /* (3 x 2m) */

                this->x_.head(3).noalias() += (K_p * nu_all);
                this->x_(2) = this->normalizeYaw(this->x_(2));
                P_pp.noalias() -= (K_p * HpP);

                /* The rigid update touches ONLY the 3x3 pose block and the pose
                   mean — both valid host mirrors — so it is computed on the host
                   above and the result pushed back to the resident device P/x. */
                if (this->use_gpu_)
                {
                    float p3[3] = { this->x_(0), this->x_(1), this->x_(2) };
                    this->gpu_->uploadPose3(p3);
                    Matrix3f Pm = this->P_.topLeftCorner(3,3);
                    float P33[9] = { Pm(0,0), Pm(0,1), Pm(0,2),
                                     Pm(1,0), Pm(1,1), Pm(1,2),
                                     Pm(2,0), Pm(2,1), Pm(2,2) };
                    this->gpu_->uploadPoseBlock3x3(P33);
                }
            }
        }
    }
#endif  /* USE_CUDA: joint (batch) update path */

    return associated;
}

/* Return current filter state */
VectorXf EKFOdom::getState() const {
    return x_;
}

/* Return the diagonal (variances) of the 3x3 pose covariance block */
Vector3f EKFOdom::getPoseCovariance() const
{
    Vector3f pose_cov;
    for (size_t i = 0; i < 3; i++)
    {
        pose_cov(i) = this->P_(i,i);
    }

    return pose_cov;
}

void EKFOdom::setFirstLapCompleted(const bool first_lap_completed)
{
    this->is_first_lap_completed = first_lap_completed;
}
void EKFOdom::setFreezeMap(const bool enable)
{
    this->freeze_map_ = enable;
}
void EKFOdom::setAssocMahaGate(const float gate)
{
    this->assoc_maha_gate_ = gate;
}

void EKFOdom::setPose(const Vector3f pose)
{
    /* First FAST-LIMO frame: do NOT seed the EKF pose from LIMO. The EKF frame
       is anchored at the track origin (x_ starts at 0), and LIMO is used purely
       as a relative motion source, so here we only record the first frame as the
       reference for the next delta. */
    if (!this->is_pose_initialized)
    {
        this->prev_limo_pose_ = pose;
        this->is_pose_initialized = true;
        return;
    }

    /* Relative motion of FAST-LIMO since the previous frame, expressed in the
       previous LIMO body frame. */
    float dx = pose(0) - this->prev_limo_pose_(0);
    float dy = pose(1) - this->prev_limo_pose_(1);
    float prev_theta = this->prev_limo_pose_(2);

    float local_dx =  cos(prev_theta) * dx + sin(prev_theta) * dy;
    float local_dy = -sin(prev_theta) * dx + cos(prev_theta) * dy;
    float dtheta   =  normalizeAngle(pose(2) - prev_theta);

    /* World-frame displacement applied to the EKF pose (R(theta) * local). */
    float etheta = this->x_(2);
    float dwx = cos(etheta) * local_dx - sin(etheta) * local_dy;
    float dwy = sin(etheta) * local_dx + cos(etheta) * local_dy;

    /* Covariance propagation through the motion Jacobian (predict step):
       P <- G P G^T, with G = I except the 3x3 pose block. G couples the heading
       uncertainty into the position covariance and rotates the pose-landmark
       cross-covariances; without it the pose covariance would only grow on its
       diagonal (in setPoseCovariance) and corrections would distribute badly
       between x, y and theta. The additive process noise Q_motion is added
       afterwards in setPoseCovariance. Only the active sub-block is touched
       (inactive landmarks are decoupled). */
    Matrix3f G = Matrix3f::Identity();
    G(0,2) = -dwy;   /* d f_x / d theta = -sin(theta)*lx - cos(theta)*ly = -dwy */
    G(1,2) =  dwx;   /* d f_y / d theta =  cos(theta)*lx - sin(theta)*ly =  dwx */

    const size_t na = 3 + 2 * this->landmark_count;

    /* Additive motion process noise (the Q of the predict step). Inject pose
       uncertainty proportional to how far the car moved this step, so P cannot
       collapse to ~0 over the laps as cone corrections keep shrinking it.
       Without it the filter grows overconfident in its FAST-LIMO-propagated
       pose; then S = H P H^T + Q is tiny, the Mahalanobis association gate
       rejects the very cones that would correct the accumulating drift, and the
       map diverges in the late laps. Scaling with travelled distance/turn
       (rather than a constant per-scan term) mirrors how odometry error
       actually accumulates and is zero when the car is stopped. */
    const float dtrans = std::sqrt(local_dx*local_dx + local_dy*local_dy);
    const float add_xx  = this->q_motion_pos_ * dtrans;
    const float add_yy  = this->q_motion_pos_ * dtrans;
    const float add_yaw = this->q_motion_yaw_ * std::fabs(dtheta);

#ifdef USE_CUDA
    if (this->use_gpu_)
    {
        /* Propagate P pose strips + noise and compose the pose mean on the
           resident device state, then refresh the host mirrors. */
        const float Grm[9] = { G(0,0), G(0,1), G(0,2),
                               G(1,0), G(1,1), G(1,2),
                               G(2,0), G(2,1), G(2,2) };
        this->gpu_->motionPropagate(static_cast<int>(na), Grm,
                                    dwx, dwy, dtheta, add_xx, add_yy, add_yaw);
        this->syncFromDevice();
    }
    else
#endif
    {
        auto Pa = this->P_.topLeftCorner(na, na);
        Pa.topRows(3)  = (G * Pa.topRows(3)).eval();              /* rows: G * P    */
        Pa.leftCols(3) = (Pa.leftCols(3) * G.transpose()).eval(); /* cols: P * G^T  */

        this->P_(0,0) += add_xx;
        this->P_(1,1) += add_yy;
        this->P_(2,2) += add_yaw;

        /* Compose the relative motion onto the (cone-corrected) EKF pose mean, so
           the correction applied by correct() is preserved instead of overwritten. */
        this->x_(0) += dwx;
        this->x_(1) += dwy;
        this->x_(2)  = normalizeYaw(etheta + dtheta);
    }

    this->prev_limo_pose_ = pose;
}

void EKFOdom::setPoseCovariance(const Vector3f pos_cov)
{
    /* First frame: just record the reference covariance, add nothing. */
    if (!this->is_cov_initialized)
    {
        this->prev_limo_cov_ = pos_cov;
        this->is_cov_initialized = true;
        return;
    }

    /* Relative model: inflate the pose covariance only by the INCREMENT of
       LIMO's own covariance over this step, not by its absolute value. This
       mirrors using LIMO's relative motion for the mean (setPose). correct()
       then shrinks it back when cones anchor the pose. The increment is clamped
       to >= 0 since LIMO's covariance can occasionally drop (e.g. its own
       relocalisation), which must not shrink the EKF covariance here. */
#ifdef USE_CUDA
    if (this->use_gpu_)
    {
        float inc[3];
        for (size_t i = 0; i < 3; i++)
        {
            float d = pos_cov(i) - this->prev_limo_cov_(i);
            inc[i] = (d > 0.0f) ? d : 0.0f;
        }
        this->gpu_->addPoseCovDiag(inc[0], inc[1], inc[2]);
        float cov3[3];
        this->gpu_->downloadPoseCov3(cov3);
        this->P_(0,0) = cov3[0];
        this->P_(1,1) = cov3[1];
        this->P_(2,2) = cov3[2];
        this->prev_limo_cov_ = pos_cov;
        return;
    }
#endif
    for (size_t i = 0; i < 3; i++)
    {
        float d = pos_cov(i) - this->prev_limo_cov_(i);
        this->P_(i,i) += (d > 0.0f) ? d : 0.0f;
    }
    this->prev_limo_cov_ = pos_cov;
}

size_t EKFOdom::getActMappedLandmarks() const
{
    return this->landmark_count;
}

SignatureVector EKFOdom::getSignatures() const
{
    return this->s_;
}


/* Destructor */
EKFOdom::~EKFOdom()
{
    ;
}