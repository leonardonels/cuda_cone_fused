#include <cuda_cone_fused/ekf_odom.hpp>

#include <omp.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <utility>
#include <vector>

/* Constructor */
EKFOdom::EKFOdom(Vector2f meas_noise, Vector2f proc_noise, const float alpha, int eigen_threads)
{
    /* Eigen/OpenMP thread count. Default 1: at the matrix sizes here multithreaded
       GEMM mostly adds fork/join jitter (p99 latency blows up at 6 threads on the
       Orin's 6-core SoC) — and with the GPU backend the heavy work is offloaded
       anyway. Configurable via the `eigen_threads` node parameter for CPU tuning. */
    if (eigen_threads < 1) eigen_threads = 1;
    omp_set_num_threads(eigen_threads);
    Eigen::setNbThreads(eigen_threads);

    std::cerr << "Eigen threads: " << Eigen::nbThreads() << "\n";

    /* Initialize the host mirrors. The backend below initialises the
       authoritative state identically (x = 0, pose-cov block = I3), so these
       mirrors start consistent with it. */
    this->x_ = VectorXf::Zero(2 * N_CONES + 3);
    this->P_pose_ = Matrix3f::Identity();

    /* Initialize the (range, bearing) measurement-noise covariance R_. It is the
       additive term in the innovation covariance S = H P H^T + R_ used by every
       update(). Diagonal entries come from the `meas_noise` parameter and are
       used directly as VARIANCES (var_range [m^2], var_bearing [rad^2]) — they
       are NOT squared here, so the configured values are variances, not the
       standard deviations the old "sigma" wording implied. */
    this->R_ = Matrix2f::Identity();
    for (uint8_t i=0;i < meas_noise.size(); i++)
    {
        this->R_(i,i) = meas_noise(i);
    }

    std::cerr << "R_ : \n" << this->R_ << "\n";

    /* Initialize the (predict-step) process noise Q: additive pose covariance
       per unit of travelled distance / turned angle (see predict). */
    this->q_motion_pos_ = proc_noise(0);
    this->q_motion_yaw_ = proc_noise(1);

    std::cerr << "M_:\n" << this->q_motion_pos_ << "    0\n   0 " << this->q_motion_yaw_ << "\n";

    /* Initialize max new cone dist */
    this->max_new_cone_dist = alpha;

    /* Bring up the backend Bridge. The factory picks the CUDA peer when the build
       enables it and the device comes up, else the CPU peer; either way it
       initialises P = blockdiag(I3, INF*I) and x = 0, matching the mirrors above.
       max_two_m = 256 -> up to 128 cones/scan before an (automatic) GPU scratch
       grow (ignored by the CPU peer). */
    this->backend_ = makeEkfBackend(2 * N_CONES + 3, 256, INF);
}

void EKFOdom::syncFromBackend() {
    /* Pull the small host mirrors back from the authoritative backend state. */
    float pose3[3];
    this->backend_->downloadPose3(pose3);
    this->x_(0) = pose3[0];
    this->x_(1) = pose3[1];
    this->x_(2) = pose3[2];
    if (this->landmark_count > 0)
        this->backend_->downloadLandmarks(this->x_.data() + 3,
                                          static_cast<int>(this->landmark_count));
    float P33[9];
    this->backend_->downloadPoseBlock3x3(P33);
    this->P_pose_ << P33[0], P33[1], P33[2],
                     P33[3], P33[4], P33[5],
                     P33[6], P33[7], P33[8];
}

/* EKF correction step. Single code path for both backends: data association,
   Jacobian assembly and gating run here on the host mirrors; the O(na^2 m) joint
   solve is dispatched to the backend (CPU or GPU). */
size_t EKFOdom::update(const std::vector<Observation>& obs) {
    /* Diagnostic: # of observations that passed data association this scan. */
    size_t associated = 0;
    Vector2f tmp_cone;

    /* Reset the per-scan association diagnostics (see AssocStats). */
    this->assoc_stats_ = AssocStats{};
    double nn_sum = 0.0;
    size_t nn_samples = 0;   /* observations that HAD a nearest neighbour to measure against */

    /* LAP-1 effective new-vs-existing radius for this scan: the configured
       floor widened by the current pose uncertainty and capped below the
       physical cone spacing. Computed ONCE per scan (P does not change inside
       the association loop) — see the comment on new_cone_sigma_scale_. */
    const float pose_sigma = std::sqrt(std::max(0.0f,
                                 std::max(this->P_pose_(0,0), this->P_pose_(1,1))));
    float new_cone_radius = this->max_new_cone_dist
                          + this->new_cone_sigma_scale_ * pose_sigma;
    if (new_cone_radius > this->new_cone_dist_cap_)
        new_cone_radius = this->new_cone_dist_cap_;
    if (new_cone_radius < this->max_new_cone_dist)
        new_cone_radius = this->max_new_cone_dist;   /* cap below the floor: floor wins */
    this->assoc_stats_.gate_radius = new_cone_radius;

    /* Associations collected this scan for the joint (batch) update:
       (landmark index k, observed (range, bearing)). */
    std::vector<std::pair<size_t, Vector2f>> batch_obs;
    batch_obs.reserve(obs.size());

    for (const auto& o : obs)
    {
        const Vector3f& z = o.z;

        /* Discard orange cones */
        if ((z(2) == 2) || (z(2) == 3))
        {
            continue;
        }

        /* Reset min distance and k index */
        float min_dist = FLT_MAX;
        size_t k = 0;

        /* Compute expected NEW cone position */
        tmp_cone << this->x_(0), this->x_(1);
        tmp_cone = tmp_cone + (z(0) * Vector2f(cos(z(1) + this->x_(2)), sin(z(1) + this->x_(2))));

        /* Check if this cone has been seen before based on nearest neighbour (min distance) */
        for (size_t j = 0; j < this->landmark_count; j++)
        {
            float tempDistance = euclideanDistance(this->x_(j * 2 + 3), tmp_cone(0), this->x_(j * 2 + 4), tmp_cone(1));
            if (tempDistance < min_dist)
            {
                min_dist = tempDistance;
                k = j;
            }
        }

        this->assoc_stats_.observed++;
        if (this->landmark_count > 0)
        {
            /* Nearest-neighbour offset: how far the observation projected through
               the CURRENT pose lands from the map. This IS the association-level
               innovation (the red-vs-yellow offset on /slam/input_cones_debug).
               Its distribution against the lap-1 radius is the confirmation
               measurement — log it per scan and take the p95 offline. */
            nn_sum += min_dist;
            nn_samples++;
            if (min_dist > this->assoc_stats_.nn_max)
                this->assoc_stats_.nn_max = min_dist;
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
            qd = (qd == 0.0f) ? FLT_MIN : qd;
            const float sqd = sqrt(qd);

            Vector2f expz(sqd, normalizeAngle(atan2(dlt(1), dlt(0)) - this->x_(2)));
            Vector2f nu = Vector2f(z(0), z(1)) - expz;
            nu(1) = normalizeAngle(nu(1));

            /* Pose-only innovation covariance S = H_p P_pp H_p^T + R (2x2). Uses
               the FULL 3x3 pose covariance mirror (including off-diagonals). */
            Matrix<float,2,3> Hp;
            Hp << -sqd*dlt(0), -sqd*dlt(1),  0.0f,
                      dlt(1),     -dlt(0),    -qd;
            Hp /= qd;
            Matrix2f S = (Hp * this->P_pose_ * Hp.transpose()) + this->R_;

            const float d2 = nu.transpose() * S.inverse() * nu;
            if (d2 > this->assoc_maha_gate_)
            {
                this->assoc_stats_.dropped_gate++;
                continue;   /* not consistent with the nearest landmark -> drop */
            }
            this->s_(k).setColor(static_cast<uint32_t>(z(2)));
        }
        else
        {
            /* LAP 1: build the map with Euclidean nearest-neighbour against a
               P-SCALED radius. The metric form is kept (the map covariance is
               still immature, so a full Mahalanobis gate would be optimistic
               here) but the radius now widens with the pose uncertainty, which
               is what stops the "far from the map" -> "must be a new cone" ->
               "no correction" -> "drift further" loop from closing. See
               new_cone_sigma_scale_. */
            if (new_cone_radius < min_dist)
            {
                /* New cone */
                if (this->landmark_count >= N_CONES)
                {
                    /* Map is full: allocating another landmark would overrun the
                       fixed-size state (x_), backend P/x, and signature (s_)
                       buffers, all sized for N_CONES. Drop the observation.
                       This is a CLIFF, not a graceful degradation: from here on
                       EVERY further observation is dropped, including all the
                       good ones, and the filter goes permanently blind. The
                       counter below is what makes the node say so out loud. */
                    this->assoc_stats_.dropped_full++;
                    continue;
                }
                if (this->landmark_count > 0)
                {
                    /* Record how far this "new" cone landed from the nearest
                       thing already on the map (see AssocStats::new_nn_min). */
                    if (this->assoc_stats_.created == 0 ||
                        min_dist < this->assoc_stats_.new_nn_min)
                        this->assoc_stats_.new_nn_min = min_dist;
                    if (min_dist > this->assoc_stats_.new_nn_max)
                        this->assoc_stats_.new_nn_max = min_dist;
                }
                k = this->landmark_count;
                this->landmark_count++;
                this->assoc_stats_.created++;
                this->x_.segment(3 + (2*k), 2) = tmp_cone;
                /* Mirror the new landmark mean onto the authoritative backend
                   state. Its covariance is already INF there (pre-initialised). */
                this->backend_->insertLandmark(static_cast<int>(k), tmp_cone(0), tmp_cone(1));
                ColorLogic c;
                c.setColor(static_cast<uint32_t>(z(2)));
                this->s_(k) = c;
            }
            else
            {
                /* Existing cone: update color counter */
                this->s_(k).setColor(static_cast<uint32_t>(z(2)));
            }
        }

        associated++;   /* reached only by observations that passed association */
        batch_obs.emplace_back(k, Vector2f(z(0), z(1)));
    }

    this->assoc_stats_.associated = associated;
    if (nn_samples > 0)
        this->assoc_stats_.nn_mean =
            static_cast<float>(nn_sum / static_cast<double>(nn_samples));

    /* ---- Batch (joint) measurement update --------------------------------
       Fuse ALL cones associated this scan in a single joint EKF update instead
       of one-at-a-time. Applying M independent SEQUENTIAL updates per scan
       shrinks P ~M-fold and drives the filter overconfident until it diverges;
       a joint update instead linearises every cone at the SAME state and does
       exactly ONE covariance reduction, so it uses all the measurement
       information while staying consistent. */
    if (batch_obs.empty())
        return associated;

    const size_t na = 3 + 2 * this->landmark_count;

    /* Per-cone Jacobian block (2x5: [pose | landmark k]) and innovation.
       Survivors are stacked below. */
    struct ConeUpdate { size_t k; Matrix<float,2,5> Hb; Vector2f nu; };
    std::vector<ConeUpdate> ups;
    ups.reserve(batch_obs.size());

    for (const auto &ob : batch_obs)
    {
        const size_t kk = ob.first;

        Vector2f d = this->x_.segment(3 + 2*kk, 2) - Vector2f(this->x_(0), this->x_(1));
        float q = d.transpose() * d;
        q = (q == 0.0f) ? FLT_MIN : q;
        const float sq = sqrt(q);

        Vector2f exp_z(sq, normalizeAngle(atan2(d(1), d(0)) - this->x_(2)));

        /* (1/q) * low-Jacobian, columns = [x, y, theta | m_kx, m_ky]. */
        Matrix<float,2,5> Hb;
        Hb << -sq*d(0), -sq*d(1),  0.0f,  sq*d(0),  sq*d(1),
                  d(1),    -d(0),    -q,    -d(1),     d(0);
        Hb /= q;

        Vector2f nu = ob.second - exp_z;
        nu(1) = normalizeAngle(nu(1));

        /* No gate here: lap-2+ associations were already validated by the
           Mahalanobis gate at association time. */
        ups.push_back({kk, Hb, nu});
    }

    const size_t m = ups.size();
    if (m == 0)
        return associated;

    /* Stacked innovation nu_all (2m) and block-diagonal measurement noise
       R (2m x 2m), common to both regimes. */
    VectorXf nu_all = VectorXf::Zero(2*m);
    MatrixXf R      = MatrixXf::Zero(2*m, 2*m);
    for (size_t r = 0; r < m; r++)
    {
        nu_all.segment(2*r, 2)  = ups[r].nu;
        R.block(2*r, 2*r, 2, 2) = this->R_;
    }

    if (!(this->is_first_lap_completed && this->freeze_map_))
    {
        /* FULL-STATE joint update — build/refine the map. Used in lap 1 (mapping)
           and, when freeze_map_ is false, in lap 2+ too (continuous SLAM). H is
           sparse: each cone touches pose + its landmark. The pose is corrected in
           every lap, including lap 1 (the cones anchor FAST-LIMO drift while
           mapping). The O(na^2 m) solve runs in the backend. */
        MatrixXf H = MatrixXf::Zero(2*m, na);
        for (size_t r = 0; r < m; r++)
        {
            H.block(2*r, 0, 2, 3)              = ups[r].Hb.leftCols(3);
            H.block(2*r, 3 + 2*ups[r].k, 2, 2) = ups[r].Hb.rightCols(2);
        }

        /* H, R, nu_all are column-major contiguous (Eigen default), so the same
           buffers feed either backend peer without transposes. */
        if (!this->backend_->batchUpdateFullState(
                static_cast<int>(na), H.data(), R.data(),
                nu_all.data(), static_cast<int>(2*m)))
        {
            std::cerr << "EKFOdom: backend batch update failed: "
                      << this->backend_->lastError() << "\n";
        }

        /* The backend updated x_head and P (no yaw wrap): refresh host mirrors,
           wrap yaw on the host, and push the corrected pose back so the backend's
           pose mean stays consistent with the mirror. */
        this->syncFromBackend();
        this->x_(2) = this->normalizeYaw(this->x_(2));
        float p3[3] = { this->x_(0), this->x_(1), this->x_(2) };
        this->backend_->uploadPose3(p3);
    }
    else
    {
        /* LAP 2+ (freeze_map_): RIGID-MAP joint localization. Treat the map as
           fixed and update ONLY the pose, using just the pose columns of each
           Jacobian. It touches only the 3x3 pose block and the pose mean — both
           valid host mirrors — so it is computed here on the host and pushed back
           to the backend. Consistent P (3x3 block), no gauge drift, no snap from
           a truncated gain. */
        MatrixXf Hp = MatrixXf::Zero(2*m, 3);
        for (size_t r = 0; r < m; r++)
        {
            Hp.block(2*r, 0, 2, 3) = ups[r].Hb.leftCols(3);
        }

        MatrixXf HpP = Hp * this->P_pose_;              /* (2m x 3)   */
        MatrixXf S   = (HpP * Hp.transpose()) + R;      /* (2m x 2m)  */
        MatrixXf K_p = this->P_pose_ * Hp.transpose() * S.inverse(); /* (3 x 2m) */

        this->x_.head(3).noalias() += (K_p * nu_all);
        this->x_(2) = this->normalizeYaw(this->x_(2));
        this->P_pose_.noalias() -= (K_p * HpP);

        float p3[3] = { this->x_(0), this->x_(1), this->x_(2) };
        this->backend_->uploadPose3(p3);
        float P33[9] = { this->P_pose_(0,0), this->P_pose_(0,1), this->P_pose_(0,2),
                         this->P_pose_(1,0), this->P_pose_(1,1), this->P_pose_(1,2),
                         this->P_pose_(2,0), this->P_pose_(2,1), this->P_pose_(2,2) };
        this->backend_->uploadPoseBlock3x3(P33);
    }

    return associated;
}

/* Return current filter state (host mirror) */
VectorXf EKFOdom::getState() const {
    return x_;
}

/* Return the diagonal (variances) of the 3x3 pose covariance block */
Vector3f EKFOdom::getPoseCovariance() const
{
    Vector3f pose_cov;
    for (size_t i = 0; i < 3; i++)
    {
        pose_cov(i) = this->P_pose_(i,i);
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
void EKFOdom::setNewConeGate(const float sigma_scale, const float dist_cap)
{
    this->new_cone_sigma_scale_ = (sigma_scale > 0.0f) ? sigma_scale : 0.0f;
    this->new_cone_dist_cap_ = dist_cap;
}

void EKFOdom::predict(const MotionIncrement& inc)
{
    /* Local relative increment from the motion adapter (LIMO body frame). */
    const float local_dx = inc.delta_pose(0);
    const float local_dy = inc.delta_pose(1);
    const float dtheta   = inc.delta_pose(2);

    /* World-frame displacement applied to the EKF pose (R(theta) * local), using
       the filter's CURRENT heading — this is the bit that depends on filter state
       and so must live in the core, not the adapter. */
    const float etheta = this->x_(2);
    const float dwx = cos(etheta) * local_dx - sin(etheta) * local_dy;
    const float dwy = sin(etheta) * local_dx + cos(etheta) * local_dy;

    /* Covariance propagation through the motion Jacobian (predict step):
       P <- G P G^T, with G = I except the 3x3 pose block. G couples the heading
       uncertainty into the position covariance and rotates the pose-landmark
       cross-covariances. */
    Matrix3f G = Matrix3f::Identity();
    G(0,2) = -dwy;   /* d f_x / d theta = -dwy */
    G(1,2) =  dwx;   /* d f_y / d theta =  dwx */

    const size_t na = 3 + 2 * this->landmark_count;

    /* Additive motion process noise (the Q of the predict step). Inject pose
       uncertainty proportional to how far the car moved this step, so P cannot
       collapse to ~0 over the laps as cone corrections keep shrinking it
       (invariant 5). Scaling with travelled distance/turn mirrors how odometry
       error actually accumulates and is zero when the car is stopped. This is a
       MOTION-MODEL property of the filter, not an adapter concern. */
    const float dtrans = std::sqrt(local_dx*local_dx + local_dy*local_dy);
    const float add_xx  = this->q_motion_pos_ * dtrans;
    const float add_yy  = this->q_motion_pos_ * dtrans;
    const float add_yaw = this->q_motion_yaw_ * std::fabs(dtheta);

    /* Propagate P pose strips + motion noise and compose the pose mean on the
       authoritative backend state (single code path for CPU/GPU). */
    const float Grm[9] = { G(0,0), G(0,1), G(0,2),
                           G(1,0), G(1,1), G(1,2),
                           G(2,0), G(2,1), G(2,2) };
    this->backend_->motionPropagate(static_cast<int>(na), Grm,
                                    dwx, dwy, dtheta, add_xx, add_yy, add_yaw);

    /* Inflate the pose covariance by the adapter-supplied LIMO covariance
       increment (already clamped >= 0). correct()/update() shrinks it back when
       cones anchor the pose. */
    this->backend_->addPoseCovDiag(inc.cov_increment(0),
                                   inc.cov_increment(1),
                                   inc.cov_increment(2));

    this->syncFromBackend();
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
