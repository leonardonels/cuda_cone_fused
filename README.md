# cudaCONEFUSED

Cone-based EKF-SLAM that fuses **FAST-LIMO/FAST-LIO** odometry (used as a *relative*
motion source) with the track's **static cones** (used as landmarks) to anchor the
global drift of the LiDAR-inertial odometry.

Main output: an odometric pose `/Odometry` in the `track` frame which, from the second lap
onward, is realigned onto the cone map and therefore does not drift with FAST-LIMO.

> **Code structure.** The node was refactored into a layered architecture (ROS-free
> filter core, inbound adapters, backend Bridge, outbound publishers). See
> **§9 Code architecture** for the map of the components and **§10 Adding an input
> adapter** for how to plug in an IMU source.

---

## 1. ROS interface

### Subscribe
| Topic | Type | Use |
|---|---|---|
| `/clusters` (`cones_topic`) | `visualization_msgs/Marker` | cones observed by the LiDAR (clusters), in sensor frame |
| `/fast_limo/state` (`input_odom_topic`) | `nav_msgs/Odometry` | FAST-LIMO pose + covariance |
| `/planning/race_status` (`race_status_topic`) | `mmr_base/RaceStatus` | current lap (for the "first lap completed" logic) |

### Publish
| Topic | Type | Use |
|---|---|---|
| `/Odometry` (`output_odom_topic`) | `nav_msgs/Odometry` | pose estimated by the EKF (frame `track` → `imu_link`) + TF |
| `/slam/cones_positions` (`mapped_cones_topic`) | `visualization_msgs/Marker` | cone map (live during mapping, frozen afterwards) |
| `/slam/input_cones_debug` (`input_cones_debug_topic`) | `visualization_msgs/Marker` | **debug** (optional): raw input cones projected into the map frame with the current EKF pose (red markers), for an input-vs-map comparison |

### Frame
All state lives in the **`track`** frame, whose origin coincides with the vehicle's starting
pose. The pose is planar: `track → imu_link` with only XY translation and yaw rotation.

---

## 2. State representation

The state is the classic EKF-SLAM one (Thrun, *Probabilistic Robotics*, ch. 10) with
position landmarks (range/bearing):

$$
\mathbf{x} = \begin{bmatrix} x \\ y \\ \theta \\ m_{1,x} \\ m_{1,y} \\ \vdots \\ m_{N,x} \\ m_{N,y}\end{bmatrix}
\in \mathbb{R}^{2N+3}, \qquad
\mathbf{P} \in \mathbb{R}^{(2N+3)\times(2N+3)}
$$

- $(x,y,\theta)$ = 2D vehicle pose in the `track` frame.
- $(m_{j,x}, m_{j,y})$ = absolute position of cone $j$.
- $N$ = `N_CONES` = **400** (compile-time capacity, `ekf_odom.hpp`). The matrix size is
  fixed at $2N+3 = 803$; the number of actually mapped cones is `landmark_count` $\le N$.

**Initialization** (`EKFOdom` ctor):

$$
\mathbf{x} = \mathbf{0}, \qquad
\mathbf{P} = \begin{bmatrix} I_3 & 0 \\ 0 & \texttt{INF}\cdot I_{2N}\end{bmatrix},
\quad \texttt{INF}=10
$$

Landmarks start with "infinite" covariance and zero cross-covariance: they are **decoupled**
until observed. This is what allows working only on the active sub-block (§5).

---

## 3. Prediction step (relative motion from FAST-LIMO)

> **Design decision.** FAST-LIMO is **not** treated as absolute pose ground truth, but as
> **relative odometry**. The EKF state is anchored at the `track` origin; at each message only
> the *increment* of FAST-LIMO motion is applied. This way the cone correction stays in the
> state instead of being overwritten by the raw pose at every frame.

> **Post-refactor naming.** The old `setPose()` + `setPoseCovariance()` pair is now the single
> sensor-agnostic `EKFOdom::predict(const MotionIncrement&)` (§9). All the LIMO-specific
> bookkeeping below (the previous-frame reference, the first-frame seed, the quaternion→yaw
> conversion, the 0/7/35 covariance indexing) moved **out of the filter** into
> `LimoMotionAdapter`; the filter now consumes a ready-made `MotionIncrement` and never sees a
> ROS message. The math is unchanged — only the boundary moved.

`predict()` receives the increment built by `LimoMotionAdapter` from the FAST-LIMO pose
$p_k = (x_k^{L}, y_k^{L}, \theta_k^{L})$:

**First frame** ($\texttt{is\_pose\_initialized}=\text{false}$): only the reference is
recorded, the state stays at the origin.

$$
p_{\text{prev}} \leftarrow p_k, \qquad \mathbf{x}_{0:3} \;\text{unchanged} \;(=\mathbf{0})
$$

**Subsequent frames:** the relative displacement is computed in the body-frame of the
previous frame and composed onto the EKF pose (rigid composition $T_{\text{ekf}} \leftarrow
T_{\text{ekf}}\cdot T_{\text{prev}}^{-1}\cdot T_k$):

$$
\begin{aligned}
\Delta_x &= x_k^L - x_{\text{prev}}^L, \quad \Delta_y = y_k^L - y_{\text{prev}}^L \\
\ell_x &= \;\;\,\cos\theta_{\text{prev}}\,\Delta_x + \sin\theta_{\text{prev}}\,\Delta_y \\
\ell_y &= -\sin\theta_{\text{prev}}\,\Delta_x + \cos\theta_{\text{prev}}\,\Delta_y \\
\Delta_\theta &= \operatorname{wrap}(\theta_k^L - \theta_{\text{prev}}^L)
\end{aligned}
$$

$$
\begin{aligned}
x &\mathrel{+}= \cos\theta\,\ell_x - \sin\theta\,\ell_y \\
y &\mathrel{+}= \sin\theta\,\ell_x + \cos\theta\,\ell_y \\
\theta &\leftarrow \operatorname{wrap}(\theta + \Delta_\theta)
\end{aligned}
$$

**Covariance — propagation** (in `setPose`): the pose increment is also propagated through
the motion Jacobian $G$, exactly like an EKF *predict* step:

$$
\mathbf{P} \leftarrow G\,\mathbf{P}\,G^\top, \qquad
G = \begin{bmatrix} 1 & 0 & -d_{w,y} \\ 0 & 1 & \;\;d_{w,x} \\ 0 & 0 & 1\end{bmatrix} \;\text{(pose block; } I \text{ elsewhere)}
$$

where $(d_{w,x}, d_{w,y}) = R(\theta)\,(\ell_x,\ell_y)$ is the world-frame displacement. $G$
couples the heading uncertainty into the position covariance and **rotates the pose–landmark
cross-covariances**; without it, $\mathbf{P}$ would grow only on its diagonal and corrections
would distribute badly between $x,y,\theta$. Only the active sub-block is touched (inactive
landmarks are decoupled), so the cost is $O(n_a)$.

**Covariance — additive, motion-based process noise** (`predict`, `noises.proc_noise`):
right after the $G\mathbf{P}G^\top$ propagation, **pose uncertainty is injected proportional to
how far the vehicle moved** this step:

$$
P_{00} \mathrel{+}= q_{\text{pos}}\,\lVert(\ell_x,\ell_y)\rVert, \quad
P_{11} \mathrel{+}= q_{\text{pos}}\,\lVert(\ell_x,\ell_y)\rVert, \quad
P_{22} \mathrel{+}= q_{\text{yaw}}\,\lvert\Delta_\theta\rvert
$$

This is the $Q$ term of the *predict* step and it is the **critical** part: without it,
$\mathbf{P}$ can only **shrink** (cone corrections reduce it, nothing makes it grow back) and
collapses to $\sim 0$ after a few laps. With $\mathbf{P}$ collapsed the filter becomes
**overconfident** in the FAST-LIMO-propagated pose: $S = H\mathbf{P}H^\top + R$ becomes tiny
and the **Mahalanobis gate** (§4.4) starts **rejecting the very cones** that would correct the
accumulated drift → map divergence in the late laps. Scaling the noise with the
travelled distance/rotation (rather than a constant per-scan term) mirrors how odometry error
actually accumulates and is **zero when the vehicle is stopped**. Default `[0.05, 0.02]`;
`[0, 0]` disables it (historical behavior).

**Covariance — FAST-LIMO covariance increment** (`setPoseCovariance`): in addition, **only the
increment** of the covariance reported by FAST-LIMO is added (not the absolute value, which
summed at every high-rate frame would blow up $\mathbf{P}$):

$$
P_{ii} \mathrel{+}= \max\!\big(0,\; \sigma^{L}_{k,i} - \sigma^{L}_{k-1,i}\big), \quad i\in\{0,1,2\}
$$

The clamp to $\ge 0$ prevents an occasional drop in FAST-LIMO's covariance (e.g. its own
relocalization) from improperly shrinking $\mathbf{P}$: the shrinking must come **only** from
the cones, in `update()`. Together, the three steps realize the classic
$\mathbf{P}\leftarrow G\mathbf{P}G^\top + Q_{\text{motion}}$, where $Q_{\text{motion}}$ is the
sum of the motion-based term (above, the primary and reliable source) and the FAST-LIMO
increment.

> **Note.** `predict()` is sensor-agnostic: it absorbs any `MotionIncrement`, not just the
> LIMO one. A second motion source (e.g. an IMU integrating $v$, $\omega$ over $\Delta t$) plugs
> in by emitting its own increment from a new adapter — see §10. The relative LIMO increment
> above is currently the only wired motion source.

---

## 4. Correction step (cones)

> **Post-refactor naming.** `correct()` is now `EKFOdom::update(const std::vector<Observation>&)`
> (§9). The vehicle-frame→polar translation and the colour→signature mapping moved **out of the
> filter** into `ConeObsAdapter` (the latter via an injected `IColorClassifier` Strategy); the
> filter now consumes ready-made `Observation`s. The math is unchanged.

`update()` receives $M$ observations $z_i = (\rho_i, \phi_i, c_i)$ = range, bearing,
color, built by `ConeObsAdapter` from the cluster points:

$$
\rho_i = \sqrt{p_x^2 + p_y^2}, \qquad \phi_i = \operatorname{wrap}\!\big(\operatorname{atan2}(p_y, p_x)\big)
$$

### 4.1 Data association (nearest-neighbour)

For each observation the cone is projected into the global frame given the current pose:

$$
\hat{m}_i = \begin{bmatrix} x \\ y \end{bmatrix} + \rho_i \begin{bmatrix}\cos(\phi_i + \theta)\\ \sin(\phi_i + \theta)\end{bmatrix}
$$

and the nearest mapped landmark in Euclidean distance is found, index $k$, distance
$d_{\min}$. The association **decision**, however, depends on the lap:

- **Lap 1 (mapping):** fixed Euclidean gate. If $d_{\min} > \alpha$ (`min_new_cone_distance`)
  → new landmark ($m_k \leftarrow \hat{m}_i$, `landmark_count++`); otherwise → association to $k$.
- **From lap 2 (localization):** **Mahalanobis** gate on the nearest candidate $k$. The
  innovation $\nu$ and the **pose-only** innovation covariance $S = H_p P_{pp} H_p^\top + Q$
  (§4.3) are computed, and the observation is accepted only if $d^2 = \nu^\top S^{-1}\nu \le$
  `assoc_maha_gate`; otherwise it is discarded (fixed map, no new cones). This replaces the
  fixed Euclidean radius: the capture region **adapts to the uncertainty** ($\propto S$)
  instead of being a constant circle. This matters because, under heading drift, far cones get
  displaced beyond a fixed radius and the Euclidean NN would discard them → the correction
  stalls and the drift diverges; the Mahalanobis gate keeps them. It is the **only gating
  point** for lap 2+ (the update steps do not re-gate).

**Orange** cones ($c\in\{2,3\}$) are discarded.

### 4.2 Measurement model and Jacobian

For the associated landmark $k$, with $\delta = m_k - (x,y)^\top$ and $q = \delta^\top\delta$:

$$
\hat{z} = h(\mathbf{x}) = \begin{bmatrix}\sqrt{q}\\ \operatorname{wrap}\big(\operatorname{atan2}(\delta_y,\delta_x) - \theta\big)\end{bmatrix}
$$

"Low" Jacobian $2\times 5$ with respect to $[x,y,\theta,m_{k,x},m_{k,y}]$:

$$
{}^{low}H = \frac{1}{q}\begin{bmatrix}
-\sqrt{q}\,\delta_x & -\sqrt{q}\,\delta_y & 0 & \sqrt{q}\,\delta_x & \sqrt{q}\,\delta_y \\
\delta_y & -\delta_x & -q & -\delta_y & \delta_x
\end{bmatrix}
$$

mapped onto the state via $F_{x,k}$ (selects the 3 pose columns and the 2 of landmark
$k$): $H = {}^{low}H \, F_{x,k}$.

### 4.3 Kalman update

$$
S = H\,\mathbf{P}\,H^\top + R \;(2\times2), \qquad
K = \mathbf{P}\,H^\top S^{-1}, \qquad
\mathbf{x} \mathrel{+}= K\,(z - \hat z), \qquad
\mathbf{P} \leftarrow \mathbf{P} - K\,(H\mathbf{P})
$$

where $R$ = `meas_noise` (the measurement noise; see §6). The $\mathbf{P}$ update is written
as $\mathbf{P} - K(H\mathbf{P})$, algebraically identical to $(I-KH)\mathbf{P}$ but in $O(n_a^2)$
instead of $O(n^3)$ (§5).

**Innovation wrapping.** The bearing component of the innovation $\nu = z - \hat z$ is
normalized to $[-\pi,\pi]$: both $\phi_i$ and $\hat z_\theta$ are already normalized
individually, but their **difference** can straddle $\pm\pi$ (e.g. $+3.0-(-3.0)=6.0$
rad instead of $-0.28$). Without the wrapping, a cone re-observed across the angular
discontinuity — typical at lap closure — would inject a huge, spurious innovation that
"snaps" the pose.

### 4.4 Mahalanobis gate (association, from lap 2)

From lap 2, accepting an association goes through the normalized Mahalanobis distance,
which follows a chi-square with 2 degrees of freedom (range + bearing):

$$
d^2 = \nu^\top S^{-1} \nu, \qquad \nu = z - \hat z, \qquad S = H_p P_{pp} H_p^\top + Q
$$

If $d^2 >$ `assoc_maha_gate` (default `9.21`, 99% bound for 2 DoF) the observation is
inconsistent with the nearest landmark and is **discarded** (`continue`) — be it a **wrong
association** (e.g. at lap closure) or a cone that is too far. The gate is evaluated at
**association** time (§4.1) and is the only gating point for lap 2+: the update steps (single
cone and batch) do not re-do it. During lap 1 it does not apply (the pose is frozen,
$K_{0:3}=0$, and the map is still forming; the Euclidean gate `min_new_cone_distance` holds).
Since it can only **discard** observations, it never improperly shrinks $\mathbf{P}$.

### 4.5 Update mode: single cone vs batch (`batch_cone_update`)

By default the Kalman update (§4.3) is applied **only to the last cone** of the message
(`i == M-1`): cheap (one $O(n_a^2)$ update per scan) and stable, but it uses a single
measurement out of $M$ and depends on detection order. With `generic.batch_cone_update: true`
a **batch (joint)** path is enabled: during the loop **all** associations are collected and,
after the loop, **a single** joint update is applied.

$$
H = \begin{bmatrix} H_1 \\ \vdots \\ H_m \end{bmatrix}_{(2m\times n_a)}, \quad
\nu = \begin{bmatrix}\nu_1 \\ \vdots \\ \nu_m\end{bmatrix}, \quad
R = \operatorname{blkdiag}(Q,\dots,Q)_{(2m\times 2m)}
$$
$$
S = H\mathbf{P}H^\top + R, \qquad K = \mathbf{P}H^\top S^{-1}, \qquad
\mathbf{x}\mathrel{+}=K\nu, \qquad \mathbf{P}\leftarrow\mathbf{P}-K(H\mathbf{P})
$$

The cones in `batch_obs` have already been validated by the Mahalanobis gate at association
time (§4.4), and during lap 1 $K_{0:3}=0$ still holds. Key point: the fusion is **joint**, not
sequential. Applying $M$ separate updates per scan would shrink $\mathbf{P}$ by ~$M$ times,
making the filter overconfident until it diverges (observed in practice: the map blows up);
the joint update linearizes all cones at the **same** state and does **a single** covariance
reduction, using all the information while staying consistent. Cost: $S$ is $2m\times2m$ →
$O((2m)^3)$ for the inversion, negligible for $m\sim20$; the products stay $O(m\,n_a^2)$.

---

## 5. Design decisions and optimizations

### Two regimes: mapping (lap 1) → localization on a rigid map (from lap 2)
The anchor's strength depends on a mature map, and `correct()` uses **two different
formulations** of the update depending on the lap:

- **Lap 1** (`¬ is_first_lap_completed`): **full-state** EKF-SLAM update on the active
  sub-block (pose + landmarks), but with the **pose rows of the gain zeroed** ($K_{0:3}=0$).
  This way the cones **build and refine the map** while the pose follows FAST-LIMO in pure
  relative mode (pose corrections from a sparse map would be noisy). This pose freeze is the
  default and is controlled by `generic.freeze_pose_first_lap: true`. Setting it to **`false`**
  keeps the pose gain (no zeroing), so lap 1 runs **full EKF-SLAM**: the cones also correct the
  pose while mapping, anchoring FAST-LIMO drift as the map is built (most effective at loop
  closure, when you re-see the start cones). Use `false` when LIMO drifts noticeably within a
  single lap and bakes a **smeared/duplicated map** (re-seen cones drifting past
  `min_new_cone_distance` spawn duplicates instead of associating). The cost: a wrong data
  association in the still-sparse lap-1 map now corrupts the **pose**, not just adds a stray
  landmark — and lap 1 has no Mahalanobis gate (it uses the Euclidean new-cone gate), so the
  protection is weaker. Keep `true` if lap-1 LIMO is clean.
- **From lap 2** (`current_lap > 1`): **pose-only** update (localization on a known map). The
  landmark is treated as a **constant** and only the $3\times3$ pose block is updated:
  $$
  H_p \in \mathbb{R}^{2\times3}, \quad S = H_p P_{pp} H_p^\top + Q, \quad
  K_p = P_{pp} H_p^\top S^{-1}, \quad x_{0:3}\mathrel{+}=K_p\nu, \quad P_{pp}\leftarrow P_{pp}-K_p H_p P_{pp}
  $$
  This is preferable to zeroing the landmark rows of a full-state gain after the fact: that
  truncation makes $K \neq P H^\top S^{-1}$ and **breaks the consistency/symmetry of
  $\mathbf{P}$** (pose drift and snap over multiple laps). The pose-only update keeps
  $\mathbf{P}$ consistent **and** fixes the **gauge** degree of freedom: the global orientation
  of a SLAM map is observable only from the initial anchor, so continuing to update the
  landmarks too would let small biases **slowly rotate** the whole map after a few laps. From
  lap 2 no more cones are added.

**Optional: continuous SLAM (`freeze_map: false`).** The lap-2 rigid-map behavior above is the
default and is selected by `generic.freeze_map: true`. Setting it to **`false`** makes lap 2+
keep doing the **full-state** update (the same path as lap 1 but **without** the pose freeze),
so pose **and** landmark positions are corrected continuously — the legacy behavior. This
refines the map every lap, at the cost of re-opening the gauge: small biases can make the whole
map **slowly rotate** over many laps (exactly the failure the rigid mode was introduced to
prevent). No *new* cones are added after lap 1 in either mode; "continuous" only refines the
positions of cones already mapped in lap 1. Use `false` for A/B comparison or short runs where
map refinement matters more than long-horizon gauge stability; keep `true` for multi-lap races.

### Lap-1 → lap-2 handoff
During lap 1 the pose accumulates the **FAST-LIMO drift** and $P_{pp}$ **grows** (the pose is
frozen, so no correction shrinks it). At the transition to lap 2 the cones start correcting the
pose: a large $P_{pp}$ means a high Kalman gain, so the pose is pulled onto the map — but that
same correction sharply shrinks $P_{pp}$, so the next gain is smaller, and the realignment
**self-tapers over a handful of scans** rather than a single jump. No explicit ramp is needed,
and the motion-based process noise (§3) keeps $P_{pp}$ from then collapsing past a healthy floor.

> **Removed: gain-scaled anchor ramp.** An earlier `anchor_ramp_scans` knob scaled the pose
> correction by $\alpha=\min(1,n/N)$ over the first $N$ lap-2 scans. It was removed because
> scaling the gain by $\alpha$ is **not a valid partial Kalman update**: it scaled both
> $x\mathrel{+}=\alpha K\nu$ **and** $P\mathrel{-}=\alpha K(HP)$, so early in the ramp the pose
> stayed nearly open-loop while $P$ was barely reduced — $P$ ran away (observed: $P_{yy}$
> blowing up to ~2000 during the ramp window). If a softer handoff is ever needed, do it by
> **inflating and decaying the measurement noise** ($R_{\text{eff}}=R/\alpha$), which keeps
> $S$, $K$ and the $P$ update mutually consistent — not by scaling the gain.

### Active sub-block + $O(n_a^2)$ update
The matrices are sized at $n=803$ but the unmapped landmarks are `INF·I` decoupled:
their rows in $K$ are zero anyway. All of `correct()`'s operations therefore run on the
active block only, $n_a = 3 + 2\cdot\texttt{landmark\_count}$. Moreover the covariance
update uses the form $\mathbf{P}-K(H\mathbf{P})$ ($O(n_a^2)$) instead of the full product
$(I-KH)\mathbf{P}$ ($O(n^3)$). This is essential on a Jetson Orin (modest CPU IPC), and scales
well if `N_CONES` is increased.

### GPU (CUDA) backend (`USE_CUDA`)
Even with the $O(n_a^2)$ form, the **lap-1 full-state update** is the bottleneck on the Orin
Nano: `freeze_map` cannot help it (the cheap rigid pose-only path only applies from lap 2 —
lap 1 always runs the dense full-state update while the map grows to $n_a\approx 803$). It costs
~15 ms/scan on a single CPU core, and ~50–80 ms with the old 6-thread Eigen setting (at these
matrix sizes multithreaded GEMM mostly adds fork/join jitter). An optional **resident-state GPU
backend** offloads the heavy linear algebra:

- **Compile-time switch** (cuBLAS/cuSOLVER vs Eigen), *not* a runtime parameter. Built only with
  `-DUSE_CUDA=ON` (**default ON**). `OFF` → the original CPU-only build with no CUDA dependency —
  the debug backend, "like before".
- `CudaBackend` (`include/cuda_cone_fused/backend/cuda_backend.hpp`, `src/backend/cuda_backend.cu`),
  one of the two `IEkfBackend` peers (§9), keeps **$\mathbf{P}$
  ($n\times n$) and $\mathbf{x}$ resident on the device** for the whole run (allocated/initialised
  once; `thrust::device_vector`, RAII). The $n\times n$ covariance never crosses the bus.
- **On device** (cuBLAS + cuSOLVER): the batch full-state update
  ($HP=H\mathbf{P}$, $S=HP\,H^\top+R$, solve $S\,K^\top=HP$, $\mathbf{x}\mathrel{+}=K\nu$,
  $\mathbf{P}\mathrel{-}=K\,HP$), the `setPose` motion propagation ($\mathbf{P}$ pose strips
  through $G$, additive noise, pose compose), and the `setPoseCovariance` increments.
- **On CPU** (cheap, branchy): data association, color logic, marker building. `EKFOdom` keeps
  **host mirrors** of $\mathbf{x}$ (active head) and the $3\times3$ pose block of $\mathbf{P}$ —
  the only parts the CPU reads — refreshed by `syncFromBackend()`. Per scan only tiny slices move:
  $H/R/\nu$ up; pose, $3\times3$ cov and landmark means down.
- In GPU mode all corrections go through the **joint (batch)** path by necessity (the device holds
  the authoritative $\mathbf{P}$, so a single-cone CPU update would read a stale host copy) — it
  implies batch fusion regardless of `batch_cone_update`.

Verified on the Orin Nano Super (sm_87): GPU vs Eigen relative error ~$10^{-4}$; full-map
($n_a=803$) update **median ~1.3 ms** vs ~15 ms single-thread CPU. **Pin the clocks** for the
tail and throughput (target <1 ms): `sudo nvpmodel -m 0 && sudo jetson_clocks`.

```bash
colcon build --packages-select cuda_cone_fused                                # GPU backend (default)
colcon build --packages-select cuda_cone_fused --cmake-args -DUSE_CUDA=OFF    # CPU-only (debug)
```

### A single publication source
- `/Odometry` is published **only** by `fastLimoDataCallback` (FAST-LIMO rate, high), reading
  the EKF state `getState().head(3)` — **not** the raw LIMO pose.
- The cone markers are published **only** by `conesCallback`, to avoid two different sets
  (live map vs frozen) ending up on the same topic and making the cones "jitter" in RViz.

---

## 6. Parameter tuning (`config/config.yaml`)

> ℹ️ **Noise convention.** The two noises follow the standard EKF naming, and both are entered
> as **variance** (the values go straight onto the covariance diagonal — they are *not* squared
> internally, so enter $\sigma^2$, not $\sigma$):
> - `noises.meas_noise` → matrix $R$ (2×2), the **measurement noise** added to the innovation
>   covariance $S = H\mathbf{P}H^\top + R$ in `update()`.
> - `noises.proc_noise` → the **process noise** $Q$ of the `predict()` step (additive pose
>   covariance, scaled with travelled distance / turn).

| Parameter | Effect | How to tune |
|---|---|---|
| `noises.meas_noise` `[var_range, var_bearing]` | How much the filter trusts the cones. It is the $R$ in $S=HPH^\top+R$. **Large** → cones less credible → soft corrections, slow anchoring, less jitter. **Small** → aggressive corrections, fast anchoring, more jitter and risk of divergence under wrong associations. | Start at `[0.1, 0.1]`. If the cones "jitter"/the pose snaps from lap 2, **increase**. If the anchor is too slow to recover the drift, **decrease**. `var_bearing` in rad²: 0.1 ≈ σ≈18°, fairly wide. |
| `noises.proc_noise` `[q_pos, q_yaw]` | Additive, motion-based process noise (§3): $P_{xx},P_{yy} \mathrel{+}= q_{\text{pos}}\cdot\lVert\Delta\rVert$, $P_{\theta\theta}\mathrel{+}= q_{\text{yaw}}\cdot\lvert\Delta_\theta\rvert$ each step. It is the $Q$ of the *predict* step and prevents $\mathbf{P}$ from **collapsing** lap after lap (which would make the filter overconfident and have the Mahalanobis gate reject good cones → late divergence). **Large** → $\mathbf{P}$ stays higher → pose more reactive to the cones but more jitter. **Small/0** → $\mathbf{P}$ collapses, gate too selective, drift in the late laps. | Start at `[0.05, 0.02]` (`q_pos` in m²/m, `q_yaw` in rad²/rad). In the diagnostic log `Pyy` should settle to a **small but non-zero** value (~0.01–0.05) instead of decaying toward ~0.0008, and `corrected` should track `detected` even in laps 7–8. If drift persists, **raise**; if the pose gets jittery, **lower**. Note: `setPose` runs at the FAST-LIMO rate (~100 Hz), so the term accumulates over many steps between cone scans. |
| `noises.min_new_cone_distance` ($\alpha$) [m] | Euclidean threshold to create a new cone **only in lap 1** (§4.1). **Large** → fewer new cones (risk of merging distinct cones). **Small** → more cones (risk of duplicates from noise). From lap 2 association uses `assoc_maha_gate` instead. | Set it **below half** the minimum spacing between adjacent cones on track and **above** the per-frame position noise. |
| `generic.assoc_maha_gate` | Chi-square gate (2 DoF) for the Mahalanobis association from lap 2 (§4.1/§4.4): accept if $d^2=\nu^\top S^{-1}\nu \le$ gate. Replaces the fixed Euclidean radius with an **uncertainty-adaptive** capture region. **High** → keeps correcting under larger drift (but more risk of wrong associations); **low** → more aggressive rejection. | `5.99`=95%, `9.21`=99%, `13.8`=99.9%. If the pose "slides" and the red (debug) cones stop landing on the map on far stretches, **raise** the gate; if jumps from wrong associations appear, **lower** it. |
| `generic.cone_time_seen_th` | How many times a cone must be observed before entering the published/frozen map. **High** → cleaner map but cones that appear late. | `4` is a good compromise; raise if you see spurious cones, lower if the map populates too slowly. |
| `generic.cones_pub_for_debug` | `true` → publishes the EKF's **live** map even after lap 1 (debug). `false` → publishes the **frozen** map. | Keep `false` in a race. In debug, remember the live one moves (the associated cones are still corrected by the filter). |
| `generic.pub_input_cones_debug` | `true` → publishes on `input_cones_debug_topic` the raw input cones projected into the map frame with the current EKF pose (**red** markers). | Debug tool: the red ones should land on the mapped cones; if they "slide away" the pose is drifting. Keep `false` in a race. |
| `generic.batch_cone_update` | Correction mode (§4.5). `false` → update on the last cone/scan only (default, cheap, stable). `true` → **joint** update over all associated cones (more information, less "nervous" pose). | Leave `false` as baseline; set `true` for the A/B test. If the pose becomes unstable in batch, raise `noises.meas_noise` (the $R$ in $S$): the joint update is more aggressive because it fuses more measurements. |
| `generic.freeze_pose_first_lap` | Lap-1 pose handling (§5). `true` (default) → **freeze the pose**: trust FAST-LIMO completely, cones only build the map. `false` → **full SLAM in lap 1**: cones also correct the pose while mapping, so LIMO drift is anchored as the map is built (esp. at loop closure). | Keep `true` if lap-1 LIMO is clean. Switch to `false` if you see the map **smear/duplicate** during lap 1 (LIMO drift baked in). Watch out: with `false`, a wrong association in the sparse early map can corrupt the pose (no Mahalanobis gate in lap 1) — if lap 1 gets *worse*, revert. |
| `generic.freeze_map` | Map handling from lap 2 (§5). `true` (default) → **rigid map**: pose-only localization, landmarks fixed, gauge locked (no slow map rotation). `false` → **continuous SLAM**: keep correcting pose *and* landmark positions (legacy; refines the map but can let it slowly rotate over laps). No new cones are added after lap 1 either way. | Keep `true` for multi-lap races (gauge stability). Use `false` only for A/B tests or short runs where map refinement matters more than long-horizon stability. If you enable it and the map starts rotating after a few laps, that's the expected gauge drift — switch back to `true`. |
| `generic.is_colorblind` | `true` → all cones treated as yellow (color ignored in association). | Leave `true` if the color from the perceptor is unreliable. |
| `generic.is_skidpad_mission` | Skidpad mode: publishes pose only, no cone markers. | `false` for missions with cone mapping. |
| `generic.eigen_threads` | Eigen/OpenMP threads for the **CPU** linear algebra. Default `1`. At these matrix sizes multithreaded GEMM mostly adds fork/join jitter (p99 latency blows up at 6 threads on the 6-core SoC). With the GPU backend the heavy work is offloaded, so this only matters for a CPU-only build. | Keep `1`. Try `2` only when profiling a CPU-only (`-DUSE_CUDA=OFF`) build on a full-size map; never the old `6`. |
| `USE_CUDA` (compile-time, CMake) | **Build flag**, not a yaml param. `ON` (default) → GPU backend (needs CUDA toolkit). `OFF` → CPU-only build, no CUDA dependency (debug). See §5. | `colcon build --packages-select cuda_cone_fused --cmake-args -DUSE_CUDA=OFF` for the CPU debug build. |
| `N_CONES` (compile-time, `ekf_odom.hpp`) | Maximum landmark capacity. Increasing it enlarges the matrices (cost $\propto N$ on the inactive block, but `correct()` only works on $n_a$). | Raise if the track has more than ~400 cones. |

---

## 7. Assumptions

1. **Planar motion**: only $(x,y,\theta)$ are estimated; $z$, roll, pitch are ignored. Yaw is
   extracted from the FAST-LIMO quaternion.
2. **Locally accurate FAST-LIMO**: drift is small over the short term (one lap) and
   accumulates over the long term → the cones anchor it from lap 2.
3. **Static world**: cones do not move; they are a fixed global reference.
4. **Start at the origin**: the vehicle starts at the `track` origin ($\mathbf{x}=\mathbf{0}$);
   from FAST-LIMO **only the relative motion** is used, so its absolute frame is irrelevant.
5. **Range/bearing observations** from LiDAR clusters, referred to the vehicle origin (no
   sensor→vehicle extrinsic offset applied).
6. **A single measurement correction per cone frame**: currently the Kalman update is
   performed only on the **last** cone of the list (`i == M-1`); the others are only
   associated/mapped. See §8.
7. **Orange cones discarded** in the update.

---

## 8. Code architecture (post-refactor)

The node was refactored from a monolith into a **layered architecture**: translate at the
edges, estimate in a ROS-free core, orchestrate in the node. Every component touches ROS **or**
the domain, never both. This section is the map of *what lands where*.

```
[ROS topics]                                          [ROS topics / TF]
     │                                                       ▲
     ▼                                                       │
 INBOUND ADAPTERS              FILTER CORE              OUTBOUND PUBLISHERS
 (ROS msg → domain)            (ROS-free)               (domain → ROS msg)
 ─────────────────             ───────────────          ───────────────────
 LimoMotionAdapter ─predict─▶                  ──────▶  OdometryPublisher
 ConeObsAdapter    ─update──▶  EKFOdom         ──────▶  ConeMapPublisher
 (ImuMotionAdapter)─predict─▶  predict()/update()
                                      │
                                      ▼
                              IEkfBackend  (Bridge: CpuBackend | CudaBackend)

              THE NODE (ConeFusion) = orchestrator (Mediator / Facade):
        composition root · timestamp-ordered drain · input-owned publish policy
```

### Layout

| Layer | Files | Role |
|---|---|---|
| **Filter I/O types** | `ekf_types.hpp` | `MotionIncrement` / `Observation` — the *only* things crossing the filter boundary. ROS-free, stamp is a plain `uint64_t` ns. |
| **Filter core** | `ekf_odom.hpp` / `.cpp` | `EKFOdom`: `predict(MotionIncrement)` (was `setPose`+`setPoseCovariance`), `update(vector<Observation>)` (was `correct`), plus plain state getters. Keeps only small **host mirrors** (pose, active landmark means, 3×3 pose-cov); the authoritative state lives in the backend. |
| **Backend Bridge** | `backend/ekf_backend.hpp`, `backend/cpu_backend.*`, `backend/cuda_backend.*` | `IEkfBackend` with `CpuBackend` and `CudaBackend` as **peers**. The backend owns the authoritative full `x` and `dim×dim` `P`; the heavy joint update is dispatched to `batchUpdateFullState()`. `makeEkfBackend(dim, max_two_m, inf_init)` is the **single** place `USE_CUDA` is still consulted (§5). |
| **Inbound adapters** | `input/limo_motion_adapter.hpp`, `input/cone_obs_adapter.hpp`, `input/color_classifier.hpp` | One per sensor; `convert(ROS msg) → domain`. `LimoMotionAdapter` now owns *all* the LIMO bookkeeping; `ConeObsAdapter` does vehicle-frame→polar and delegates colour→signature to an injected `IColorClassifier` (Strategy + `makeColorClassifier` Factory). |
| **Command queue** | `input/filter_command.hpp` | `FilterCommand` (reified `Predict`/`Update`) + `CommandQueue`: a `multimap` sorted by stamp. `drainUpTo(horizon, filter)` applies commands in **timestamp order**, not arrival order (fixes the out-of-sequence-measurement problem). |
| **Outbound publishers** | `output/odometry_publisher.hpp`, `output/cone_map_publisher.*` | Mirror of an adapter: `publish(filter, stamp) → ROS msg`. `OdometryPublisher` (pose → `Odometry`+TF) is **triggered by FAST-LIMO**; `ConeMapPublisher` (map → `Marker`) is **triggered by cones** and owns the lap-freeze latch (it calls `setFirstLapCompleted`). Both stamp with the **source** timestamp, never `now()`. |
| **Node / orchestrator** | `cuda_cone_fused.hpp` / `.cpp`, `cuda_cone_fused_node.cpp` | `ConeFusion`: builds everything (composition root), converts each callback's message to a `FilterCommand`, enqueues, drains in stamp order, and lets each **main** input trigger its output. No translation or estimator logic lives here. |

### Key invariants the structure enforces

- **The filter imports zero ROS headers** on either boundary. If a translation creeps into
  `EKFOdom`, the split is wrong — push it into an adapter (input) or a publisher (output).
- **One update code path.** The `#ifdef USE_CUDA` braiding that used to interleave CPU
  single-cone and GPU batch updates is gone: both backends are peers behind `IEkfBackend`, and
  `update()` always dispatches the joint solve to `batchUpdateFullState()`.
- **`predict()` is never a silent no-op** — a missing motion model fails by *slow divergence*,
  not a loud error.
- **Main vs optional is an orchestration property, not an adapter one** (see §10).

---

## 9. Adding an input adapter (IMU)

Adapters are **structurally identical** — the asymmetry between a "main" and an "optional"
sensor lives one layer up, in the node callback, not in the adapter. To add a sensor you write
one small translation class and wire one callback. There are two adapter shapes:

- **Motion source** (IMU, wheel encoders) → `MotionIncrement` → `EKFOdom::predict()`.
- **Correction source** (a second cone detector) → `Observation`s → `EKFOdom::update()`.

And two orchestration roles:

- **Main input** — owns an output and triggers it (enqueue → drain → **publish**). LIMO and
  cones are the two mains today.
- **Optional input** — pure state enrichment; emits nothing (**enqueue only**, no drain, no
  publish). Its correction rides out on the next main publish (≤ ~11 ms later at 90 Hz).

### Recipe

1. **Write the adapter** in `include/cuda_cone_fused/input/`, mirroring an existing one. A
   motion adapter looks like `LimoMotionAdapter` (returns `std::optional<MotionIncrement>` so it
   can swallow its first frame while it seeds a reference); a correction adapter looks like
   `ConeObsAdapter` (returns `std::vector<Observation>`). **Keep all ROS↔domain translation and
   all sensor bookkeeping inside the adapter** — the filter must stay ROS-free.
2. **Declare the topic param** in `loadParameters()` (an `imu_topic` is already declared and
   plumbed as a member, just unused) and add the subscriber + an adapter member to `ConeFusion`.
3. **Wire the callback.** Convert → `FilterCommand::makePredict/​makeUpdate` → `cmd_queue_.push`.
   For a **main** input, then `drainQueue()` and call the owned publisher. For an **optional**
   input, **stop after the push** — do not drain or publish.

### IMU adapter (optional, motion) — worked example

```cpp
// include/cuda_cone_fused/input/imu_motion_adapter.hpp
class ImuMotionAdapter {
public:
  std::optional<MotionIncrement> convert(const sensor_msgs::msg::Imu& m) {
    // integrate gyro/accel over Δt since the previous sample → local dx,dy,dθ
    MotionIncrement inc;
    inc.delta_pose    = /* integrated local increment */;
    inc.cov_increment = /* MUST grow with Δt: cov ∝ Δt (see caveat 1) */;
    inc.stamp_ns      = toNs(m.header.stamp);
    return inc;          // nullopt on the first sample (seeds the Δt reference)
  }
};
```

```cpp
// in ConeFusion: subscribe to imu_topic, then —
void ConeFusion::imuCallback(sensor_msgs::msg::Imu::SharedPtr m) {
  if (auto inc = imu_adapter_.convert(*m))
    cmd_queue_.push(FilterCommand::makePredict(*inc));
  // OPTIONAL input → enqueue only. No drain, no publish: the IMU enriches the
  // state; its effect ships out on the next FAST-LIMO odom publish.
}
```

One **motion-model** requirement (a correctness invariant, not adapter style):

1. **IMU increment covariance must grow with integration time** (`cov ∝ Δt`). A fixed small
   covariance recreates the overconfidence / `P`-collapse failure that makes the Mahalanobis
   gate reject good cones (§3).

---

## 10. Data flow (summary)

Each callback **converts** its message to a `FilterCommand`, **enqueues** it, and (for a main
input) **drains in timestamp order** before publishing — see §9–§10.

```
/fast_limo/state ──▶ fastLimoDataCallback        (MAIN motion)
                      ├─ limo_adapter_.convert()  : Odometry → MotionIncrement (relative; nullopt on 1st frame)
                      ├─ cmd_queue_.push(Predict) ─┐
                      ├─ drainQueue()  ────────────┤ stamp-ordered: predict() / update()
                      │   └─ EKFOdom::predict()    │   x ⊕= incr; P[pose] ← G P Gᵀ + Q_motion(|Δ|) + ΔcovLIMO
                      └─ odom_out_.publish()       : /Odometry + TF from getState() (high rate, source-stamped)

/clusters ──────────▶ conesCallback              (MAIN correction)
                      ├─ cone_adapter_.convert()  : Marker → vector<Observation> (polar + IColorClassifier signature)
                      ├─ cmd_queue_.push(Update)  ─┐
                      ├─ drainQueue()  ────────────┤ stamp-ordered drain
                      │   └─ EKFOdom::update()     │   NN/Mahalanobis assoc + innovation wrap + joint Kalman update
                      │                            │   (lap 1: K[pose]=0 → mapping only; lap 2+: anchors pose, no new cones)
                      └─ cone_map_out_.publish()   : cones seen ≥ cone_time_seen_th; owns the lap-freeze latch
                                                     (→ setFirstLapCompleted on rollover)

/planning/race_status ─▶ raceStatusCallback       : stores current_lap (drives the freeze latch in ConeMapPublisher)

(optional inputs — e.g. IMU — would push to cmd_queue_ and STOP: no drain, no publish; §9)

            authoritative x, P  ◀────────────────  IEkfBackend (CpuBackend | CudaBackend)
            (host mirrors in EKFOdom: pose, active landmark means, 3×3 pose cov)
```
