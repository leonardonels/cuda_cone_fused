# cudaCONEFUSED

Cone-based EKF-SLAM that fuses **FAST-LIMO/FAST-LIO** odometry (used as a *relative*
motion source) with the track's **static cones** (used as landmarks) to anchor the
global drift of the LiDAR-inertial odometry.

Main output: an odometric pose `/Odometry` in the `track` frame which, from the second lap
onward, is realigned onto the cone map and therefore does not drift with FAST-LIMO.

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

`setPose(p_k)` with $p_k = (x_k^{L}, y_k^{L}, \theta_k^{L})$ the FAST-LIMO pose:

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

**Covariance — additive, motion-based process noise** (`setPose`, `noises.motion_noise`):
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
**overconfident** in the FAST-LIMO-propagated pose: $S = H\mathbf{P}H^\top + Q$ becomes tiny
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
the cones, in `correct()`. Together, the three steps realize the classic
$\mathbf{P}\leftarrow G\mathbf{P}G^\top + Q_{\text{motion}}$, where $Q_{\text{motion}}$ is the
sum of the motion-based term (above, the primary and reliable source) and the FAST-LIMO
increment.

> **Note.** There is also a `predict(dt)` with a velocity/angular-velocity kinematic model
> ($v$, $\omega$), but it is **not wired** in the current pipeline (`setActVel`/`setActAngVel`
> are never called). Prediction is done entirely by the relative increment above. The method
> is left for possible future use with an IMU/encoder.

---

## 4. Correction step (cones)

`correct(z, M)` receives $M$ observations $z_i = (\rho_i, \phi_i, c_i)$ = range, bearing,
color, computed in `conesCallback` from the cluster points:

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
S = H\,\mathbf{P}\,H^\top + Q \;(2\times2), \qquad
K = \mathbf{P}\,H^\top S^{-1}, \qquad
\mathbf{x} \mathrel{+}= K\,(z - \hat z), \qquad
\mathbf{P} \leftarrow \mathbf{P} - K\,(H\mathbf{P})
$$

where $Q$ = `proc_noise` (see §6, non-standard naming). The $\mathbf{P}$ update is written
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
- `EkfCudaBackend` (`include/cuda_cone_fused/ekf_cuda.hpp`, `src/ekf_cuda.cu`) keeps **$\mathbf{P}$
  ($n\times n$) and $\mathbf{x}$ resident on the device** for the whole run (allocated/initialised
  once; `thrust::device_vector`, RAII). The $n\times n$ covariance never crosses the bus.
- **On device** (cuBLAS + cuSOLVER): the batch full-state update
  ($HP=H\mathbf{P}$, $S=HP\,H^\top+R$, solve $S\,K^\top=HP$, $\mathbf{x}\mathrel{+}=K\nu$,
  $\mathbf{P}\mathrel{-}=K\,HP$), the `setPose` motion propagation ($\mathbf{P}$ pose strips
  through $G$, additive noise, pose compose), and the `setPoseCovariance` increments.
- **On CPU** (cheap, branchy): data association, color logic, marker building. `EKFOdom` keeps
  **host mirrors** of $\mathbf{x}$ (active head) and the $3\times3$ pose block of $\mathbf{P}$ —
  the only parts the CPU reads — refreshed by `syncFromDevice()`. Per scan only tiny slices move:
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

> ⚠️ **Non-standard naming.** In the code the two noises are mapped the opposite way compared
> to the EKF convention, and they are used as **variance** (the square is commented out, so the
> entered value is $\sigma^2$, not $\sigma$, despite the comments saying "Sigma"):
> - `noises.proc_noise` → matrix $Q$ (2×2), **measurement innovation covariance** in `correct()`.
> - `noises.meas_noise` → matrix $R$ (3×3), process noise of `predict()` → **currently inert** (predict is not called).

| Parameter | Effect | How to tune |
|---|---|---|
| `noises.proc_noise` `[var_range, var_bearing]` | How much the filter trusts the cones. It is the $Q$ in $S=HPH^\top+Q$. **Large** → cones less credible → soft corrections, slow anchoring, less jitter. **Small** → aggressive corrections, fast anchoring, more jitter and risk of divergence under wrong associations. | Start at `[0.1, 0.1]`. If the cones "jitter"/the pose snaps from lap 2, **increase**. If the anchor is too slow to recover the drift, **decrease**. `var_bearing` in rad²: 0.1 ≈ σ≈18°, fairly wide. |
| `noises.motion_noise` `[q_pos, q_yaw]` | Additive, motion-based process noise (§3): $P_{xx},P_{yy} \mathrel{+}= q_{\text{pos}}\cdot\lVert\Delta\rVert$, $P_{\theta\theta}\mathrel{+}= q_{\text{yaw}}\cdot\lvert\Delta_\theta\rvert$ each step. It is the $Q$ of the *predict* step and prevents $\mathbf{P}$ from **collapsing** lap after lap (which would make the filter overconfident and have the Mahalanobis gate reject good cones → late divergence). **Large** → $\mathbf{P}$ stays higher → pose more reactive to the cones but more jitter. **Small/0** → $\mathbf{P}$ collapses, gate too selective, drift in the late laps. | Start at `[0.05, 0.02]` (`q_pos` in m²/m, `q_yaw` in rad²/rad). In the diagnostic log `Pyy` should settle to a **small but non-zero** value (~0.01–0.05) instead of decaying toward ~0.0008, and `corrected` should track `detected` even in laps 7–8. If drift persists, **raise**; if the pose gets jittery, **lower**. Note: `setPose` runs at the FAST-LIMO rate (~100 Hz), so the term accumulates over many steps between cone scans. |
| `noises.meas_noise` `[x,y,yaw]` | $R$ of `predict()`. **Inert** until `predict()` is wired. | Leave as is; relevant only if the velocity model is enabled. |
| `noises.min_new_cone_distance` ($\alpha$) [m] | Euclidean threshold to create a new cone **only in lap 1** (§4.1). **Large** → fewer new cones (risk of merging distinct cones). **Small** → more cones (risk of duplicates from noise). From lap 2 association uses `assoc_maha_gate` instead. | Set it **below half** the minimum spacing between adjacent cones on track and **above** the per-frame position noise. |
| `generic.assoc_maha_gate` | Chi-square gate (2 DoF) for the Mahalanobis association from lap 2 (§4.1/§4.4): accept if $d^2=\nu^\top S^{-1}\nu \le$ gate. Replaces the fixed Euclidean radius with an **uncertainty-adaptive** capture region. **High** → keeps correcting under larger drift (but more risk of wrong associations); **low** → more aggressive rejection. | `5.99`=95%, `9.21`=99%, `13.8`=99.9%. If the pose "slides" and the red (debug) cones stop landing on the map on far stretches, **raise** the gate; if jumps from wrong associations appear, **lower** it. |
| `generic.cone_time_seen_th` | How many times a cone must be observed before entering the published/frozen map. **High** → cleaner map but cones that appear late. | `4` is a good compromise; raise if you see spurious cones, lower if the map populates too slowly. |
| `generic.cones_pub_for_debug` | `true` → publishes the EKF's **live** map even after lap 1 (debug). `false` → publishes the **frozen** map. | Keep `false` in a race. In debug, remember the live one moves (the associated cones are still corrected by the filter). |
| `generic.pub_input_cones_debug` | `true` → publishes on `input_cones_debug_topic` the raw input cones projected into the map frame with the current EKF pose (**red** markers). | Debug tool: the red ones should land on the mapped cones; if they "slide away" the pose is drifting. Keep `false` in a race. |
| `generic.batch_cone_update` | Correction mode (§4.5). `false` → update on the last cone/scan only (default, cheap, stable). `true` → **joint** update over all associated cones (more information, less "nervous" pose). | Leave `false` as baseline; set `true` for the A/B test. If the pose becomes unstable in batch, raise `noises.proc_noise` (the $Q$ in $S$): the joint update is more aggressive because it fuses more measurements. |
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

## 8. Known limitations / TODO

- **Single-cone update per frame** (`i == M-1`, default): under-constrains the pose (a single
  fixed range/bearing measurement, ~2 DoF out of 3), depends on detection order and makes the
  corrections "nervous". An alternative **batch (joint)** path is available behind the flag
  `generic.batch_cone_update` (§4.5) which fuses all of the frame's associations in a single
  update, reducing variance by ~$1/M$. ⚠️ A naive *sequential* update (M separate updates per
  scan) **diverges** — it shrinks $\mathbf{P}$ by ~$M$ times per scan, making the filter
  overconfident: that's why the fusion is done in **joint** form (a single covariance
  reduction), not sequentially.
- **Single-lap map** (default): with the rigid anchor from lap 2 (§5) the map is the one built
  in lap 1 **only** and is no longer refined. It's a deliberate trade-off (it fixes the gauge
  and eliminates the slow map rotation), but any lap-1 mapping errors remain: if lap 1 is noisy,
  it's better to improve detection/association upstream than to reopen the landmark corrections
  (which would reintroduce the rotational drift). This can be toggled with
  `generic.freeze_map: false` to re-enable continuous landmark refinement, accepting the gauge
  drift as the cost — see §5.
- **`predict()` disconnected**: the velocity model is dormant code.
- **Dead member `Fx_k`**: replaced by the local `Fx_k_a` in `correct()`; removable.
- **Inverted noise naming** and used as variance: see §6.

---

## 9. Data flow (summary)

```
/fast_limo/state ──▶ fastLimoDataCallback
                      ├─ setPose()            : x ⊕= relative LIMO increment   (predict)
                      │                          P[pose] ← G P Gᵀ + Q_motion(|Δ|)  (process noise)
                      ├─ setPoseCovariance()  : P[pose] += ΔcovLIMO  (clamp ≥0)
                      └─ updatePose()         : publishes /Odometry from getState() (high rate)

/clusters ──────────▶ conesCallback
                      ├─ correct()                 : NN association + innovation wrapping +
                      │                               Mahalanobis gate + Kalman update (active block)
                      │                               (lap 1: K[pose]=0 → mapping only;
                      │                                lap 2: full K → anchors the pose, no new cones)
                      └─ pubConesMarkers()         : publishes cones seen ≥ cone_time_seen_th times
                                                      (live map in debug/lap 1, or frozen)

/planning/race_status ─▶ raceStatusCallback   : current_lap → setFirstLapCompleted(lap>1)
```
