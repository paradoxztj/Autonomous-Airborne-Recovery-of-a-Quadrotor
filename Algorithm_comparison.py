import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import casadi as ca
import pandas as pd

#  Dryden turbulence

class DrydenWindModel1D:
    def __init__(self, dt, L=30.0, sigma=0.5):
        self.dt = dt
        self.L = L
        self.sigma = sigma
        self.state = 0.0

    def step(self):
        eta = np.random.normal(0, 1)
        dw = (-self.state / self.L + self.sigma * np.sqrt(2 / self.L) * eta) * self.dt
        self.state += dw
        return self.state

class DrydenWind3D:
    def __init__(self, dt, L_xyz=(533, 266, 266), sigma_xyz=(0.5, 0.5, 0.2)):
        self.wx = DrydenWindModel1D(dt, L_xyz[0], sigma_xyz[0])
        self.wy = DrydenWindModel1D(dt, L_xyz[1], sigma_xyz[1])
        self.wz = DrydenWindModel1D(dt, L_xyz[2], sigma_xyz[2])

    def step(self):
        return np.array([self.wx.step(), self.wy.step(), self.wz.step()])


# ===================== Simulation / Model Parameters =====================

mother_speed    = 10.0
child_speed_max = 16.0  # for APF
child_acc_max   = 8.0
dt              = 0.1
target_distance = 5.0
z_mother        = 20.0
N               = 20               # NMPC horizon
frames          = 2000

np.random.seed(0)

# === DRAG ADDED: aerodynamic drag parameters (axis-wise) ===
gamma_xyz = np.array([0.03, 0.03, 0.01])  #
v_wind    = np.array([0.0, 0.0, 0.0])     #

def drag_accel_numpy(v, vwind, gamma_xyz, eps=0.0):

    vrel = v - vwind
    if eps > 0.0:
        return -gamma_xyz * np.sqrt(vrel**2 + eps**2) * vrel
    else:
        return -gamma_xyz * np.abs(vrel) * vrel


#  Mother trajectory

def generate_circle_path(center_xy, radius, z, speed, dt):
    dtheta = speed * dt / radius
    thetas = np.arange(0.0, 2*np.pi, dtheta)  # CCW
    xs = center_xy[0] + radius * np.cos(thetas)
    ys = center_xy[1] + radius * np.sin(thetas)
    zs = np.full_like(thetas, z)
    path = np.stack([xs, ys, zs], axis=1)
    return path

mother_center = np.array([0.0, 0.0])
radius = 50.0
mother_path = generate_circle_path(mother_center, radius, z_mother, mother_speed, dt)
mother_path_len = len(mother_path)

# ===================== Recording =====================

def record_crossing(t, px, prev_px, cross_count, recording, file_saved,
                    rec_data, child_pos, child_vel, target_pos, mother_pos, mother_vel, filename):

    crossed = False
    if prev_px is not None:
        crossed = ((prev_px > 0 and px <= 0) or (prev_px < 0 and px >= 0))

    if crossed and not file_saved:
        df = pd.DataFrame(rec_data, columns=[
            't',
            'x_child','y_child','z_child',
            'vx_child','vy_child','vz_child',
            'tx_target','ty_target','tz_target',
            'x_mother','y_mother','z_mother',
            'vx_mother','vy_mother','vz_mother'
        ])
        df.to_csv(filename, index=False)
        print(f"saved {filename}")
        file_saved = True

    if not file_saved:
        rec_data.append([
            t,
            child_pos[0], child_pos[1], child_pos[2],
            child_vel[0], child_vel[1], child_vel[2],
            target_pos[0], target_pos[1], target_pos[2],
            mother_pos[0], mother_pos[1], mother_pos[2],
            mother_vel[0], mother_vel[1], mother_vel[2]
        ])

    return cross_count, recording, file_saved


# ===================== APF Controller State =====================

apf_child_pos = np.array([0.0, 0.0, 0.0])
apf_child_vel = np.array([0.0, 0.0, 0.0])
apf_child_path = [apf_child_pos.copy()]
apf_mother_path_hist = []
k_att = 1.5
apf_wind = DrydenWind3D(dt=dt)
# recording
apf_rec_data = []
apf_file_saved = False
apf_cross_count = 0
apf_prev_px = None
apf_recording = False


# ===================== NMPC (baseline) setup with drag =====================

nx, nu = 6, 3
Q_base = np.diag([20, 20, 20, 5, 5, 5])
R_base = np.diag([0.5, 0.5, 0.5])

# Symbols
x = ca.MX.sym("x", nx)
u = ca.MX.sym("u", nu)

# === DRAG ADDED in prediction model ===
GAMMA = ca.MX.sym("GAMMA", 3)   # gamma_x, gamma_y, gamma_z
VWIND = ca.MX.sym("VWIND", 3)   # background wind velocity (m/s)

v_rel = ca.vertcat(x[3] - VWIND[0], x[4] - VWIND[1], x[5] - VWIND[2])
# axis-wise |s|*s
adx = -GAMMA[0] * ca.fabs(v_rel[0]) * v_rel[0]
ady = -GAMMA[1] * ca.fabs(v_rel[1]) * v_rel[1]
adz = -GAMMA[2] * ca.fabs(v_rel[2]) * v_rel[2]
ax_total = u[0] + adx
ay_total = u[1] + ady
az_total = u[2] + adz

x_next_drag = ca.vertcat(
    x[0] + x[3] * dt + 0.5 * ax_total * dt**2,
    x[1] + x[4] * dt + 0.5 * ay_total * dt**2,
    x[2] + x[5] * dt + 0.5 * az_total * dt**2,
    x[3] + ax_total * dt,
    x[4] + ay_total * dt,
    x[5] + az_total * dt
)
f_drag = ca.Function("f_drag", [x, u, GAMMA, VWIND], [x_next_drag])

opti1 = ca.Opti()
X1 = opti1.variable(nx, N+1)
U1 = opti1.variable(nu, N)
X1_ref = opti1.parameter(nx, N+1)
x1_0 = opti1.parameter(nx)
GAMMA_p1 = opti1.parameter(3)
VWIND_p1 = opti1.parameter(3)

opti1.subject_to(X1[:, 0] == x1_0)

cost1 = 0
for k in range(N):
    x_k, u_k = X1[:, k], U1[:, k]
    x_next_k = f_drag(x_k, u_k, GAMMA_p1, VWIND_p1)
    opti1.subject_to(X1[:, k+1] == x_next_k)
    opti1.subject_to(opti1.bounded(-child_acc_max, u_k, child_acc_max))
    dx = x_k - X1_ref[:, k]
    cost1 += ca.mtimes([dx.T, Q_base, dx]) + ca.mtimes([u_k.T, R_base, u_k])
dx_term1 = X1[:, N] - X1_ref[:, N]
cost1 += ca.mtimes([dx_term1.T, Q_base, dx_term1])
opti1.minimize(cost1)
opti1.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0})

def nmpc1_controller(x_init, ref_traj):
    opti1.set_value(x1_0, x_init)
    opti1.set_value(X1_ref, ref_traj)
    # set drag params
    opti1.set_value(GAMMA_p1, gamma_xyz)
    opti1.set_value(VWIND_p1, v_wind)
    try:
        sol = opti1.solve()
        return sol.value(U1[:, 0])
    except:
        return np.zeros(3)

nmpc1_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
nmpc1_child_path = [nmpc1_state[:3].copy()]
nmpc1_mother_path_hist = []
nmpc1_wind = DrydenWind3D(dt=dt)
# recording
nmpc1_rec_data = []
nmpc1_file_saved = False
nmpc1_cross_count = 0
nmpc1_prev_px = None
nmpc1_recording = False


# ===================== NMPC (velocity-penalty) setup with drag =====================

Q_pos = np.diag([20, 20, 20])
Q_vel = np.diag([10, 10, 10])
R_v  = np.diag([0.5, 0.5, 0.5])

opti2 = ca.Opti()
X2 = opti2.variable(nx, N+1)
U2 = opti2.variable(nu, N)
X2_ref = opti2.parameter(nx, N+1)
x2_0 = opti2.parameter(nx)
GAMMA_p2 = opti2.parameter(3)
VWIND_p2 = opti2.parameter(3)

opti2.subject_to(X2[:, 0] == x2_0)

cost2 = 0
for k in range(N):
    x_k = X2[:, k]
    u_k = U2[:, k]
    x_ref_k = X2_ref[:, k]

    # distance-aware velocity penalty
    pos_err = x_k[0:3] - x_ref_k[0:3]
    vel_err = x_k[3:6] - x_ref_k[3:6]
    dist_sq = ca.sumsqr(pos_err)
    w_vel = 1 + 50 * ca.exp(-dist_sq / 0.25)

    cost2 += ca.mtimes([pos_err.T, Q_pos, pos_err]) \
          + w_vel * ca.mtimes([vel_err.T, Q_vel, vel_err]) \
          + ca.mtimes([u_k.T, R_v, u_k])

    x_next_k = f_drag(x_k, u_k, GAMMA_p2, VWIND_p2)  # === DRAG ADDED ===
    opti2.subject_to(X2[:, k+1] == x_next_k)
    opti2.subject_to(opti2.bounded(-child_acc_max, u_k, child_acc_max))

pos_err_T = X2[0:3, N] - X2_ref[0:3, N]
vel_err_T = X2[3:6, N] - X2_ref[3:6, N]
dist_T_sq = ca.sumsqr(pos_err_T)
w_vel_T = 1 + 50 * ca.exp(-dist_T_sq / 0.25)
cost2 += ca.mtimes([pos_err_T.T, Q_pos, pos_err_T]) \
      + w_vel_T * ca.mtimes([vel_err_T.T, Q_vel, vel_err_T])
opti2.minimize(cost2)
opti2.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0})

def nmpc2_controller(x_init, ref_traj):
    opti2.set_value(x2_0, x_init)
    opti2.set_value(X2_ref, ref_traj)
    # set drag params
    opti2.set_value(GAMMA_p2, gamma_xyz)
    opti2.set_value(VWIND_p2, v_wind)
    try:
        sol = opti2.solve()
        return sol.value(U2[:, 0])
    except:
        return np.zeros(3)

nmpc2_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
nmpc2_child_path = [nmpc2_state[:3].copy()]
nmpc2_mother_path_hist = []
nmpc2_wind = DrydenWind3D(dt=dt)
# recording
nmpc2_rec_data = []
nmpc2_file_saved = False
nmpc2_cross_count = 0
nmpc2_prev_px = None
nmpc2_recording = False

# ===================== Plot Setup =====================

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 18,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_xlim(-55, 55)
ax.set_ylim(-55, 55)
ax.set_zlim(0, 30)
ax.set_box_aspect([1, 1, 0.5])

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("APF vs NMPC vs NMPC-VP (with Dryden Wind + Quadratic Drag)")

mother_plot, = ax.plot([], [], [], 'ro', label="mothership")
mother_trail_plot, = ax.plot([], [], [], 'r--', linewidth=0.8, label="mothership trail")

# APF:
child_plot_apf, = ax.plot([], [], [], linestyle='None', marker='o', markersize=6, label="APF child UAV")
trail_plot_apf, = ax.plot([], [], [], linestyle='-', linewidth=1.0, label="APF trail")

# NMPC baseline:
child_plot_n1, = ax.plot([], [], [], linestyle='None', marker='^', markersize=7, label="NMPC child UAV")
trail_plot_n1, = ax.plot([], [], [], linestyle='-', linewidth=1.0, label="NMPC trail")

# NMPC-VP:
child_plot_n2, = ax.plot([], [], [], linestyle='None', marker='s', markersize=6, label="NMPC-VP child UAV")
trail_plot_n2, = ax.plot([], [], [], linestyle='-', linewidth=1.0, label="NMPC-VP trail")

trail_plot_apf.set_color('b')
child_plot_apf.set_color('b')

trail_plot_n1.set_color('g')
child_plot_n1.set_color('g')

trail_plot_n2.set_color('m')
child_plot_n2.set_color('m')

ax.legend(loc="upper left", ncols=2, fontsize=12)

all_artists = (
    mother_plot, mother_trail_plot,
    child_plot_apf, trail_plot_apf,
    child_plot_n1, trail_plot_n1,
    child_plot_n2, trail_plot_n2
)

# ===================== Reference Builder =====================

def build_ref_traj(mother_idx):
    ref_traj = []
    for i in range(N + 1):
        idx = (mother_idx + i) % mother_path_len
        pos = mother_path[idx]
        next_idx = (idx + 1) % mother_path_len
        direction = mother_path[next_idx] - pos
        n = np.linalg.norm(direction)
        direction = direction / n if n > 1e-3 else np.array([1.0, 0.0, 0.0])
        target_pos = pos - direction * target_distance
        target_vel = direction * mother_speed
        ref_traj.append(np.concatenate([target_pos, target_vel]))
    return np.array(ref_traj).T

# ===================== Animation Update =====================

def update(frame):
    global apf_child_pos, apf_child_vel, apf_file_saved, apf_cross_count, apf_prev_px, apf_recording
    global nmpc1_state, nmpc1_file_saved, nmpc1_cross_count, nmpc1_prev_px, nmpc1_recording
    global nmpc2_state, nmpc2_file_saved, nmpc2_cross_count, nmpc2_prev_px, nmpc2_recording

    # Shared mother pose & velocity
    mother_idx = frame % mother_path_len
    mother_pos = mother_path[mother_idx]
    mother_next_idx = (mother_idx + 1) % mother_path_len
    mother_vel = (mother_path[mother_next_idx] - mother_pos) / dt
    nrm = np.linalg.norm(mother_vel)
    direction = mother_vel / nrm if nrm > 1e-9 else np.array([1.0, 0.0, 0.0])

    # ========== APF ==========
    target_pos_apf = mother_pos - direction * target_distance
    force_att = k_att * (target_pos_apf - apf_child_pos)
    des_speed = np.linalg.norm(force_att)
    desired_velocity = force_att / des_speed * child_speed_max if des_speed > child_speed_max else force_att
    acc_cmd_apf = (desired_velocity - apf_child_vel) / dt
    acc_norm = np.linalg.norm(acc_cmd_apf)
    if acc_norm > child_acc_max:
        acc_cmd_apf = acc_cmd_apf / acc_norm * child_acc_max

    # === DRAG ADDED in simulation ===
    a_drag_apf = drag_accel_numpy(apf_child_vel, v_wind, gamma_xyz)
    total_acc = acc_cmd_apf + a_drag_apf + apf_wind.step()

    apf_child_pos += apf_child_vel * dt + 0.5 * total_acc * dt**2
    apf_child_vel += total_acc * dt

    apf_child_path.append(apf_child_pos.copy())
    apf_mother_path_hist.append(mother_pos.copy())

    apf_cross_count, apf_recording, apf_file_saved = record_crossing(
        frame*dt, apf_child_pos[0], apf_prev_px, apf_cross_count, apf_recording, apf_file_saved,
        apf_rec_data, apf_child_pos, apf_child_vel, target_pos_apf, mother_pos, mother_vel,
        "uav_cross_APF.csv"
    )
    apf_prev_px = apf_child_pos[0]

    # ========== NMPC (baseline) ==========
    ref1 = build_ref_traj(mother_idx)
    acc_cmd1 = nmpc1_controller(nmpc1_state, ref1)

    a_drag_n1 = drag_accel_numpy(nmpc1_state[3:6], v_wind, gamma_xyz)  # === DRAG ADDED ===
    total_acc1 = acc_cmd1 + a_drag_n1 + nmpc1_wind.step()

    nmpc1_state[0:3] += nmpc1_state[3:6] * dt + 0.5 * total_acc1 * dt**2
    nmpc1_state[3:6] += total_acc1 * dt

    nmpc1_child_path.append(nmpc1_state[:3].copy())
    nmpc1_mother_path_hist.append(mother_pos.copy())

    nmpc1_cross_count, nmpc1_recording, nmpc1_file_saved = record_crossing(
        frame*dt, nmpc1_state[0], nmpc1_prev_px, nmpc1_cross_count, nmpc1_recording, nmpc1_file_saved,
        nmpc1_rec_data, nmpc1_state[:3], nmpc1_state[3:6], ref1[0:3, 0], mother_pos, mother_vel,
        "uav_cross_NMPC.csv"
    )
    nmpc1_prev_px = nmpc1_state[0]

    # ========== NMPC (velocity-penalty) ==========
    ref2 = build_ref_traj(mother_idx)
    acc_cmd2 = nmpc2_controller(nmpc2_state, ref2)

    a_drag_n2 = drag_accel_numpy(nmpc2_state[3:6], v_wind, gamma_xyz)  # === DRAG ADDED ===
    total_acc2 = acc_cmd2 + a_drag_n2 + nmpc2_wind.step()

    nmpc2_state[0:3] += nmpc2_state[3:6] * dt + 0.5 * total_acc2 * dt**2
    nmpc2_state[3:6] += total_acc2 * dt

    nmpc2_child_path.append(nmpc2_state[:3].copy())
    nmpc2_mother_path_hist.append(mother_pos.copy())

    nmpc2_cross_count, nmpc2_recording, nmpc2_file_saved = record_crossing(
        frame*dt, nmpc2_state[0], nmpc2_prev_px, nmpc2_cross_count, nmpc2_recording, nmpc2_file_saved,
        nmpc2_rec_data, nmpc2_state[:3], nmpc2_state[3:6], ref2[0:3, 0], mother_pos, mother_vel,
        "uav_cross_NMPCv.csv"
    )
    nmpc2_prev_px = nmpc2_state[0]

    # ==================== Draw ====================

    mother_plot.set_data([mother_pos[0]], [mother_pos[1]])
    mother_plot.set_3d_properties([mother_pos[2]])
    trail_m = np.array(apf_mother_path_hist)
    mother_trail_plot.set_data(trail_m[:, 0], trail_m[:, 1])
    mother_trail_plot.set_3d_properties(trail_m[:, 2])

    # APF
    child_plot_apf.set_data([apf_child_pos[0]], [apf_child_pos[1]])
    child_plot_apf.set_3d_properties([apf_child_pos[2]])
    trail_apf = np.array(apf_child_path)
    trail_plot_apf.set_data(trail_apf[:, 0], trail_apf[:, 1])
    trail_plot_apf.set_3d_properties(trail_apf[:, 2])

    # NMPC baseline
    child_plot_n1.set_data([nmpc1_state[0]], [nmpc1_state[1]])
    child_plot_n1.set_3d_properties([nmpc1_state[2]])
    trail_n1 = np.array(nmpc1_child_path)
    trail_plot_n1.set_data(trail_n1[:, 0], trail_n1[:, 1])
    trail_plot_n1.set_3d_properties(trail_n1[:, 2])

    # NMPC-VP
    child_plot_n2.set_data([nmpc2_state[0]], [nmpc2_state[1]])
    child_plot_n2.set_3d_properties([nmpc2_state[2]])
    trail_n2 = np.array(nmpc2_child_path)
    trail_plot_n2.set_data(trail_n2[:, 0], trail_n2[:, 1])
    trail_plot_n2.set_3d_properties(trail_n2[:, 2])

    return all_artists


# ===================== Run Animation =====================

ani = FuncAnimation(fig, update, frames=frames, interval=dt*1000, blit=True)
plt.tight_layout()
plt.show()
