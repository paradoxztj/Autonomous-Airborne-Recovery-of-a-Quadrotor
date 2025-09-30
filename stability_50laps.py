import numpy as np
import casadi as ca
import pandas as pd

# ===================== Dryden turbulence =====================

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


# ==========================================

mother_speed    = 10.0
child_speed_max = 16.0
child_acc_max   = 8.0
dt              = 0.1
target_distance = 5.0
z_mother        = 20.0
N               = 20    # NMPC horizon

laps            = 50
np.random.seed(0)


gamma_xyz = np.array([0.03, 0.03, 0.01])  # [1/m]
v_wind    = np.array([0.0, 0.0, 0.0])

def drag_accel_numpy(v, vwind, gamma_xyz, eps=0.0):

    vrel = v - vwind
    if eps > 0.0:
        return -gamma_xyz * np.sqrt(vrel**2 + eps**2) * vrel
    else:
        return -gamma_xyz * np.abs(vrel) * vrel


# ===================== 母机轨迹（圆形） =====================

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


# ==========================================

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
    return np.array(ref_traj).T  # shape: (6, N+1)


# ===================== APF =====================

apf_child_pos = np.array([0.0, 0.0, 0.0])
apf_child_vel = np.array([0.0, 0.0, 0.0])
k_att = 1.5
apf_wind = DrydenWind3D(dt=dt)

# ===================== NMPC（baseline） =====================

nx, nu = 6, 3
Q_base = np.diag([20, 20, 20, 5, 5, 5])
R_base = np.diag([0.5, 0.5, 0.5])

x = ca.MX.sym("x", nx)
u = ca.MX.sym("u", nu)
GAMMA = ca.MX.sym("GAMMA", 3)
VWIND = ca.MX.sym("VWIND", 3)

v_rel = ca.vertcat(x[3] - VWIND[0], x[4] - VWIND[1], x[5] - VWIND[2])
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

# baseline NMPC
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
    dx_k = x_k - X1_ref[:, k]
    cost1 += ca.mtimes([dx_k.T, Q_base, dx_k]) + ca.mtimes([u_k.T, R_base, u_k])
dx_term1 = X1[:, N] - X1_ref[:, N]
cost1 += ca.mtimes([dx_term1.T, Q_base, dx_term1])
opti1.minimize(cost1)
opti1.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0})

def nmpc1_controller(x_init, ref_traj):
    opti1.set_value(x1_0, x_init)
    opti1.set_value(X1_ref, ref_traj)
    opti1.set_value(GAMMA_p1, gamma_xyz)
    opti1.set_value(VWIND_p1, v_wind)
    try:
        sol = opti1.solve()
        return sol.value(U1[:, 0])
    except:
        return np.zeros(3)

nmpc1_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
nmpc1_wind = DrydenWind3D(dt=dt)

# ===================== NMPC=====================

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

    pos_err = x_k[0:3] - x_ref_k[0:3]
    vel_err = x_k[3:6] - x_ref_k[3:6]
    dist_sq = ca.sumsqr(pos_err)
    w_vel = 1 + 50 * ca.exp(-dist_sq / 0.25)

    cost2 += ca.mtimes([pos_err.T, Q_pos, pos_err]) \
          + w_vel * ca.mtimes([vel_err.T, Q_vel, vel_err]) \
          + ca.mtimes([u_k.T, R_v, u_k])

    x_next_k = f_drag(x_k, u_k, GAMMA_p2, VWIND_p2)
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
    opti2.set_value(GAMMA_p2, gamma_xyz)
    opti2.set_value(VWIND_p2, v_wind)
    try:
        sol = opti2.solve()
        return sol.value(U2[:, 0])
    except:
        return np.zeros(3)

nmpc2_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
nmpc2_wind = DrydenWind3D(dt=dt)


# ===================== Headless  =====================

total_frames = mother_path_len * laps

csv_cols = [
    't',
    'x_child','y_child','z_child',
    'vx_child','vy_child','vz_child',
    'tx_target','ty_target','tz_target',
    'x_mother','y_mother','z_mother',
    'vx_mother','vy_mother','vz_mother'
]

apf_log   = []
nmpc1_log = []
nmpc2_log = []

for frame in range(total_frames):
    t = frame * dt

    # 共享母机位置与速度
    mother_idx = frame % mother_path_len
    mother_pos = mother_path[mother_idx]
    mother_next_idx = (mother_idx + 1) % mother_path_len
    mother_vel = (mother_path[mother_next_idx] - mother_pos) / dt
    nrm = np.linalg.norm(mother_vel)
    direction = mother_vel / nrm if nrm > 1e-9 else np.array([1.0, 0.0, 0.0])

    # ===== APF =====
    target_pos_apf = mother_pos - direction * target_distance
    force_att = k_att * (target_pos_apf - apf_child_pos)
    des_speed = np.linalg.norm(force_att)
    desired_velocity = (force_att / des_speed * child_speed_max) if des_speed > child_speed_max else force_att
    acc_cmd_apf = (desired_velocity - apf_child_vel) / dt
    acc_norm = np.linalg.norm(acc_cmd_apf)
    if acc_norm > child_acc_max:
        acc_cmd_apf = acc_cmd_apf / acc_norm * child_acc_max

    a_drag_apf = drag_accel_numpy(apf_child_vel, v_wind, gamma_xyz)
    total_acc_apf = acc_cmd_apf + a_drag_apf + apf_wind.step()

    apf_child_pos += apf_child_vel * dt + 0.5 * total_acc_apf * dt**2
    apf_child_vel += total_acc_apf * dt

    apf_log.append([
        t,
        apf_child_pos[0], apf_child_pos[1], apf_child_pos[2],
        apf_child_vel[0], apf_child_vel[1], apf_child_vel[2],
        target_pos_apf[0], target_pos_apf[1], target_pos_apf[2],
        mother_pos[0], mother_pos[1], mother_pos[2],
        mother_vel[0], mother_vel[1], mother_vel[2]
    ])

    # ===== NMPC baseline =====
    ref1 = build_ref_traj(mother_idx)
    acc_cmd1 = nmpc1_controller(nmpc1_state, ref1)

    a_drag_n1 = drag_accel_numpy(nmpc1_state[3:6], v_wind, gamma_xyz)
    total_acc1 = acc_cmd1 + a_drag_n1 + nmpc1_wind.step()

    nmpc1_state[0:3] += nmpc1_state[3:6] * dt + 0.5 * total_acc1 * dt**2
    nmpc1_state[3:6] += total_acc1 * dt

    nmpc1_log.append([
        t,
        nmpc1_state[0], nmpc1_state[1], nmpc1_state[2],
        nmpc1_state[3], nmpc1_state[4], nmpc1_state[5],
        ref1[0,0], ref1[1,0], ref1[2,0],
        mother_pos[0], mother_pos[1], mother_pos[2],
        mother_vel[0], mother_vel[1], mother_vel[2]
    ])

    # ===== NMPC (velocity-penalty) =====
    ref2 = build_ref_traj(mother_idx)
    acc_cmd2 = nmpc2_controller(nmpc2_state, ref2)

    a_drag_n2 = drag_accel_numpy(nmpc2_state[3:6], v_wind, gamma_xyz)
    total_acc2 = acc_cmd2 + a_drag_n2 + nmpc2_wind.step()

    nmpc2_state[0:3] += nmpc2_state[3:6] * dt + 0.5 * total_acc2 * dt**2
    nmpc2_state[3:6] += total_acc2 * dt

    nmpc2_log.append([
        t,
        nmpc2_state[0], nmpc2_state[1], nmpc2_state[2],
        nmpc2_state[3], nmpc2_state[4], nmpc2_state[5],
        ref2[0,0], ref2[1,0], ref2[2,0],
        mother_pos[0], mother_pos[1], mother_pos[2],
        mother_vel[0], mother_vel[1], mother_vel[2]
    ])


# ===================== CSV =====================

pd.DataFrame(apf_log,   columns=csv_cols).to_csv("uav_50laps_APF.csv",    index=False)
pd.DataFrame(nmpc1_log, columns=csv_cols).to_csv("uav_50laps_NMPC.csv",   index=False)
pd.DataFrame(nmpc2_log, columns=csv_cols).to_csv("uav_50laps_NMPCv.csv",  index=False)

print("saved uav_50laps_APF.csv, uav_50laps_NMPC.csv, uav_50laps_NMPCv.csv")
