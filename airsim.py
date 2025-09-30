import airsim
import numpy as np
import time
import threading
import cv2
import math
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import KalmanFilter
import casadi as ca
import csv

# ======
target_distance = 4
z_mother = -30
z_subdrone = -20
dt = 0.1
child_speed_max = 16.0  #
child_acc_max = 48
k_att = 1.5
vehicle_mother = "Drone1"
vehicle_child = "Drone2"
DEBUG_PRINT = True


V_MAX_NO_LIMIT = 1e6

# ======
circle_center = np.array([0.0, 0.0])
circle_radius = 100.0
speed_mother = 10.0
circle_resolution = 360

# === PD ===
Kp_lat = 0.4
Kd_lat = 0.02
vy_lat_max = 4.0
x_deadband = 0.02
pd_timeout = 0.3


_shared_lock = threading.Lock()
_shared_ex = 0.0
_shared_ex_time = 0.0
_shared_valid = False

# ======
_traj_lock = threading.Lock()
_traj_log = []  # [t_unix, mother_x, mother_y, mother_z, child_x, child_y, child_z]
_recording = False

# ======
_exit_event = threading.Event()


# ======
def circle_point(theta):
    x = circle_center[0] + circle_radius * math.cos(theta)
    y = circle_center[1] + circle_radius * math.sin(theta)
    return x, y


def fly_mother():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    x0, y0 = circle_point(0.0)
    client.moveToPositionAsync(x0, y0, z_mother, speed_mother,
                               drivetrain=airsim.DrivetrainType.ForwardOnly,
                               yaw_mode=airsim.YawMode(False, 0),
                               vehicle_name=vehicle_mother).join()
    theta = 0.0
    dtheta = 2 * math.pi / circle_resolution
    while not _exit_event.is_set():
        theta = (theta + dtheta) % (2 * math.pi)
        x, y = circle_point(theta)
        client.moveToPositionAsync(x, y, z_mother, speed_mother,
                                   drivetrain=airsim.DrivetrainType.ForwardOnly,
                                   yaw_mode=airsim.YawMode(False, 0),
                                   vehicle_name=vehicle_mother).join()



# =========================
class NMPCController:


    def __init__(self, N=15, dt=0.1, v_max=16.0, a_max=48.0,
                 alpha=50.0, beta=0.25):
        self.N = N
        self.dt = dt
        self.v_max_default = v_max
        self.a_max = a_max
        self.alpha = alpha
        self.beta = beta
        self._build()

    def _build(self):
        N, dt = self.N, self.dt
        alpha, beta = self.alpha, self.beta

        opti = ca.Opti()


        X = opti.variable(6, N + 1)  # [x,y,z,vx,vy,vz]
        U = opti.variable(3, N)  # [ax,ay,az]


        X0 = opti.parameter(6)  #
        Pref_pos = opti.parameter(3, N + 1)
        Pref_vel = opti.parameter(3, N + 1)
        VMAX = opti.parameter(1)  #


        opti.subject_to(X[:, 0] == X0)


        for k in range(N):
            xk, yk, zk = X[0, k], X[1, k], X[2, k]
            vxk, vyk, vzk = X[3, k], X[4, k], X[5, k]
            axk, ayk, azk = U[0, k], U[1, k], U[2, k]


            opti.subject_to(X[0, k + 1] == xk + vxk * dt + 0.5 * axk * dt * dt)
            opti.subject_to(X[1, k + 1] == yk + vyk * dt + 0.5 * ayk * dt * dt)
            opti.subject_to(X[2, k + 1] == zk + vzk * dt + 0.5 * azk * dt * dt)
            opti.subject_to(X[3, k + 1] == vxk + axk * dt)
            opti.subject_to(X[4, k + 1] == vyk + ayk * dt)
            opti.subject_to(X[5, k + 1] == vzk + azk * dt)


            opti.subject_to(ca.fabs(axk) <= self.a_max)
            opti.subject_to(ca.fabs(ayk) <= self.a_max)
            opti.subject_to(ca.fabs(azk) <= self.a_max)

            opti.subject_to(ca.fabs(X[3, k + 1]) <= VMAX)
            opti.subject_to(ca.fabs(X[4, k + 1]) <= VMAX)
            opti.subject_to(ca.fabs(X[5, k + 1]) <= VMAX)


        Qp = ca.diag(ca.DM([4.0, 4.0, 2.0]))
        Qv = ca.diag(ca.DM([1.5, 1.5, 1.0]))
        Ru = ca.diag(ca.DM([0.02, 0.02, 0.02]))
        Rdu = ca.diag(ca.DM([0.01, 0.01, 0.01]))


        cost = 0
        for k in range(N + 1):
            pos_err = X[0:3, k] - Pref_pos[:, k]
            vel_err = X[3:6, k] - Pref_vel[:, k]
            dist_sq = ca.sumsqr(pos_err)
            w_vel = 1 + self.alpha * ca.exp(-dist_sq / self.beta)

            cost += ca.mtimes([pos_err.T, Qp, pos_err]) \
                    + w_vel * ca.mtimes([vel_err.T, Qv, vel_err])

            if k < N:
                uk = U[:, k]
                cost += ca.mtimes([uk.T, Ru, uk])
                if k > 0:
                    duk = U[:, k] - U[:, k - 1]
                    cost += ca.mtimes([duk.T, Rdu, duk])

        opti.minimize(cost)
        opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "max_iter": 100})


        self.opti = opti
        self.X, self.U = X, U
        self.X0, self.Pref_pos, self.Pref_vel = X0, Pref_pos, Pref_vel
        self.VMAX = VMAX
        self._u_init = np.zeros((3, self.N))
        self._x_init = None

    def solve(self, x0, ref_pos_traj, ref_vel_traj, vmax_now):
        opti = self.opti
        opti.set_value(self.X0, x0)
        opti.set_value(self.Pref_pos, ref_pos_traj)
        opti.set_value(self.Pref_vel, ref_vel_traj)
        opti.set_value(self.VMAX, vmax_now)

        # warm start
        if self._x_init is None:
            Xinit = np.tile(x0.reshape(-1, 1), (1, self.N + 1))
            opti.set_initial(self.X, Xinit)
        else:
            opti.set_initial(self.X, self._x_init)
        opti.set_initial(self.U, self._u_init)

        try:
            sol = opti.solve()
            Xsol = sol.value(self.X)
            Usol = sol.value(self.U)

            self._x_init = np.hstack([Xsol[:, 1:], Xsol[:, -1:]])
            self._u_init = np.hstack([Usol[:, 1:], Usol[:, -1:]])
            u0 = Usol[:, 0]
            v1 = Xsol[3:6, 1]
            return u0, v1, True
        except RuntimeError as e:
            if DEBUG_PRINT:
                print("[NMPC] fail：", e)
            return np.zeros(3), x0[3:6], False



nmpc = NMPCController(N=15, dt=dt, v_max=child_speed_max, a_max=child_acc_max,
                      alpha=50.0, beta=0.25)


def follow_with_nmpc():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    prev_mother_pos = None


    prev_ex = 0.0
    prev_t = time.time()
    pd_started_this_cycle = False
    pd_ever_started = False
    target_distance_current = target_distance

    while not _exit_event.is_set():
        state_m = client.getMultirotorState(vehicle_name=vehicle_mother)
        pos_m = state_m.kinematics_estimated.position
        vel_m = state_m.kinematics_estimated.linear_velocity
        mother_pos = np.array([pos_m.x_val, pos_m.y_val, pos_m.z_val])
        mother_vel = np.array([vel_m.x_val, vel_m.y_val, vel_m.z_val])

        direction = mother_vel.copy()
        if np.linalg.norm(direction[:2]) < 1e-3:
            direction = mother_pos - prev_mother_pos if prev_mother_pos is not None else np.array([1.0, 0.0, 0.0])
        nrm = np.linalg.norm(direction)
        direction = direction / nrm if nrm > 1e-3 else np.array([1.0, 0.0, 0.0])
        prev_mother_pos = mother_pos.copy()


        z_follow_offset = 1.0
        target_pos = mother_pos - direction * target_distance_current
        target_pos[2] += z_follow_offset


        state_c = client.getMultirotorState(vehicle_name=vehicle_child)
        pos_c = state_c.kinematics_estimated.position
        vel_c = state_c.kinematics_estimated.linear_velocity
        child_pos = np.array([pos_c.x_val, pos_c.y_val, pos_c.z_val])
        v_current = np.array([vel_c.x_val, vel_c.y_val, vel_c.z_val])


        N = nmpc.N
        ref_pos_traj = np.tile(target_pos.reshape(3, 1), (1, N + 1))
        ref_vel = mother_vel.copy()
        spd_m = np.linalg.norm(ref_vel)
        if spd_m > child_speed_max:
            ref_vel = ref_vel / spd_m * child_speed_max
        ref_vel_traj = np.tile(ref_vel.reshape(3, 1), (1, N + 1))


        now = time.time()
        with _shared_lock:
            ex = _shared_ex
            t_ex = _shared_ex_time
            valid = _shared_valid

        pd_started_this_cycle = (valid and (now - t_ex) <= pd_timeout)


        if pd_started_this_cycle and not pd_ever_started:
            pd_ever_started = True
            target_distance_current = 1
            if DEBUG_PRINT:
                print("[PD] start")


        vmax_now = V_MAX_NO_LIMIT if pd_ever_started else child_speed_max
        x0 = np.hstack([child_pos, v_current])
        u0, v1_pred, ok = nmpc.solve(x0, ref_pos_traj, ref_vel_traj, vmax_now)

        if not ok:
            force_att = k_att * (target_pos - child_pos)
            acc_cmd = (force_att - v_current) / dt
            acc_norm = np.linalg.norm(acc_cmd)
            if acc_norm > child_acc_max:
                acc_cmd = acc_cmd / acc_norm * child_acc_max
            v_cmd = v_current + acc_cmd * dt
        else:
            v_cmd = v_current + u0 * dt


        if pd_started_this_cycle:
            if abs(ex) < x_deadband:
                ex = 0.0
            dt_pd = max(1e-3, now - prev_t)
            dex = (ex - prev_ex) / dt_pd

            vy_body = Kp_lat * ex + Kd_lat * dex
            vy_body = max(-vy_lat_max, min(vy_lat_max, vy_body))


            ori = state_c.kinematics_estimated.orientation
            yaw = R.from_quat([ori.x_val, ori.y_val, ori.z_val, ori.w_val]).as_euler('xyz', degrees=False)[2]
            world_right = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
            v_corr = vy_body * world_right
            v_cmd[:2] += v_corr[:2]

            if DEBUG_PRINT:
                print(f"[PD] ex={ex:+.3f} dex={dex:+.3f} | vy_body={vy_body:+.3f} | "
                      f"v_corr=({v_corr[0]:+.3f},{v_corr[1]:+.3f}) | "
                      f"v_cmd=({v_cmd[0]:+.3f},{v_cmd[1]:+.3f},{v_cmd[2]:+.3f})")

            prev_ex = ex
            prev_t = now
        else:
            prev_ex = 0.0
            prev_t = now

        # ======
        if not pd_ever_started:
            v_cmd = np.clip(v_cmd, -child_speed_max, child_speed_max)
            spd = np.linalg.norm(v_cmd)
            if spd > child_speed_max:
                v_cmd = v_cmd / spd * child_speed_max


        if _recording:
            with _traj_lock:
                _traj_log.append([
                    now,
                    float(mother_pos[0]), float(mother_pos[1]), float(mother_pos[2]),
                    float(child_pos[0]), float(child_pos[1]), float(child_pos[2]),
                ])


        client.moveByVelocityAsync(*v_cmd, dt,
                                   drivetrain=airsim.DrivetrainType.ForwardOnly,
                                   yaw_mode=airsim.YawMode(False, 0),
                                   vehicle_name=vehicle_child)
        time.sleep(dt)


def detect_aruco_from_drone2():
    global _shared_ex, _shared_ex_time, _shared_valid
    client = airsim.MultirotorClient()
    client.confirmConnection()

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    with np.load("camera_calibration.npz") as data:
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']

    marker_length = 0.5
    sqrt3 = math.sqrt(3)
    x_33 = -0.015 * sqrt3 - 0.03
    x_99 = 0.015 * sqrt3 + 0.03
    z_shift = 0.015

    def create_precise_transform(x_offset, z_offset, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        Rm, _ = cv2.Rodrigues(np.array([0, angle_rad, 0]))
        T = np.eye(4)
        T[:3, :3] = Rm
        T[:3, 3] = [x_offset, 0, z_offset]
        return T

    T_66_33_known = create_precise_transform(x_33, z_shift, +30)
    T_66_99_known = create_precise_transform(x_99, z_shift, -30)
    T_33_66_known = np.linalg.inv(T_66_33_known)
    T_99_66_known = np.linalg.inv(T_66_99_known)

    def get_T(rvec, tvec):
        Rm, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = Rm
        T[:3, 3] = tvec.flatten()
        return T

    def decompose_T_quat(T):
        pos = T[:3, 3]
        rot = R.from_matrix(T[:3, :3])
        return pos, rot.as_quat()

    def compose_T_quat(pos, quat):
        T = np.eye(4)
        T[:3, :3] = R.from_quat(quat).as_matrix()
        T[:3, 3] = pos
        return T

    def weighted_average_quaternions(quats, weights):
        quat_array = np.array(quats)
        weighted = quat_array.T @ np.array(weights)
        return weighted / np.linalg.norm(weighted)

    kf = KalmanFilter(dim_x=3, dim_z=3)
    kf.F = np.eye(3)
    kf.H = np.eye(3)
    kf.P *= 1.0
    kf.R *= 0.002
    kf.Q *= 1e-4
    kf_initialized = False
    # ======
    RECORD_ENABLE = True
    RECORD_DURATION_S = 42
    RECORD_FPS = 12
    RECORD_PATH = "airsim_capture.mp4"

    writer = None
    rec_start_time = None
    frames_written = 0

    while not _exit_event.is_set():
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ], vehicle_name=vehicle_child)
        response = responses[0]
        if response.width <= 0:
            continue

        frame = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)
        frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        valid_now = False
        ex_now = 0.0

        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
            if rvecs is not None and tvecs is not None:
                poses = {int(i[0]): (r, t) for i, r, t in zip(ids, rvecs, tvecs)}
                T_cam = {mid: get_T(*poses[mid]) for mid in poses}

                if all(mid in T_cam for mid in [33, 66, 99]):
                    T_66s = [T_cam[33] @ T_33_66_known, T_cam[66], T_cam[99] @ T_99_66_known]
                    positions, quats = zip(*(decompose_T_quat(T) for T in T_66s))

                    if not kf_initialized:
                        kf.x = np.mean(positions, axis=0)
                        kf_initialized = True

                    for pos in positions:
                        kf.predict()
                        kf.update(pos)

                    quat_fused = weighted_average_quaternions(quats, [0.25, 0.5, 0.25])
                    _ = compose_T_quat(kf.x, quat_fused)

                    ex_now = float(kf.x[0])
                    valid_now = True

                    if DEBUG_PRINT:
                        print(f"[ARUCO] xyz_cam66 = ({kf.x[0]:+.3f}, {kf.x[1]:+.3f}, {kf.x[2]:+.3f}) m")


                for rvec, tvec in zip(rvecs, tvecs):
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                # --------------
        if RECORD_ENABLE:
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(RECORD_PATH, fourcc, RECORD_FPS, (w, h))
                if not writer.isOpened():
                    print("[REC] open VideoWriter fail。")
                    RECORD_ENABLE = False
                else:
                    rec_start_time = time.time()
                    frames_written = 0
                    print(
                        f"[REC] recording：{RECORD_PATH}，target {RECORD_DURATION_S}s @ {RECORD_FPS}fps，size {w}x{h}")


            if writer is not None and (time.time() - rec_start_time) <= RECORD_DURATION_S:
                writer.write(frame)  # 此处写入的是带叠加的可视化帧
                frames_written += 1

            elif writer is not None and (time.time() - rec_start_time) > RECORD_DURATION_S:
                writer.release()
                writer = None
                print(f"[REC] record：{RECORD_PATH}，write {frames_written} 帧。")

                RECORD_ENABLE = False

        with _shared_lock:
            if valid_now:
                _shared_ex = ex_now
                _shared_ex_time = time.time()
                _shared_valid = True
            else:
                _shared_valid = False


        cv2.imshow("Drone2 ArUco Detection (press 'q' )", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            _exit_event.set()
            break

        time.sleep(0.02)

    cv2.destroyAllWindows()
    try:
        if writer is not None:
            writer.release()
            print(f"[REC] close：{RECORD_PATH}，write {frames_written} 帧。")
    except Exception as e:
        print(f"[REC] error：{e}")


# === CSV ===
def save_traj_csv(path="traj_log.csv"):
    with _traj_lock:
        rows = list(_traj_log)
    if not rows:
        print("[LOG] no_data。")
        return
    header = ["t_unix", "mother_x", "mother_y", "mother_z", "child_x", "child_y", "child_z"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[LOG] saved: {path} （{len(rows)} 行）")


# ======
client = airsim.MultirotorClient()
client.confirmConnection()


client.enableApiControl(True, vehicle_name=vehicle_mother)
client.armDisarm(True, vehicle_name=vehicle_mother)
client.takeoffAsync(vehicle_name=vehicle_mother).join()
client.moveToZAsync(z_mother, 3, vehicle_name=vehicle_mother).join()
threading.Thread(target=fly_mother, daemon=True).start()

print("wait 10s start Drone2...")
for _ in range(100):
    if _exit_event.is_set(): break
    time.sleep(0.1)

if not _exit_event.is_set():
    client.enableApiControl(True, vehicle_name=vehicle_child)
    client.armDisarm(True, vehicle_name=vehicle_child)


    _recording = True
    print("[LOG] recording")

    client.takeoffAsync(vehicle_name=vehicle_child).join()
    client.moveToZAsync(z_subdrone, 3, vehicle_name=vehicle_child).join()
    client.moveToPositionAsync(-10, 0, z_subdrone, 3, vehicle_name=vehicle_child).join()


    threading.Thread(target=follow_with_nmpc, daemon=True).start()
    threading.Thread(target=detect_aruco_from_drone2, daemon=True).start()


try:
    while not _exit_event.is_set():
        time.sleep(0.1)
finally:

    try:
        save_traj_csv("traj_log.csv")
    except Exception as e:
        print(f"[LOG] save fail: {e}")
    for vname in [vehicle_mother, vehicle_child]:
        try:
            client.armDisarm(False, vehicle_name=vname)
            client.enableApiControl(False, vehicle_name=vname)
        except Exception as e:
            print(f"[LOG] release {vname} fail: {e}")
    print("[LOG] quit")
