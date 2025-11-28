from lab_amoro.parallel_robot import *
from lab_amoro.plot_tools import * 
from biglide_models import *  # Modify this to use the biglide
import sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

## =======================================
# Singularity Part

ti = 0  # example initial time
ts = 1  # example time at singularity
tf = 3.0  # example final time

P = np.array([
    [ti**8, ti**7, ti**6, ti**5, ti**4, ti**3, ti**2, ti, 1],
    [8*ti**7, 7*ti**6, 6*ti**5, 5*ti**4, 4*ti**3, 3*ti**2, 2*ti, 1, 0],
    [56*ti**6, 42*ti**5, 30*ti**4, 20*ti**3, 12*ti**2, 6*ti, 2, 0, 0],
    
    [ts**8, ts**7, ts**6, ts**5, ts**4, ts**3, ts**2, ts, 1],
    [8*ts**7, 7*ts**6, 6*ts**5, 5*ts**4, 4*ts**3, 3*ts**2, 2*ts, 1, 0],
    [56*ts**6, 42*ts**5, 30*ts**4, 20*ts**3, 12*ts**2, 6*ts, 2, 0, 0],
    
    [tf**8, tf**7, tf**6, tf**5, tf**4, tf**3, tf**2, tf, 1],
    [8*tf**7, 7*tf**6, 6*tf**5, 5*tf**4, 4*tf**3, 3*tf**2, 2*tf, 1, 0],
    [56*tf**6, 42*tf**5, 30*tf**4, 20*tf**3, 12*tf**2, 6*tf, 2, 0, 0]
])

xi = 0.0
xs = 1.0
xsD = -0.01
xsDD = -6.8e-4
xf = 2.0

cx = np.array([xi, 0, 0, xs, xsD, xsDD, xf, 0, 0])

# Solve for the coefficients a
a = np.linalg.solve(P, cx)

# Define functions for x(t), dot_x(t), and ddot_x(t)
def x(t):
    return (a[0] * t**8 + a[1] * t**7 + a[2] * t**6 +
            a[3] * t**5 + a[4] * t**4 + a[5] * t**3 +
            a[6] * t**2 + a[7] * t + a[8])

def xD(t):
    return (8 * a[0] * t**7 + 7 * a[1] * t**6 + 6 * a[2] * t**5 +
            5 * a[3] * t**4 + 4 * a[4] * t**3 + 3 * a[5] * t**2 +
            2 * a[6] * t + a[7])

def xDD(t):
    return (56 * a[0] * t**6 + 42 * a[1] * t**5 + 30 * a[2] * t**4 +
            20 * a[3] * t**3 + 12 * a[4] * t**2 + 6 * a[5] * t +
            2 * a[6])

## =======================================

def st(t, tf):
    return 10*(t/tf)**3 - 15*(t/tf)**4 + 6*(t/tf)**5

def stD(t, tf):
    return 30/tf*(t/tf)**2 - 60/tf*(t/tf)**3 + 30/tf*(t/tf)**4

def stDD(t, tf):
    return 60/(tf**2)*(t/tf) - 180/(tf**2)*(t/tf)**2 + 120/(tf**2)*(t/tf)**3

def trajectory_generation(A, B, tf):
    dt = 0.01
    Ax = A[0]
    Bx = B[0]
    Ay = A[1]
    By = B[1]
        
    t = np.arange(0, tf + dt, dt).astype(float)
    
    xt = Ax + st(t, tf)*(Bx - Ax)
    yt = Ay + st(t, tf)*(By - Ay)
    Xt = np.vstack([xt, yt])
    
    xtD = stD(t, tf)*(Bx - Ax)
    ytD = stD(t, tf)*(By - Ay)
    XtD = np.vstack([xtD, ytD])
    
    xtDD = stDD(t, tf)*(Bx - Ax)
    ytDD = stDD(t, tf)*(By - Ay)
    XtDD = np.vstack([xtDD, ytDD])
    return Xt.T, XtD.T, XtDD.T, t  # return time vector too

def joint_trajectory(A, B, tf):
    Xt, XtD, XtDD, t = trajectory_generation(A, B, tf)
    x_arr = Xt[:, 0]
    y_arr = Xt[:, 1]
    
    Vx = XtD[:, 0]
    Vy = XtD[:, 1]
    
    ax = XtDD[:, 0]
    ay = XtDD[:, 1]
    
    gamma1 = -1 
    gamma2 = -1
    assembly_mode = -1
    
    n = x_arr.shape[0]
    q11 = np.zeros((n, 1))
    q21 = np.zeros((n, 1))
    q12 = np.zeros((n, 1))
    q22 = np.zeros((n, 1))
    q11D = np.zeros((n, 1))
    q21D = np.zeros((n, 1))
    q12D = np.zeros((n, 1))
    q22D = np.zeros((n, 1))
    q11DD = np.zeros((n, 1))
    q21DD = np.zeros((n, 1))
    # === ADDED: desired passive accelerations
    q12DD = np.zeros((n, 1))
    q22DD = np.zeros((n, 1))

    for i in range(n):
        q11[i], q21[i] = igm(x_arr[i], y_arr[i], gamma1, gamma2)
        q12[i], q22[i] = dgm_passive(q11[i][0], q21[i][0], assembly_mode)
        q11D[i], q21D[i] = ikm(q11[i][0], q12[i][0], q21[i][0], q22[i][0], Vx[i], Vy[i])
        q12D[i], q22D[i] = dkm_passive(q11[i][0], q12[i][0], q21[i][0], q22[i][0], q11D[i][0], q21D[i][0], Vx[i], Vy[i])
        q11DD[i], q21DD[i] = ikm2(q11[i][0], q12[i][0], q21[i][0], q22[i][0],
                                  q11D[i][0], q12D[i][0], q21D[i][0], q22D[i][0], ax[i], ay[i])
        # === ADDED: desired passive accelerations using dkm2_passive
        q12DD[i], q22DD[i] = dkm2_passive(q11[i][0], q12[i][0], q21[i][0], q22[i][0],
                                          q11D[i][0], q12D[i][0], q21D[i][0], q22D[i][0],
                                          q11DD[i][0], q21DD[i][0], ax[i], ay[i])
        
    # also return Cartesian trajectory and time
    # === CHANGED: return passive joint desired states too
    return (q11, q21, q11D, q21D, q11DD, q21DD,
            q12, q22, q12D, q22D, q12DD, q22DD,
            Xt, XtD, XtDD, t)

def plot_errors(time, joint1_errors, joint2_errors):
    plt.figure()
    plt.plot(time, joint1_errors, label='Joint 1 Error')
    plt.plot(time, joint2_errors, label='Joint 2 Error')
    plt.title("Joint Errors")
    plt.xlabel("Time (s)")
    plt.ylabel("Error")
    plt.legend()
    plt.grid()

    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.yaxis.get_offset_text().set_fontsize(12)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.show()

# === ADDED: generic multi-series overlay helper (supports linestyles)
def plot_time_multi(t, series_list, labels, title, ylabel, linestyles=None):
    plt.figure()
    for i, y in enumerate(series_list):
        ls = linestyles[i] if (linestyles is not None and i < len(linestyles)) else None
        plt.plot(t, y, label=labels[i], linestyle=ls)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# === ADDED: XY overlay (square) for end-effector path
def plot_xy_overlay_square(x_true, y_true, x_ref, y_ref,
                           title="End-effector Trajectory (XY)",
                           true_label="Actual", ref_label="Desired"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x_true, y_true, linewidth=2, label=true_label)
    ax.plot(x_ref, y_ref, linestyle=':', linewidth=2, label=ref_label)  # dotted desired

    xs = np.concatenate([np.asarray(x_true), np.asarray(x_ref)])
    ys = np.concatenate([np.asarray(y_true), np.asarray(y_ref)])
    xmin, xmax = np.nanmin(xs), np.nanmax(xs)
    ymin, ymax = np.nanmin(ys), np.nanmax(ys)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half_range = 0.5 * max(xmax - xmin, ymax - ymin)
    if half_range == 0:
        half_range = 1e-6
    pad = 0.05 * half_range
    half = half_range + pad
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect('equal', adjustable='box')
    try:
        ax.set_box_aspect(1)
    except Exception:
        pass

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()

def main(args=None):
    # Initialize and start the ROS2 robot interface
    rclpy.init(args=args)
    robot = Robot("biglide")  # Modify this to use the biglide
    start_robot(robot)

    # Prepare plots
    app = QtGui.QApplication([])
    scope_joint1 = Scope("Joint 1", -0.5, 1.5)
    scope_joint2 = Scope("Joint 2", -1.5, 1.5)

    # Desired Cartesian trajectory
    A = np.array([0., 0.])
    B = np.array([-0.06, 1.4])
    tf = 2

    # Desired joint & Cartesian trajectories
    (q11, q21, q11D, q21D, q11DD, q21DD,
     q12, q22, q12D, q22D, q12DD, q22DD,
     Xt, XtD, XtDD, t_ref) = joint_trajectory(A, B, tf)

    qt = np.hstack([q11, q21])
    qtD = np.hstack([q11D, q21D])
    qtDD = np.hstack([q11DD, q21DD])

    # === ADDED: desired passive joint arrays
    qtp = np.hstack([q12, q22])        # passive positions
    qtpD = np.hstack([q12D, q22D])     # passive velocities
    qtpDD = np.hstack([q12DD, q22DD])  # passive accelerations

    x_ref = Xt[:, 0]; y_ref = Xt[:, 1]
    xD_ref = XtD[:, 0]; yD_ref = XtD[:, 1]
    xDD_ref = XtDD[:, 0]; yDD_ref = XtDD[:, 1]

    index = 0

    # series for combined plots (actual vs desired)
    series = {
        "time": [],
        # active joints (q11, q21)
        "q1": [], "q1_ref": [],
        "q2": [], "q2_ref": [],
        "q1D": [], "q1D_ref": [],
        "q2D": [], "q2D_ref": [],
        "q1DD": [], "q1DD_ref": [],
        "q2DD": [], "q2DD_ref": [],
        # === ADDED: passive joints (q12, q22)
        "q12": [], "q12_ref": [],
        "q22": [], "q22_ref": [],
        "q12D": [], "q12D_ref": [],
        "q22D": [], "q22D_ref": [],
        "q12DD": [], "q12DD_ref": [],
        "q22DD": [], "q22DD_ref": [],
        # cartesian pos/vel/acc
        "x": [], "x_ref": [],
        "y": [], "y_ref": [],
        "xD": [], "xD_ref": [],
        "yD": [], "yD_ref": [],
        "xDD": [], "xDD_ref": [],
        "yDD": [], "yDD_ref": [],
        # control efforts
        "tau1": [], "tau2": []
    }

    # Controller
    try:
        robot.apply_efforts(0.0, 0.0)  # Required to start the simulation
        while True:
            if robot.data_updated():
                # Actual measurements
                q11_meas = robot.active_left_joint.position
                q21_meas = robot.active_right_joint.position
                q11D_meas = robot.active_left_joint.velocity
                q21D_meas = robot.active_right_joint.velocity
                try:
                    q11DD_meas = robot.active_left_joint.acceleration
                    q21DD_meas = robot.active_right_joint.acceleration
                except Exception:
                    q11DD_meas = 0.0
                    q21DD_meas = 0.0

                assembly_mode = -1
                # Actual passive joints (position)
                q12_meas, q22_meas = dgm_passive(q11_meas, q21_meas, assembly_mode)
                # Cartesian kinematics
                x_meas, y_meas = dgm(q11_meas, q21_meas, assembly_mode)
                xD_meas, yD_meas = dkm(q11_meas, q12_meas, q21_meas, q22_meas, q11D_meas, q21D_meas)
                # Actual passive joint velocities via DKM passive
                q12D_meas, q22D_meas = dkm_passive(q11_meas, q12_meas, q21_meas, q22_meas,
                                                   q11D_meas, q21D_meas, xD_meas, yD_meas)
                # Cartesian accelerations
                xDD_meas, yDD_meas = dkm2(q11_meas, q12_meas, q21_meas, q22_meas,
                                          q11D_meas, q12D_meas, q21D_meas, q22D_meas,
                                          q11DD_meas, q21DD_meas)
                # === ADDED: actual passive accelerations via dkm2_passive
                q12DD_meas, q22DD_meas = dkm2_passive(q11_meas, q12_meas, q21_meas, q22_meas,
                                                      q11D_meas, q12D_meas, q21D_meas, q22D_meas,
                                                      q11DD_meas, q21DD_meas, xDD_meas, yDD_meas)

                # CTC controller
                qa = np.array([q11_meas, q21_meas])
                qaD = np.array([q11D_meas, q21D_meas])
                M, c = dynamic_model(q11_meas, q12_meas, q21_meas, q22_meas,
                                     q11D_meas, q12D_meas, q21D_meas, q22D_meas)

                kD = np.ones((2, 2)) * 2
                kP = np.ones((2, 2)) * 4
                a_cmd = qtDD[index, :] + kD @ (qtD[index, :] - qaD) + kP @ (qt[index, :] - qa)

                tau = M @ a_cmd + c
                tau_left = tau[0]
                tau_right = tau[1]

                robot.apply_efforts(tau_left, tau_right)

                # Time and logging
                time = robot.get_time()
                if time < 5.0:
                    scope_joint1.update(time, qt[index, 0], qa[0])
                    scope_joint2.update(time, qt[index, 1], qa[1])

                    # series logging (actual vs desired)
                    series["time"].append(time)
                    # active joints
                    series["q1"].append(qa[0]);        series["q1_ref"].append(qt[index, 0])
                    series["q2"].append(qa[1]);        series["q2_ref"].append(qt[index, 1])
                    series["q1D"].append(qaD[0]);      series["q1D_ref"].append(qtD[index, 0])
                    series["q2D"].append(qaD[1]);      series["q2D_ref"].append(qtD[index, 1])
                    series["q1DD"].append(q11DD_meas); series["q1DD_ref"].append(qtDD[index, 0])
                    series["q2DD"].append(q21DD_meas); series["q2DD_ref"].append(qtDD[index, 1])
                    # === ADDED: passive joints
                    series["q12"].append(q12_meas);       series["q12_ref"].append(qtp[index, 0])
                    series["q22"].append(q22_meas);       series["q22_ref"].append(qtp[index, 1])
                    series["q12D"].append(q12D_meas);     series["q12D_ref"].append(qtpD[index, 0])
                    series["q22D"].append(q22D_meas);     series["q22D_ref"].append(qtpD[index, 1])
                    series["q12DD"].append(q12DD_meas);   series["q12DD_ref"].append(qtpDD[index, 0])
                    series["q22DD"].append(q22DD_meas);   series["q22DD_ref"].append(qtpDD[index, 1])
                    # cartesian
                    series["x"].append(x_meas);           series["x_ref"].append(x_ref[index])
                    series["y"].append(y_meas);           series["y_ref"].append(y_ref[index])
                    series["xD"].append(xD_meas);         series["xD_ref"].append(xD_ref[index])
                    series["yD"].append(yD_meas);         series["yD_ref"].append(yD_ref[index])
                    series["xDD"].append(xDD_meas);       series["xDD_ref"].append(xDD_ref[index])
                    series["yDD"].append(yDD_meas);       series["yDD_ref"].append(yDD_ref[index])
                    # efforts
                    series["tau1"].append(float(tau_left))
                    series["tau2"].append(float(tau_right))

                if index < len(qt) - 1:
                    index += 1  # Next point in trajectory

    except KeyboardInterrupt:
        pass

    # =========================
    # Plots (after control loop)
    # =========================
    t = np.array(series["time"])
    if len(t) > 1:
        dotted = ':'

        # Active joint positions in one plot (desired dotted)
        plot_time_multi(
            t,
            [series["q1"], series["q1_ref"], series["q2"], series["q2_ref"]],
            ["q11 actual", "q11 desired", "q21 actual", "q21 desired"],
            "Active Joint Positions (actual vs desired)",
            "Position",
            linestyles=[None, dotted, None, dotted]
        )

        # Active joint velocities
        plot_time_multi(
            t,
            [series["q1D"], series["q1D_ref"], series["q2D"], series["q2D_ref"]],
            ["q11D actual", "q11D desired", "q21D actual", "q21D desired"],
            "Active Joint Velocities (actual vs desired)",
            "Velocity",
            linestyles=[None, dotted, None, dotted]
        )

        # Active joint accelerations
        plot_time_multi(
            t,
            [series["q1DD"], series["q1DD_ref"], series["q2DD"], series["q2DD_ref"]],
            ["q11DD actual", "q11DD desired", "q21DD actual", "q21DD desired"],
            "Active Joint Accelerations (actual vs desired)",
            "Acceleration",
            linestyles=[None, dotted, None, dotted]
        )

        # === ADDED: Passive joint positions
        plot_time_multi(
            t,
            [series["q12"], series["q12_ref"], series["q22"], series["q22_ref"]],
            ["q12 actual", "q12 desired", "q22 actual", "q22 desired"],
            "Passive Joint Positions (actual vs desired)",
            "Position",
            linestyles=[None, dotted, None, dotted]
        )

        # === ADDED: Passive joint velocities
        plot_time_multi(
            t,
            [series["q12D"], series["q12D_ref"], series["q22D"], series["q22D_ref"]],
            ["q12D actual", "q12D desired", "q22D actual", "q22D desired"],
            "Passive Joint Velocities (actual vs desired)",
            "Velocity",
            linestyles=[None, dotted, None, dotted]
        )

        # === ADDED: Passive joint accelerations
        plot_time_multi(
            t,
            [series["q12DD"], series["q12DD_ref"], series["q22DD"], series["q22DD_ref"]],
            ["q12DD actual", "q12DD desired", "q22DD actual", "q22DD desired"],
            "Passive Joint Accelerations (actual vs desired)",
            "Acceleration",
            linestyles=[None, dotted, None, dotted]
        )

        # Cartesian position (x,y)
        plot_time_multi(
            t,
            [series["x"], series["x_ref"], series["y"], series["y_ref"]],
            ["x actual", "x desired", "y actual", "y desired"],
            "Cartesian Position (actual vs desired)",
            "Position (Cartesian)",
            linestyles=[None, dotted, None, dotted]
        )

        # Cartesian velocity (ẋ,ẏ)
        plot_time_multi(
            t,
            [series["xD"], series["xD_ref"], series["yD"], series["yD_ref"]],
            ["xD actual", "xD desired", "yD actual", "yD desired"],
            "Cartesian Velocity (actual vs desired)",
            "Velocity (Cartesian)",
            linestyles=[None, dotted, None, dotted]
        )

        # Cartesian acceleration (ẍ,ÿ)
        plot_time_multi(
            t,
            [series["xDD"], series["xDD_ref"], series["yDD"], series["yDD_ref"]],
            ["xDD actual", "xDD desired", "yDD actual", "yDD desired"],
            "Cartesian Acceleration (actual vs desired)",
            "Acceleration (Cartesian)",
            linestyles=[None, dotted, None, dotted]
        )

        # Control efforts
        plot_time_multi(
            t,
            [series["tau1"], series["tau2"]],
            ["τ1", "τ2"],
            "Control Efforts",
            "Torque / Effort"
        )

        # ---------- Tracking error & RMSE (positions) ----------
        # Active joints errors & RMSE vs time
        q11_err = np.array(series["q1"]) - np.array(series["q1_ref"])
        q21_err = np.array(series["q2"]) - np.array(series["q2_ref"])
        plot_time_multi(
            t,
            [q11_err, q21_err],
            ["q11 error", "q21 error"],
            "Active Joint Position Tracking Error",
            "Error"
        )
        rmse_q11_t = np.sqrt(np.cumsum(q11_err**2) / np.arange(1, len(t) + 1))
        rmse_q21_t = np.sqrt(np.cumsum(q21_err**2) / np.arange(1, len(t) + 1))
        plot_time_multi(
            t,
            [rmse_q11_t, rmse_q21_t],
            ["q11 RMSE", "q21 RMSE"],
            "Cumulative RMSE vs Time (Active Joint Position)",
            "RMSE"
        )

        # === ADDED: Passive joints errors & RMSE vs time
        q12_err = np.array(series["q12"]) - np.array(series["q12_ref"])
        q22_err = np.array(series["q22"]) - np.array(series["q22_ref"])
        plot_time_multi(
            t,
            [q12_err, q22_err],
            ["q12 error", "q22 error"],
            "Passive Joint Position Tracking Error",
            "Error"
        )
        rmse_q12_t = np.sqrt(np.cumsum(q12_err**2) / np.arange(1, len(t) + 1))
        rmse_q22_t = np.sqrt(np.cumsum(q22_err**2) / np.arange(1, len(t) + 1))
        plot_time_multi(
            t,
            [rmse_q12_t, rmse_q22_t],
            ["q12 RMSE", "q22 RMSE"],
            "Cumulative RMSE vs Time (Passive Joint Position)",
            "RMSE"
        )

        # XY overlay (actual vs desired)
        plot_xy_overlay_square(
            series["x"], series["y"],
            x_ref, y_ref,
            title="End-effector Trajectory (XY): Actual vs Desired",
            true_label="Actual", ref_label="Desired"
        )

if __name__ == "__main__":
    main(sys.argv)
