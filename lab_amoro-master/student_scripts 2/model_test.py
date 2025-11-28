import matplotlib.pyplot as plt
from lab_amoro.parallel_robot import *
from lab_amoro.plot_tools import *
from biglide_models import *  # Modify this to test the biglide
import sys


def main(args=None):
    # Initialize and start the ROS2 robot interface
    rclpy.init(args=args)
    robot = Robot("biglide")  # Modify this to test the biglide
    start_robot(robot)

    # Prepare plots - Comment scopes if you don't want to see them
    app = QtGui.QApplication([])
    scope_x = Scope("Position x", 0.0, 0.2)
    scope_y = Scope("Position y", -0.5, 0.5)
    scope_q12 = Scope("Position q12", -1.0, 1.5)
    scope_q22 = Scope("Position q22", 0, 3.0)
    scope_xD = Scope("Velocity x", -0.02, 0.02)
    scope_yD = Scope("Velocity y", -0.02, 0.02)
    scope_q12D = Scope("Velocity q12", -0.2, 0.2)
    scope_q22D = Scope("Velocity q22", -0.2, 0.2)
    scope_xDD = Scope("Acceleration x", -0.05, 0.05)
    scope_yDD = Scope("Acceleration y", -0.05, 0.05)
    scope_q12DD = Scope("Acceleration q12", -0.5, 0.5)
    scope_q22DD = Scope("Acceleration q22", -0.5, 0.5)

    # Lists to store errors
    errors = {
        "time": [],
        "x": [], "y": [], "q12": [], "q22": [],
        "xD": [], "yD": [], "q12D": [], "q22D": [],
        "xDD": [], "yDD": [], "q12DD": [], "q22DD": []
    }

    # === ADDED: Series for overlays (true vs model over time) ===
    series = {
        "time": [],
        "x": [], "x_model": [], "y": [], "y_model": [],
        "q12": [], "q12_model": [], "q22": [], "q22_model": [],
        "xD": [], "xD_model": [], "yD": [], "yD_model": [],
        "q12D": [], "q12D_model": [], "q22D": [], "q22D_model": [],
        "xDD": [], "xDD_model": [], "yDD": [], "yDD_model": [],
        "q12DD": [], "q12DD_model": [], "q22DD": [], "q22DD_model": [],
    }

    # === ADDED: Lists to store XY / joint-space trajectories (paths) ===
    traj = {
        "x": [], "y": [],
        "x_model": [], "y_model": [],
        "q12": [], "q22": [],
        "q12_model": [], "q22_model": [],
        "xD": [], "yD": [], "xD_model": [], "yD_model": [],
        "xDD": [], "yDD": [], "xDD_model": [], "yDD_model": [],
    }

    # Compare the models
    robot.start_oscillate()
    while True:  # Runs the simulation forever
        try:
            if robot.data_updated():
                # Test of the Direct Geometric Model
                # Data from gazebo
                q11 = robot.active_left_joint.position
                q21 = robot.active_right_joint.position
                x = robot.end_effector.position_x
                y = robot.end_effector.position_y
                q12 = robot.passive_left_joint.position
                q22 = robot.passive_right_joint.position
                # Dgm
                x_model, y_model = dgm(q11, q21, -1)
                q12_model, q22_model = dgm_passive(q11, q21, -1)
                # Plot update
                time = robot.get_time()
                scope_y.update(time, y, y_model)
                scope_x.update(time, x, x_model)
                scope_q12.update(time, q12, q12_model)
                scope_q22.update(time, q22, q22_model)

                # Collect errors
                errors["time"].append(time)
                errors["x"].append(x - x_model)
                errors["y"].append(y - y_model)
                errors["q12"].append(q12 - q12_model)
                errors["q22"].append(q22 - q22_model)

                # === ADDED: overlay series (position)
                series["time"].append(time)
                series["x"].append(x);             series["x_model"].append(x_model)
                series["y"].append(y);             series["y_model"].append(y_model)
                series["q12"].append(q12);         series["q12_model"].append(q12_model)
                series["q22"].append(q22);         series["q22_model"].append(q22_model)

                # === ADDED: XY & joint-space trajectory points
                traj["x"].append(x);               traj["y"].append(y)
                traj["x_model"].append(x_model);   traj["y_model"].append(y_model)
                traj["q12"].append(q12);           traj["q22"].append(q22)
                traj["q12_model"].append(q12_model); traj["q22_model"].append(q22_model)

                # Test of the Direct Kinematic Model
                # Data from gazebo
                q11D = robot.active_left_joint.velocity
                q21D = robot.active_right_joint.velocity
                xD = robot.end_effector.velocity_x
                yD = robot.end_effector.velocity_y
                q12D = robot.passive_left_joint.velocity
                q22D = robot.passive_right_joint.velocity
                # Dkm
                xD_model, yD_model = dkm(q11, q12_model, q21, q22_model, q11D, q21D)
                q12D_model, q22D_model = dkm_passive(
                    q11, q12_model, q21, q22_model, q11D, q21D, xD_model, yD_model
                )
                # Plot update
                time = robot.get_time()
                scope_yD.update(time, yD, yD_model)
                scope_xD.update(time, xD, xD_model)
                scope_q12D.update(time, q12D, q12D_model)
                scope_q22D.update(time, q22D, q22D_model)

                # Collect errors
                errors["xD"].append(xD - xD_model)
                errors["yD"].append(yD - yD_model)
                errors["q12D"].append(q12D - q12D_model)
                errors["q22D"].append(q22D - q22D_model)

                # === ADDED: overlay series (velocity)
                series["xD"].append(xD);           series["xD_model"].append(xD_model)
                series["yD"].append(yD);           series["yD_model"].append(yD_model)
                series["q12D"].append(q12D);       series["q12D_model"].append(q12D_model)
                series["q22D"].append(q22D);       series["q22D_model"].append(q22D_model)

                # === ADDED: velocity-plane trajectory points
                traj["xD"].append(xD);             traj["yD"].append(yD)
                traj["xD_model"].append(xD_model); traj["yD_model"].append(yD_model)

                # Test of the Direct Kinematic Model Second order
                # Data from gazebo
                q11DD = robot.active_left_joint.acceleration
                q21DD = robot.active_right_joint.acceleration
                xDD = robot.end_effector.acceleration_x
                yDD = robot.end_effector.acceleration_y
                q12DD = robot.passive_left_joint.acceleration
                q22DD = robot.passive_right_joint.acceleration
                # DKM 2nd order
                xDD_model, yDD_model = dkm2(
                    q11, q12_model, q21, q22_model,
                    q11D, q12D_model, q21D, q22D_model,
                    q11DD, q21DD
                )
                q12DD_model, q22DD_model = dkm2_passive(
                    q11, q12_model, q21, q22_model,
                    q11D, q12D_model, q21D, q22D_model,
                    q11DD, q21DD, xDD_model, yDD_model
                )
                # Plot update
                time = robot.get_time()
                scope_yDD.update(time, yDD, yDD_model)
                scope_xDD.update(time, xDD, xDD_model)
                scope_q12DD.update(time, q12DD, q12DD_model)
                scope_q22DD.update(time, q22DD, q22DD_model)

                # Collect errors
                errors["xDD"].append(xDD - xDD_model)
                errors["yDD"].append(yDD - yDD_model)
                errors["q12DD"].append(q12DD - q12DD_model)
                errors["q22DD"].append(q22DD - q22DD_model)

                # === ADDED: overlay series (acceleration)
                series["xDD"].append(xDD);         series["xDD_model"].append(xDD_model)
                series["yDD"].append(yDD);         series["yDD_model"].append(yDD_model)
                series["q12DD"].append(q12DD);     series["q12DD_model"].append(q12DD_model)
                series["q22DD"].append(q22DD);     series["q22DD_model"].append(q22DD_model)

                # === ADDED: acceleration-plane trajectory points
                traj["xDD"].append(xDD);           traj["yDD"].append(yDD)
                traj["xDD_model"].append(xDD_model); traj["yDD_model"].append(yDD_model)

                # Go on with the oscillations
                robot.continue_oscillations()

        except KeyboardInterrupt:
            break

    # Plot errors (kept as-is)
    import matplotlib.ticker as ticker

    def plot_with_scientific_notation(x, y, title, xlabel, ylabel, label):
        plt.figure()
        plt.plot(x, y, label=label)
        plt.legend()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.yaxis.get_offset_text().set_fontsize(12)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # Force scientific notation
        plt.grid(True, alpha=0.3)
        plt.show()

    plot_with_scientific_notation(errors["time"], errors["x"], "Position Errors (x)", "Time", "Error in x", "Error in x")
    plot_with_scientific_notation(errors["time"], errors["y"], "Position Errors (y)", "Time", "Error in y", "Error in y")
    plot_with_scientific_notation(errors["time"], errors["q12"], "Position Errors (q12)", "Time", "Error in q12", "Error in q12")
    plot_with_scientific_notation(errors["time"], errors["q22"], "Position Errors (q22)", "Time", "Error in q22", "Error in q22")
    plot_with_scientific_notation(errors["time"], errors["xD"], "Velocity Errors (xD)", "Time", "Error in xD", "Error in xD")
    plot_with_scientific_notation(errors["time"], errors["yD"], "Velocity Errors (yD)", "Time", "Error in yD", "Error in yD")
    plot_with_scientific_notation(errors["time"], errors["q12D"], "Velocity Errors (q12D)", "Time", "Error in q12D", "Error in q12D")
    plot_with_scientific_notation(errors["time"], errors["q22D"], "Velocity Errors (q22D)", "Time", "Error in q22D", "Error in q22D")
    plot_with_scientific_notation(errors["time"], errors["xDD"], "Acceleration Errors (xDD)", "Time", "Error in xDD", "Error in xDD")
    plot_with_scientific_notation(errors["time"], errors["yDD"], "Acceleration Errors (yDD)", "Time", "Error in yDD", "Error in yDD")
    plot_with_scientific_notation(errors["time"], errors["q12DD"], "Acceleration Errors (q12DD)", "Time", "Error in q12DD", "Error in q12DD")
    plot_with_scientific_notation(errors["time"], errors["q22DD"], "Acceleration Errors (q22DD)", "Time", "Error in q22DD", "Error in q22DD")

    # === ADDED: generic time-series overlay helper ===
    def plot_time_overlay(t, y_true, y_model, title, ylabel, true_label="Gazebo", model_label="Model"):
        plt.figure()
        plt.plot(t, y_true, linewidth=2, label=true_label)
        plt.plot(t, y_model, linewidth=2, linestyle='--', label=model_label)
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    # === ADDED: Overlays for ALL parameters (true vs model over time) ===
    t = series["time"]
    # Positions
    plot_time_overlay(t, series["x"],   series["x_model"],   "Overlay: x (true vs model)",   "x")
    plot_time_overlay(t, series["y"],   series["y_model"],   "Overlay: y (true vs model)",   "y")
    plot_time_overlay(t, series["q12"], series["q12_model"], "Overlay: q12 (true vs model)", "q12")
    plot_time_overlay(t, series["q22"], series["q22_model"], "Overlay: q22 (true vs model)", "q22")
    # Velocities
    plot_time_overlay(t, series["xD"],   series["xD_model"],   "Overlay: xD (true vs model)",   "ẋ")
    plot_time_overlay(t, series["yD"],   series["yD_model"],   "Overlay: yD (true vs model)",   "ẏ")
    plot_time_overlay(t, series["q12D"], series["q12D_model"], "Overlay: q12D(true vs model)", "q12̇")
    plot_time_overlay(t, series["q22D"], series["q22D_model"], "Overlay: q22D (true vs model)", "q22̇")
    # Accelerations
    plot_time_overlay(t, series["xDD"],   series["xDD_model"],   "Overlay: xDD (true vs model)",   "ẍ")
    plot_time_overlay(t, series["yDD"],   series["yDD_model"],   "Overlay: yDD (true vs model)",   "ÿ")
    plot_time_overlay(t, series["q12DD"], series["q12DD_model"], "Overlay: q12DD (true vs model)", "q12̈")
    plot_time_overlay(t, series["q22DD"], series["q22DD_model"], "Overlay: q22DD (true vs model)", "q22̈")

    def plot_xy_overlay(x_true, y_true, x_model, y_model,
                    title="Trajectory Overlay (XY)",
                    true_label="Gazebo", model_label="Model"):
						
						fig, ax = plt.subplots(figsize=(6, 6))  # square figure

						ax.plot(x_true, y_true, linewidth=2, label=true_label)
						ax.plot(x_model, y_model, linestyle='--', linewidth=2, label=model_label)

						# --- make the plotting box truly square around all data ---
						xs = np.concatenate([np.asarray(x_true),  np.asarray(x_model)])
						ys = np.concatenate([np.asarray(y_true),  np.asarray(y_model)])

						xmin, xmax = np.nanmin(xs), np.nanmax(xs)
						ymin, ymax = np.nanmin(ys), np.nanmax(ys)

						cx = 0.5 * (xmin + xmax)
						cy = 0.5 * (ymin + ymax)

						half_range = 0.5 * max(xmax - xmin, ymax - ymin)
						if half_range == 0:
							half_range = 1e-6  # avoid degenerate window

						pad = 0.05 * half_range
						half = half_range + pad

						ax.set_xlim(cx - half, cx + half)
						ax.set_ylim(cy - half, cy + half)

						# keep units equal and the axes box square irrespective of window size
						ax.set_aspect('equal', adjustable='box')
						try:
							ax.set_box_aspect(1)  # matplotlib >= 3.3
						except Exception:
							pass

						ax.set_title(title)
						ax.set_xlabel("x")
						ax.set_ylabel("y")
						ax.legend()
						ax.grid(True, alpha=0.3)
						fig.tight_layout()
						plt.show()


if __name__ == "__main__":
    main(sys.argv)
