from lab_amoro.parallel_robot import *
from lab_amoro.plot_tools import *
from five_bar_models import *  # Modify this to use the biglide
import sys
import numpy as np


def main(args=None):
    # Initialize and start the ROS2 robot interface
    rclpy.init(args=args)
    robot = Robot("five_bar")  # Modify this to use the biglide
    start_robot(robot)

    # Prepare plots
    app = QtGui.QApplication([])
    scope_joint1 = Scope("Joint 1", -0.5, 1.5)
    scope_joint2 = Scope("Joint 2", -1.5, 1.5)

    # Create the trajectory as arrays in Cartesian space (position, velocity, acceleration)
    t_f = 2.0
    dt = 0.01
    xA, yA = 0.09, 0.06796322
    xB, yB = 0.0, 0.1
    t = np.arange(0, t_f+dt, dt)
    tau = t / t_f
    s = 10*tau**3 - 15*tau**4 + 6*tau**5
    sd = (30*tau**2 - 60*tau**3 + 30*tau**4) / t_f
    sdd = (60*tau - 180*tau**2 + 120*tau**3) / (t_f**2)

    x = xA + s*(xB - xA)
    y = yA + s*(yB - yA)
    xd = sd*(xB - xA)
    yd = sd*(yB - yA)
    xdd = sdd*(xB - xA)
    ydd = sdd*(yB - yA)

    # Create the trajectory as arrays in joint space using the inverse models (position, velocity, acceleration)
    N = len(t) 
    q_d   = np.zeros((N, 2))
    qd_d  = np.zeros((N, 2))
    qdd_d = np.zeros((N, 2))
    # Choose constant assembly configuration
    gamma1 = -1
    gamma2 = -1
    assembly_mode = -1

    for k in range(N):
    # 1) Inverse geometric model: active + passive joints
    q11, q21 = igm(x[k], y[k], gamma1, gamma2)
    q12, q22 = dgm_passive(q11, q21, assembly_mode)

    # 2) Inverse first-order kinematic model: velocities
    q11D, q21D = ikm(q11, q12, q21, q22, xd[k], yd[k])
    q12D, q22D = dkm_passive(q11, q12, q21, q22, q11D, q21D, xd[k], yd[k])

    # 3) Inverse second-order kinematic model: accelerations
    q11DD, q21DD = ikm2(q11, q12, q21, q22,
                        q11D, q12D, q21D, q22D,
                        xdd[k], ydd[k])

    # 4) Store results
    q_d[k, :]   = [q11,  q21]
    qd_d[k, :]  = [q11D, q21D]
    qdd_d[k, :] = [q11DD, q21DD]


# Optional: store in dictionary for later indexing in controller
    trajectory = {
    "t": t,
    "q_d": q_d,
    "qd_d": qd_d,
    "qdd_d": qdd_d
}        

    # Controller
    try:
        robot.apply_efforts(0.0, 0.0)  # Required to start the simulation
        while True:
            if robot.data_updated():
                # Robot available data - This is the only data thet you can get from a real robot (joint encoders)
                q11 = robot.active_left_joint.position
                q21 = robot.active_right_joint.position
                q11D = robot.active_left_joint.velocity
                q21D = robot.active_right_joint.velocity

                # CTC controller
                # TODO
                robot.apply_efforts(tau_left, tau_right)

                # Scope update
                time = robot.get_time()
                if time < 5.0:
                    scope_joint1.update(time, 0.0, 0.0)
                    scope_joint2.update(time, 0.0., 0.0)

                if index < len(trajectory)-1:
                    index += 1  # Next point in trajectory

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main(sys.argv)
