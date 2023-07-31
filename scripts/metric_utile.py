from bagpy import bagreader

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

import os
import glob
import yaml

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.collections import LineCollection
import imageio.v2 as imageio

degrees=False
title_fontsize = 26
legend_fontsize = 16
legend_title_size = 18
label_fontsize = 18
tick_size = 16
ang_limit = np.pi

def rotBtoI_np(quat):
    x = quat[:, 0]
    y = quat[:, 1]
    z = quat[:, 2]
    w = quat[:, 3]

    rot = np.array([
                        [1 - 2 * (y**2 + z**2),
                        2 * (x * y - z * w),
                        2 * (x * z + y * w)],
                        [2 * (x * y + z * w),
                        1 - 2 * (x**2 + z**2),
                        2 * (y * z - x * w)],
                        [2 * (x * z - y * w),
                        2 * (y * z + x * w),
                        1 - 2 * (x**2 + y**2)]
                    ])
    print(rot.shape)
    return rot


def response_time_idx(traj, goal, axis, threshold):
    for i, e in enumerate(traj):
        if np.abs(e[axis]-goal[axis]) < threshold:
            return i


def steady_error(traj, goal, steps, axis, threshold):
    for i, e in enumerate(traj):
        if np.abs(e[axis] - goal[axis]) < threshold:
            #start average
            steady_err = np.abs(traj[i: i+steps] - goal)
            return steady_err


def mse_t(traj, goal):
    return np.linalg.norm(traj-goal, axis=0)


def energy_estimate(inputs, max_steps):
    return np.array([np.sum(np.abs(inp)) for inp in inputs])


def error_traj(traj, goal):
    return traj-goal


def gen_metric_dict(controllers, response_time, steady_errors, energies, mses, errors):
    dict = {}
    for c, r, se, e, m, err in zip(controllers, response_time, steady_errors, energies, mses, errors):
        dict[c] = {}
        dict[c]['response_time'] = r
        dict[c]['steady_state_error'] = se
        dict[c]['energies'] = e
        dict[c]['mse'] = m
        dict[c]['error'] = err
    return dict


def load_yaml(file):
    with open(file) as file:
        dictionnary = yaml.load(file, Loader=yaml.FullLoader)
    return dictionnary


def load_goal(file):
    d = load_yaml(file)
    return np.array(d['goals'][0])


def traj_to_euler(traj):
    quat = traj[:, 3:3+4]
    r = R.from_quat(quat)
    pos = traj[:, :3]
    euler = r.as_euler('XYZ', degrees=degrees)
    vel = traj[:, -6:]

    traj = np.concatenate([pos, euler, vel], axis=-1)
    return traj


def to_euler(pose):
    quat = pose[3:7]
    r = R.from_quat(quat)
    pos = pose[:3]
    euler = r.as_euler('XYZ', degrees=degrees)
    vel = pose[-6:]
    return np.concatenate([pos, euler, vel])


def get_dict_value(entry, dict):
    if entry == "noise":
        value = dict["config"]["noise"][0][0]
    else:
        value = dict[entry]
    return value


def get_label_text(entries, dict):
    label = ""
    for entry in entries:
        if entry == "noise":
            value = dict["config"]["noise"][0][0]
        elif entry == "filter_seq":
            value = dict["filter_seq"]
        else:
            value = dict[entry]
        label = f"{value}"
    return label


def get_label_title(entries):
    title = ""
    for entry in entries:
        if entry == "filter_seq":
            title = "filter"
        else:
            title = entry
    return title


def get_title(buoyancy, action, graph):
    buoyancy_dict = {"buoy": "Neutrally Buoyant", "neg": "Negatively Buoyant", "pos": "Positively Buoyant"}
    action_dict = {"up": "Move up", "down": "Move down", "back": "Move backwards", "forward": "Move forward"}
    graph = graph + " evolution."

    return graph + "\n" + buoyancy_dict[buoyancy] + " " + action_dict[action]


def gen_gif(imgs_dir, dst_dir, name):
    files = glob.glob(os.path.join(imgs_dir, "*.png"))
    files_name = [os.path.basename(f) for f in files]
    files_name.sort()
    gif_file = os.path.join(dst_dir, f"{name}.gif")
    print(f"|\t gen gif at {gif_file}")
    with imageio.get_writer(gif_file, mode='I', duration=500) as writer:
        for f in files_name:
            image = imageio.imread(os.path.join(imgs_dir, f))
            writer.append_data(image)


def filter_trajs(dicts, key_dict, labels, trajs_euler, trajs, errors, traj_thrust, traj_thrust_t):
    filtered_list = []
    for (d, l, te, t, e, tt, ttt) in zip(dicts, labels, trajs_euler, trajs, errors, traj_thrust, traj_thrust_t):
        if is_valid_entry(d, key_dict):
            filtered_list.append((d, l, te, t, e, tt, ttt))
    return list(zip(*filtered_list))


def is_valid_entry(dict, keys_lim):
    for k in keys_lim:
        values = keys_lim[k]
        entry = dict[k]
        if isinstance(entry, (bool)):
            if entry != values:
                return False
        elif entry < values[0] or entry > values[1]:
            return False
    return True


def sort_trajs(keys, labels, trajs_euler, trajs, errors, traj_thrust, traj_thrust_t):
    list_sorted =  sorted(zip(keys, labels, trajs_euler, trajs, errors, traj_thrust, traj_thrust_t))
    return list(zip(*list_sorted))


def vel_to_body(pose, vel):
    rotBtoI = R.from_quat(pose[:, 3:7])
    rotItoB = rotBtoI.inv()

    linVel = rotItoB.apply(vel[:, 0:3])
    angVel = rotItoB.apply(vel[:, 3:])

    bVel = np.concatenate([linVel, angVel], axis=-1)
    return bVel


def pose_to_elipse(pose, elipse):
    center = np.array(elipse["center"])
    c_pos = pose[:, 0:3] - center[:, 0]
    c_pose = np.concatenate([c_pos, pose[:, 3:]], axis=-1)
    return c_pose


def desired_angle(pose, elipse, tg=True):
    axis = np.array(elipse["axis"])
    if tg:
        # Perm according to tg computation
        vec = np.zeros(pose[:, :2].shape)
        vec[:, 0] = pose[:, 1]
        vec[:, 1] = pose[:, 0]
        mapping = np.array([-axis[0, 0]/axis[1, 0], axis[1, 0]/axis[0, 0]])
    else:
        vec= pose[:, :2]
        mapping = -np.array([axis[1, 0]/axis[0, 0], axis[0, 0]/axis[1, 0]])
    vec *= mapping
    des_yaw = np.arctan2(vec[:, 1], vec[:, 0])
    des_roll = np.zeros(des_yaw.shape)
    des_pitch = np.zeros(des_yaw.shape)
    angles = np.concatenate([des_roll[:, None], des_pitch[:, None], des_yaw[:, None]], axis=-1)
    r = R.from_euler('XYZ', angles).as_euler('XYZ', degrees=degrees)
    return r


def skew(x):
    res = np.zeros((x.shape[0], 3, 3))

    res[:, 2, 1] = x[:, 0]
    res[:, 1, 2] = -x[:, 0]

    res[:, 1, 0] = x[:, 2]
    res[:, 0, 1] = -x[:, 2]
    
    res[:, 0, 2] = x[:, 1]
    res[:, 2, 0] = -x[:, 1]
    return res


def elipse_angle_error(pose, desired_angle):
    q = pose[:, 3:7]
    rot = R.from_quat(q)
    euler = rot.as_euler('XYZ', degrees=degrees)
    err_rot = np.abs(desired_angle[:, -1] - euler[:, -1])
    if degrees:
        err_rot[err_rot > 180.] = 0.
    else:
        err_rot[err_rot > 3.14] = 0.
    return err_rot


def norm_vel(vel):
    planar_vel = vel[:, 0:2]
    norm_vel = np.linalg.norm(planar_vel, axis=-1)
    return norm_vel


def generate_results_mppi(root, model_name, threshold, mode="buoy", action="forward", legend_entries=[], key_filtering=None, gifs=True, save=True, result_dir=None, show=False):
    print("="*5, f" Generating results for {legend_entries}", 5*"=")

    if save:
        if result_dir is None:
            result_dir = os.path.join(root, "imgs")

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

    axis_dict = {'up': 2, 'down': 2, 'forward': 0, 'back': 0}
    axis = axis_dict[action]

    path = os.path.join(root, "static_cost_" + mode, "bags", action + "*.bag")

    bag_files = glob.glob(path)
    base_names = [os.path.splitext(bag_file)[0] for bag_file in bag_files]
    goal_files = [base_name + "_cost.yaml" for base_name in base_names]
    conf_files = [base_name + "_conf.yaml" for base_name in base_names]
    param_files = [base_name + "_params.yaml" for base_name in base_names]
    seq_imgs_dirs = [base_name + "_seq_imgs" for base_name in base_names]

    bags = [bagreader(bag_file, verbose=False) for bag_file in bag_files]

    pose_topic = os.path.join(model_name, 'pose_gt')

    thrusters_topics = [os.path.join(model_name, 'thrusters', thruster, 'input') for thruster in ['0', '1', '2', '3', '4', '5']]

    trajs = [pd.read_csv(bag.message_by_topic(pose_topic)) for bag in bags]
    thrusters_inputs = [[pd.read_csv(bag.message_by_topic(thruster)) for thruster in thrusters_topics] for bag in bags]

    pose_entries = ['position.x', 'position.y', 'position.z', 'orientation.x', 'orientation.y', 'orientation.z', 'orientation.w']
    vel_entries = ['linear.x', 'linear.y', 'linear.z', 'angular.x', 'angular.y', 'angular.z']

    trajs_pose = [traj[[f'pose.pose.{e}' for e in pose_entries]].to_numpy() for traj in trajs]
    trajs_time = [traj['Time'].to_numpy() for traj in trajs]
    trajs_vel = [traj[[f'twist.twist.{e}' for e in vel_entries]].to_numpy() for traj in trajs]

    trajs_thrust = [[thruster['data'].to_numpy() for thruster in thrusters_input] for thrusters_input in thrusters_inputs]
    trajs_thrust_t = [[thruster['Time'].to_numpy() for thruster in thrusters_input] for thrusters_input in thrusters_inputs]
    
    goals = [load_goal(goal_file)[:7] for goal_file in goal_files]
    param_dicts = [load_yaml(param_file) for param_file in param_files]

    goal_euler = to_euler(goals[0])[:6]
    trajs_euler = [traj_to_euler(traj_pose)[:, :6] for traj_pose in trajs_pose]
    # response_time = [traj.iloc[response_time_idx(traj_pose, goal_euler, axis, threshold)]['Time'] - traj.iloc[0]['Time'] for traj_pose, traj in zip(trajs_euler, trajs)]
    # steady_errors = [np.mean(steady_error(traj_pose, goal_euler, 400, axis, threshold), axis=0) for traj_pose in trajs_euler]
    # energies = [energy_estimate(traj_thrust, 500) for traj_thrust in trajs_thrust]
    # mses = [mse_t(traj_euler, goal_euler) for traj_euler in trajs_euler]

    errors = [error_traj(traj_euler, goal_euler) for traj_euler in trajs_euler]

    # metrics_dict = gen_metric_dict(controllers, response_time, steady_errors, energies, mses, errors)
    # metric_df = pd.DataFrame.from_dict(metrics_dict)
    # metric_df.to_csv(f"res_{mode}.csv")

    labels = [get_label_text(legend_entries, param_dict) for param_dict in param_dicts]
    # assumes that the key to sort the plots with is the fist entry of the legend_entries
    if key_filtering is not None:
        param_dicts, labels, trajs_euler, trajs, errors, trajs_thrust, trajs_thrust_t = filter_trajs(param_dicts, key_filtering, labels, trajs_euler, trajs, errors, trajs_thrust, trajs_thrust_t)

    keys = [get_dict_value(legend_entries[0], param_dict) for param_dict in param_dicts]
    keys, labels, trajs_euler, trajs, errors, trajs_thrust, trajs_thrust_t = sort_trajs(keys, labels, trajs_euler, trajs, errors, trajs_thrust, trajs_thrust_t)

    # if len(labels) >= 6:
    #     bbox = (0.5, 0.7, 0.5, 0.5)
    # elif len(labels) == 5:
    #     bbox = (0.5, 0.625, 0.5, 0.5)
    # elif len(labels) == 4:
    #     bbox = (0.5, 0.575, 0.5, 0.5)
    # elif len(labels) <= 3:
    #     bbox = (0.5, 0.5, 0.5, 0.5)

    bbox = (0., 0., 1., 0.75)
    ########################
    #### Pose evolution ####
    ########################
    print("|", " Generating pose plots ")
    #fig.suptitle(f"{legend_entries[0]} Pose Evolution".upper(), fontsize=title_fontsize)
    #fig.set_figheight(10)
    #fig.set_figwidth(15)
    figs = [plt.figure() for x in range(6)]
    axes = [fig.add_axes([0.2, 0.135, 0.77, 0.85]) for fig in figs]

    for l, traj_euler, traj in zip(labels, trajs_euler, trajs):
        # Position
        axes[0].plot(traj['Time'] - traj['Time'][0], traj_euler[:, 0], label=l.upper())
        axes[1].plot(traj['Time'] - traj['Time'][0], traj_euler[:, 1], label=l.upper())
        axes[2].plot(traj['Time'] - traj['Time'][0], traj_euler[:, 2], label=l.upper())
        # Angles
        axes[3].plot(traj['Time'] - traj['Time'][0], traj_euler[:, 3], label=l.upper())
        axes[4].plot(traj['Time'] - traj['Time'][0], traj_euler[:, 4], label=l.upper())
        axes[5].plot(traj['Time'] - traj['Time'][0], traj_euler[:, 5], label=l.upper())
    
    axis_labels = ['x [m]', 'y [m]', 'z [m]', 'roll [rad]', 'pitch [rad]', 'yaw [rad]']
    for ax, l in zip(axes, axis_labels):
        ax.set_xlabel("Time [s]", fontsize=label_fontsize)
        ax.set_ylabel(l, fontsize=label_fontsize)
        ax.set_xlim(0, 60.)
        ax.tick_params(labelsize=tick_size)
        ax.grid(linestyle='--', linewidth=0.5)

    axes[1].set_ylim(-1, 1)
    if axis == 2:
        axes[0].set_ylim(-3, 3)
    if axis == 0:
        axes[2].set_ylim(-53, -47)
    axes[3].set_ylim(-ang_limit, ang_limit)
    axes[4].set_ylim(-ang_limit, ang_limit)
    axes[5].set_ylim(-ang_limit, ang_limit)

    title = get_label_title(legend_entries)
    if axis == 0:
        axes[0].legend(loc='lower right', bbox_to_anchor=bbox, fontsize=legend_fontsize,
                       title=title.upper(), title_fontsize=legend_title_size)
    elif axis == 2:
        axes[2].legend(loc='lower right', bbox_to_anchor=bbox, fontsize=legend_fontsize,
                       title=title.upper(), title_fontsize=legend_title_size)

    # plt.tight_layout()
    if save:
        axes_name = ["x", "y", "z", "roll", "pitch", "yaw"]
        for fig, ax in zip(figs, axes_name):
            fig.savefig(os.path.join(result_dir, f"{legend_entries[0]}-pose-{ax}.pdf"))
    if show:
        plt.show()
    for fig in figs:
        plt.close(fig)

    ########################
    ####    Error(t)    ####
    ########################
    print("|", " Generating error plots ")
    figs = [plt.figure() for x in range(6)]
    axes = [fig.add_axes([0.22, 0.125, 0.755, 0.855]) for fig in figs]

    for l, err, traj in zip(labels, errors, trajs):
        # Position
        axes[0].plot(traj['Time'] - traj['Time'][0], err[:, 0], label=l.upper())
        axes[1].plot(traj['Time'] - traj['Time'][0], err[:, 1], label=l.upper())
        axes[2].plot(traj['Time'] - traj['Time'][0], err[:, 2], label=l.upper())
        # Angles
        axes[3].plot(traj['Time'] - traj['Time'][0], err[:, 3], label=l.upper())
        axes[4].plot(traj['Time'] - traj['Time'][0], err[:, 4], label=l.upper())
        axes[5].plot(traj['Time'] - traj['Time'][0], err[:, 5], label=l.upper())

    axis_labels = ['x [m]', 'y [m]', 'z [m]', 'roll [rad]', 'pitch [rad]', 'yaw [rad]']
    for ax, l in zip(axes, axis_labels):
        ax.set_xlabel("Time [s]", fontsize=label_fontsize)
        ax.set_ylabel(l, fontsize=label_fontsize)
        ax.set_xlim(0, 60.)
        ax.tick_params(labelsize=tick_size)
        ax.grid(linestyle='--', linewidth=0.5)

    title = get_label_title(legend_entries)
    if axis == 0:
        axes[0].legend(loc='lower right', bbox_to_anchor=bbox, fontsize=legend_fontsize,
                       title=title.upper(), title_fontsize=legend_title_size)
    elif axis == 2:
        axes[2].legend(loc='lower right', bbox_to_anchor=bbox, fontsize=legend_fontsize,
                       title=title.upper(), title_fontsize=legend_title_size)
    
    if save:
        axes_name = ["x", "y", "z", "roll", "pitch", "yaw"]
        for fig, ax in zip(figs, axes_name):
            fig.savefig(os.path.join(result_dir, f"{legend_entries[0]}-error-{ax}.pdf"))
    if show:
        plt.show()
    for fig in figs:
        plt.close(fig)

    ########################
    #### Thruster input ####
    ########################
    print("|", " Generating thruster plots ")
    figs = [plt.figure() for x in range(6)]
    axes = [fig.add_axes([0.095, 0.125, 0.88, 0.85]) for fig in figs]

    for l, traj_thrust, traj_thrust_t, traj in zip(labels, trajs_thrust, trajs_thrust_t, trajs):
        axes[0].plot(traj_thrust_t[0] - traj['Time'][0], traj_thrust[0], label=l.upper())
        axes[1].plot(traj_thrust_t[1] - traj['Time'][0], traj_thrust[1], label=l.upper())
        axes[2].plot(traj_thrust_t[2] - traj['Time'][0], traj_thrust[2], label=l.upper())
        axes[3].plot(traj_thrust_t[3] - traj['Time'][0], traj_thrust[3], label=l.upper())
        axes[4].plot(traj_thrust_t[4] - traj['Time'][0], traj_thrust[4], label=l.upper())
        axes[5].plot(traj_thrust_t[5] - traj['Time'][0], traj_thrust[5], label=l.upper())


    t = "Thruster"
    axis_labels = [f'{t} 0', f'{t} 1', f'{t} 2', f'{t} 3', f'{t} 4', f'{t} 5']
    for ax, l in zip(axes, axis_labels):
        ax.set_xlabel("Time [s]", fontsize=label_fontsize)
        #ax.set_title(l, fontsize=label_fontsize)
        ax.set_xlim(0, 60.)
        ax.set_ylim(-260, 260)
        ax.tick_params(labelsize=tick_size)
        ax.grid(linestyle='--', linewidth=0.5)



    title = get_label_title(legend_entries)
    if axis == 0:
        axes[0].legend(loc='lower right', bbox_to_anchor=bbox, fontsize=legend_fontsize,
                       title=title.upper(), title_fontsize=legend_title_size)
    elif axis == 2:
        axes[2].legend(loc='lower right', bbox_to_anchor=bbox, fontsize=legend_fontsize,
                       title=title.upper(), title_fontsize=legend_title_size)

    if save:
        axes_name = ["t0", "t1", "t2", "t3", "t4", "t5"]
        for fig, ax in zip(figs, axes_name):
            fig.savefig(os.path.join(result_dir, f"{legend_entries[0]}-thruster-{ax}.pdf"))
    if show:
        plt.show()
    for fig in figs:
        plt.close(fig)

    if gifs:
        print("|", " Generating Action Sequence Gifs.")
        [gen_gif(seq_imgs_dir, result_dir, os.path.split(seq_imgs_dir)[-1]) for seq_imgs_dir in seq_imgs_dirs]


def generate_restuls_controllers(root, model_name, threshold, mode="buoy", action="forward", controller_list=[], save=True, result_dir=None, show=False):
    print("="*5, f" Generating Comparative results for controller {controller_list} and action {action} ", 5*"=")

    if save:
        if result_dir is None:
            result_dir = os.path.join(root, "results")

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

    axis_dict = {"up": 2, "down": 2, "forward": 0, "back": 0}
    axis = axis_dict[action]

    controller_files = [glob.glob(os.path.join(root, controller, "static_cost_" + mode, "bags", action + "*.bag")) for controller in controller_list]

    labels = [[c + " " + os.path.split(f)[1] for f in fs] for c, fs in zip(controller_list, controller_files)]

    controller_base_names = [[os.path.splitext(bag_file)[0] for bag_file in bag_files]
                for bag_files in controller_files]
    c_goal_files = [[base_name + "_cost.yaml" for base_name in base_names]
                for base_names in controller_base_names]

    c_bags = [[bagreader(bag_file, verbose=False) for bag_file in bag_files] for bag_files in controller_files]

    pose_topic = os.path.join(model_name, 'pose_gt')
    thrusters_topics = [os.path.join(model_name, 'thrusters', thruster, 'input') for thruster in ['0', '1', '2', '3', '4', '5']]


    c_trajs = [[pd.read_csv(bag.message_by_topic(pose_topic)) for bag in bags] for bags in c_bags]
    c_thrusters_inputs = [[[pd.read_csv(bag.message_by_topic(thruster)) for thruster in thrusters_topics] for bag in bags] for bags in c_bags]

    pose_entries = ['position.x', 'position.y', 'position.z', 'orientation.x', 'orientation.y', 'orientation.z', 'orientation.w']
    vel_entries = ['linear.x', 'linear.y', 'linear.z', 'angular.x', 'angular.y', 'angular.z']

    c_trajs_pose = [[traj[[f'pose.pose.{e}' for e in pose_entries]].to_numpy() for traj in trajs] for trajs in c_trajs]
    c_trajs_time = [[traj['Time'].to_numpy() for traj in trajs] for trajs in c_trajs]
    c_trajs_vel = [[traj[[f'twist.twist.{e}' for e in vel_entries]].to_numpy() for traj in trajs] for trajs in c_trajs]

    c_trajs_thrust = [[[thruster['data'].to_numpy() for thruster in thrusters_input] for thrusters_input in thrusters_inputs] for thrusters_inputs in c_thrusters_inputs]
    c_trajs_thrust_t = [[[thruster['Time'].to_numpy() for thruster in thrusters_input] for thrusters_input in thrusters_inputs] for thrusters_inputs in c_thrusters_inputs]

    c_goals = [[load_goal(goal_file)[:7] for goal_file in goal_files] for goal_files in c_goal_files]
    c_goal_euler = [to_euler(goals[0])[:6] for goals in c_goals]

    c_trajs_euler = [[traj_to_euler(traj_pose)[:, :6] for traj_pose in trajs_pose] for trajs_pose in c_trajs_pose]

    c_errors = [[error_traj(traj_euler, goal_euler) for traj_euler in trajs_euler] for (trajs_euler, goal_euler) in zip(c_trajs_euler, c_goal_euler)]


    bbox = (0., 0., 1., 0.75)

    print("|", " Generating pose plots ")
    figs = [plt.figure() for x in range(6)]
    axes = [fig.add_axes([0.2, 0.135, 0.77, 0.85]) for fig in figs]

    for c, trajs_euler, trajs in zip(controller_list, c_trajs_euler, c_trajs):
        if c == "mppi":
            c += " (ours)"
            linestyle = "--"
        else:
            linestyle = "-"
        for traj_euler, traj in zip(trajs_euler, trajs):
            # Position
            axes[0].plot(traj['Time'] - traj['Time'][0], traj_euler[:, 0], label=c.upper(), linestyle=linestyle)
            axes[1].plot(traj['Time'] - traj['Time'][0], traj_euler[:, 1], label=c.upper(), linestyle=linestyle)
            axes[2].plot(traj['Time'] - traj['Time'][0], traj_euler[:, 2], label=c.upper(), linestyle=linestyle)

            # Angles
            axes[3].plot(traj['Time'] - traj['Time'][0], traj_euler[:, 3], label=c.upper(), linestyle=linestyle)
            axes[4].plot(traj['Time'] - traj['Time'][0], traj_euler[:, 4], label=c.upper(), linestyle=linestyle)
            axes[5].plot(traj['Time'] - traj['Time'][0], traj_euler[:, 5], label=c.upper(), linestyle=linestyle)

    axis_labels = ['x [m]', 'y [m]', 'z [m]', 'roll [rad]', 'pitch [rad]', 'yaw [rad]']
    for ax, l in zip(axes, axis_labels):
        ax.set_xlabel("Time [s]", fontsize=label_fontsize)
        ax.set_ylabel(l, fontsize=label_fontsize)
        ax.set_xlim(0, 60.)
        ax.tick_params(labelsize=tick_size)
        ax.grid(linestyle='--', linewidth=0.5)

    axes[1].set_ylim(-1, 1)
    if axis == 2:
        axes[0].set_ylim(-3, 3)
    if axis == 0:
        axes[2].set_ylim(-53, -47)    
    axes[3].set_ylim(-ang_limit, ang_limit)
    axes[4].set_ylim(-ang_limit, ang_limit)
    axes[5].set_ylim(-ang_limit, ang_limit)

    if axis == 0:
        axes[0].legend(loc='lower right', bbox_to_anchor=bbox, fontsize=legend_fontsize)
    elif axis == 2:
        axes[2].legend(loc='lower right', bbox_to_anchor=bbox, fontsize=legend_fontsize)

    if save:
        axes_name = ["x", "y", "z", "roll", "pitch", "yaw"]
        for fig, ax in zip(figs, axes_name):
            fig.savefig(os.path.join(result_dir, f"{mode}-{action}-pose-{ax}.pdf"))


    if show:
        plt.show()
    for fig in figs:
        plt.close(fig)



    ########################
    ####    Error(t)    ####
    ########################
    print("|", " Generating error plots ")
    figs = [plt.figure() for x in range(6)]
    axes = [fig.add_axes([0.18, 0.125, 0.795, 0.86]) for fig in figs]

    for c, errors, trajs in zip(controller_list, c_errors, c_trajs):
        if c == "mppi":
            c += " (ours)"
            linestyle = "--"
        else:
            linestyle = "-"
        for err, traj in zip(errors, trajs):
            # Position
            axes[0].plot(traj['Time'] - traj['Time'][0], err[:, 0], label=c.upper(), linestyle=linestyle)
            axes[1].plot(traj['Time'] - traj['Time'][0], err[:, 1], label=c.upper(), linestyle=linestyle)
            axes[2].plot(traj['Time'] - traj['Time'][0], err[:, 2], label=c.upper(), linestyle=linestyle)

            # Angles
            axes[3].plot(traj['Time'] - traj['Time'][0], err[:, 3], label=c.upper(), linestyle=linestyle)
            axes[4].plot(traj['Time'] - traj['Time'][0], err[:, 4], label=c.upper(), linestyle=linestyle)
            axes[5].plot(traj['Time'] - traj['Time'][0], err[:, 5], label=c.upper(), linestyle=linestyle)

    axis_labels = ['x [m]', 'y [m]', 'z [m]', 'roll [rad]', 'pitch [rad]', 'yaw [rad]']
    for ax, l in zip(axes, axis_labels):
        ax.set_xlabel("Time [s]", fontsize=label_fontsize)
        ax.set_ylabel(l, fontsize=label_fontsize)
        ax.set_xlim(0, 60.)
        ax.tick_params(labelsize=tick_size)
        ax.grid(linestyle='--', linewidth=0.5)


    if axis == 0:
        axes[0].legend(loc='lower right', bbox_to_anchor=bbox, fontsize=legend_fontsize)
    elif axis == 2:
        axes[2].legend(loc='lower right', bbox_to_anchor=bbox, fontsize=legend_fontsize)


    if save:
        axes_name = ["x", "y", "z", "roll", "pitch", "yaw"]
        for fig, ax in zip(figs, axes_name):
            fig.savefig(os.path.join(result_dir, f"{mode}-{action}-error-{ax}.pdf"))
    if show:
        plt.show()
    for fig in figs:
        plt.close(fig)



    ########################
    #### Thruster input ####
    ########################
    print("|", " Generating thruster plots ")
    figs = [plt.figure() for x in range(6)]
    axes = [fig.add_axes([0.095, 0.065, 0.88, 0.875]) for fig in figs]

    for c, trajs_thrust, trajs_thrust_t, trajs in zip(controller_list, c_trajs_thrust, c_trajs_thrust_t, c_trajs):
        if c == "mppi":
            c += " (ours)"
            linestyle = "--"
        else:
            linestyle = "-"
        for traj_thrust, traj_thrust_t, traj in zip(trajs_thrust, trajs_thrust_t, trajs):
            axes[0].plot(traj_thrust_t[0] - traj['Time'][0], traj_thrust[0], label=c.upper(), linestyle=linestyle)
            axes[1].plot(traj_thrust_t[1] - traj['Time'][0], traj_thrust[1], label=c.upper(), linestyle=linestyle)
            axes[2].plot(traj_thrust_t[2] - traj['Time'][0], traj_thrust[2], label=c.upper(), linestyle=linestyle)
            # Angles
            axes[3].plot(traj_thrust_t[3] - traj['Time'][0], traj_thrust[3], label=c.upper(), linestyle=linestyle)
            axes[4].plot(traj_thrust_t[4] - traj['Time'][0], traj_thrust[4], label=c.upper(), linestyle=linestyle)
            axes[5].plot(traj_thrust_t[5] - traj['Time'][0], traj_thrust[5], label=c.upper(), linestyle=linestyle)

    t = "Thruster"
    axis_labels = [f'{t} 0', f'{t} 1', f'{t} 2', f'{t} 3', f'{t} 4', f'{t} 5']
    for ax, l in zip(axes, axis_labels):
        ax.set_xlabel("Time [s]", fontsize=label_fontsize)
        ax.set_title(l, fontsize=label_fontsize)
        ax.set_xlim(0, 60.)
        ax.set_ylim(-260, 260)
        ax.tick_params(labelsize=tick_size)
        ax.grid(linestyle='--', linewidth=0.5)


    if axis == 0:
        axes[0].legend(loc='lower right', bbox_to_anchor=bbox, fontsize=legend_fontsize)
    elif axis == 2:
        axes[2].legend(loc='lower right', bbox_to_anchor=bbox, fontsize=legend_fontsize)


    if save:
        axes_name = ["t0", "t1", "t2", "t3", "t4", "t5"]
        for fig, ax in zip(figs, axes_name):
            fig.savefig(os.path.join(result_dir, f"{mode}-{action}-thruster-{ax}.pdf"))
    if show:
        plt.show()
    for fig in figs:
        plt.close(fig)


def filter_tg(bags, elipses, tg):
    filtered_bags = []
    filtered_elipse = []
    for (bag, elipse) in zip(bags, elipses):
        filtered_b = []
        filtered_e = []
        for b, e in zip(bag, elipse):
            if e["tg"] == tg:
                filtered_b.append(b)
                filtered_e.append(e)
        filtered_bags.append(filtered_b)
        filtered_elipse.append(filtered_e)
    return filtered_bags, filtered_elipse


def gen_elipse(elipse):
    axis = elipse["axis"]
    a = axis[0][0]
    b = axis[1][0]
    theta = np.linspace(0, 2*np.pi, 500)
    center = elipse["center"]
    cx = center[0][0]
    cy = center[1][0]

    x = a*np.cos(theta) + cx
    y = b*np.sin(theta) + cy

    return x, y


def generate_elipse_results(root, model_name, threshold, modes, tg=True, save=True, result_dir=None, show=False):
    print("="*5, f" Generating results for ellipse cost ", 5*"=")
    if save:
        if result_dir is None:
            result_dir = os.path.join(root, "results")

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    # First extract the bag, the conf, the model, the cost and the param
    # change the velocity to express it in body frame
    # Change the pose to express it in elipse frame. As the ellipse is coplanar with the XY plane.

    # For every trajectory generate 3 Subplots. 
    # 1. For X and Y with color equal to norm of planar velocity
    # 2. The value of Z over time
    # 3. The angular error, compute the desired angle for a given (x, y) position and find the angle between the two orientation

    ###########################
    # EXTRACT BAGS AND CONFS  #
    ###########################

    bag_mode_files = [glob.glob(os.path.join(root, "elipse_cost_" + mode, "bags", "*.bag")) for mode in modes]
    
    # labels = [[c + " " + os.path.split(f)[1] for f in fs] for c, fs in zip(controller_list, controller_files)

    bag_base_names = [[os.path.splitext(bag_file)[0] for bag_file in bag_files]
                for bag_files in bag_mode_files]
    m_goal_files = [[base_name + "_cost.yaml" for base_name in base_names]
                    for base_names in bag_base_names]


    m_bags = [[bagreader(bag_file, verbose=False) for bag_file in bag_files] for bag_files in bag_mode_files]
    m_elipse = [[load_yaml(goal_file) for goal_file in goal_files] for goal_files in m_goal_files]


    m_bags, m_elipse = filter_tg(m_bags, m_elipse, tg)

    pose_topic = os.path.join(model_name, 'pose_gt')
    thrusters_topics = [os.path.join(model_name, 'thrusters', thruster, 'input') for thruster in ['0', '1', '2', '3', '4', '5']]


    m_trajs = [[pd.read_csv(bag.message_by_topic(pose_topic)) for bag in bags] for bags in m_bags]
    m_thrusters_inputs = [[[pd.read_csv(bag.message_by_topic(thruster)) for thruster in thrusters_topics] for bag in bags] for bags in m_bags]

    pose_entries = ['position.x', 'position.y', 'position.z', 'orientation.x', 'orientation.y', 'orientation.z', 'orientation.w']
    vel_entries = ['linear.x', 'linear.y', 'linear.z', 'angular.x', 'angular.y', 'angular.z']

    m_trajs_pose = [[traj[[f'pose.pose.{e}' for e in pose_entries]].to_numpy() for traj in trajs] for trajs in m_trajs]
    m_trajs_time = [[traj['Time'].to_numpy() for traj in trajs] for trajs in m_trajs]
    m_trajs_vel = [[traj[[f'twist.twist.{e}' for e in vel_entries]].to_numpy() for traj in trajs] for trajs in m_trajs]

    m_trajs_thrust = [[[thruster['data'].to_numpy() for thruster in thrusters_input] for thrusters_input in thrusters_inputs] for thrusters_inputs in m_thrusters_inputs]
    m_trajs_thrust_t = [[[thruster['Time'].to_numpy() for thruster in thrusters_input] for thrusters_input in thrusters_inputs] for thrusters_inputs in m_thrusters_inputs]

    m_trajs_euler = [[traj_to_euler(traj_pose)[:, :6] for traj_pose in trajs_pose] for trajs_pose in m_trajs_pose]

    ###########################
    # COMPUTE PLANAR VELOCITY #
    ###########################
    m_trajs_vel_body = [[vel_to_body(pose, vel) for pose, vel in zip(poses, vels)] for poses, vels in zip(m_trajs_pose, m_trajs_vel)]
    m_trajs_planar_vel = [[norm_vel(b_vel) for b_vel in b_vels] for b_vels in m_trajs_vel_body]

    ###########################
    #  COMPUTE ANGULAR ERROR  #
    ###########################
    m_pose_elipse = [[pose_to_elipse(pose, elipse) for pose, elipse in zip(poses, elipses)] for poses, elipses in zip(m_trajs_pose, m_elipse)]
    m_desired_angle = [[desired_angle(pose_e, elipse, tg) for pose_e, elipse in zip(poses_e, elipses)] for poses_e, elipses in zip(m_pose_elipse, m_elipse)]
    m_angle_errors = [[elipse_angle_error(pose, des_angle) for pose, des_angle in zip(poses, des_angles)] for poses, des_angles in zip(m_trajs_pose, m_desired_angle)]


    ###########################
    # PLOTTING ALL THE VALUES #
    ###########################

    fig, axes = plt.subplots(2, 2)    
    if tg:
        ang_label = "Tangent"
    else:
        ang_label = "Perpendicular"
    #fig.suptitle(f"Ellipse Cost {ang_label}".upper(), fontsize=title_fontsize)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    # TODO plot gt elipse from cost file.
    x, y = gen_elipse(m_elipse[0][0])
    axes[0, 0].plot(x, y, label="Target".upper(), color="r")

    print("|", " Generating pose plots ")
    elipse = m_elipse[0][0]
    center = np.array(elipse["center"])
    axis = np.array(elipse["axis"])
    axes[0, 0].set_xlim(center[0, 0]-axis[0, 0]-3, center[0, 0]+axis[0, 0]+1)
    axes[0, 0].set_ylim(center[1, 0]-axis[1, 0]-3, center[1, 0]+axis[1, 0]+1)
    axes[0, 0].set_ylabel("y [m]", fontsize=label_fontsize)
    axes[0, 0].set_xlabel("x [m]", fontsize=label_fontsize)
    axes[0, 0].tick_params(labelsize=tick_size)
    # First plot GT ellipse
    for trajs, v_trajs, elipses, mode in zip(m_trajs_euler, m_trajs_vel_body, m_elipse, modes):
        for t, v, e in zip(trajs, v_trajs, elipses):
            # points = np.array([t[:, 0], t[:, 1]]).T.reshape(-1, 1, 2)
            # segments = np.concatenate([points[:-1], points[1:]], axis=1)
            # norm = plt.Normalize(v.min(), v.max())
            # lc = LineCollection(segments, cmap='viridis', norm=norm)
            # lc.set_array(v[:-1])
            # line = axes[0].add_collection(lc)
            # fig.colorbar(line, ax=axes[0])
            axes[0, 0].plot(t[:, 0], t[:, 1], label=mode.upper())
    #axes[0, 0].legend(loc='upper right', bbox_to_anchor=(1.1, 1., 0., 0.), fontsize=legend_fontsize)

    print("|", " Generating angular error plots ")
    for angels_error, des_ang, angs, trajs_t, mode in zip(m_angle_errors, m_desired_angle, m_trajs_euler, m_trajs_time, modes):
        for ang_err, d_ang, ang, time in zip(angels_error, des_ang, angs, trajs_t):
            axes[0, 1].plot(time-time[0], ang_err, label=mode.upper())
            #xes[0, 1].set_ylim(0, 3.15)
    if degrees:
        axes[0, 1].set_ylim(-0.1, 60.)
    else:
        axes[0, 1].set_ylim(-0.1, 3.14/3.)
    axes[0, 1].set_ylabel("Yaw error [rad]", fontsize=label_fontsize)
    axes[0, 1].set_xlabel("Time [s]", fontsize=label_fontsize)
    axes[0, 1].tick_params(labelsize=tick_size)
    
    
    print("|", " Generating z error plots ")
    for trajs, trajs_t, mode in zip(m_pose_elipse, m_trajs_time, modes):
        for t, time in zip(trajs, trajs_t):
            axes[1, 0].plot(time-time[0], t[:, 2], label=mode.upper())
    axes[1, 0].set_ylim(-1., 1.)
    axes[1, 0].set_ylabel("z error [m]", fontsize=label_fontsize)
    axes[1, 0].set_xlabel("Time [s]", fontsize=label_fontsize)
    axes[1, 0].tick_params(labelsize=tick_size)

    
    print("|", " Generating velocity plots ")
    for v_trajs, trajs_t, mode in zip(m_trajs_planar_vel, m_trajs_time, modes):
        for v, time in zip(v_trajs, trajs_t):
            axes[1, 1].plot(time-time[0], v, label=mode.upper())
    axes[1, 1].set_ylim(-0.2, 1.5)
    axes[1, 1].set_ylabel("Velocity [m/s]", fontsize=label_fontsize)
    axes[1, 1].set_xlabel("Time [s]", fontsize=label_fontsize)
    axes[1, 1].tick_params(labelsize=tick_size)

    axes[0, 0].grid(linestyle = '--', linewidth = 0.5)
    axes[0, 1].grid(linestyle = '--', linewidth = 0.5)
    axes[1, 0].grid(linestyle = '--', linewidth = 0.5)
    axes[1, 1].grid(linestyle = '--', linewidth = 0.5)

    #axes[0, 1].legend(loc='upper right', bbox_to_anchor=(1.1, 1., 0., 0.), fontsize=legend_fontsize)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(result_dir, f"elipse-plot-{ang_label}.pdf"))
    if show:
        plt.show()
    plt.close()


def trim_data(p, h, ylim, zlim, axis="z"):
    if axis == "x":
        pass
    if axis == "y":
        h = np.abs(ylim[0] - ylim[1])
        p[1] = ylim[0]
    if axis == "z":
        h = np.abs(zlim[0] - zlim[1])
        p[2] = zlim[0]
    return p, h


def data_for_cylinder(center_x, center_y, center_z, radius, height, axis="z"):
    # Generate the plot assuming main axis is z
    points = 500
    main_axis = np.linspace(0, height, points)
    theta = np.linspace(0, 2*np.pi, points)
    theta_grid, main_grid=np.meshgrid(theta, main_axis)
    second_grid = radius*np.cos(theta_grid)
    third_grid = radius*np.sin(theta_grid)
    if axis == "x":
        return main_grid + center_x, second_grid + center_y, third_grid + center_z
    if axis == "y":
        return third_grid + center_x, main_grid + center_y, second_grid + center_z
    if axis == "z":
        return second_grid + center_x, third_grid + center_y, main_grid + center_z


def get_modes_label(modes):
    labels = []
    for m in modes:
        if m == "buoy":
            labels.append("neutral")
        elif m == "pos":
            labels.append("positive")
        elif m == "neg":
            labels.append("negative")
    return labels


def generate_obs_results(root, model_name, threshold, modes, save=True, result_dir=None, show=False):
    print("="*5, f" Generating results for Obstacle cost ", 5*"=")
    if save:
        if result_dir is None:
            result_dir = os.path.join(root, "results")

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    
    # Generate 3D plot of trajectory with obstacles
    # Generate 2D XY plot of trajectory 
    # Generate 2D XZ plot of trajectory

    ###########################
    # EXTRACT BAGS AND CONFS  #
    ###########################

    bag_mode_files = [glob.glob(os.path.join(root, "static_cost_" + mode, "bags", "*.bag")) for mode in modes]
    bag_base_names = [[os.path.splitext(bag_file)[0] for bag_file in bag_files]
                for bag_files in bag_mode_files]
    m_goal_files = [[base_name + "_cost.yaml" for base_name in base_names]
                    for base_names in bag_base_names]


    m_bags = [[bagreader(bag_file, verbose=False) for bag_file in bag_files] for bag_files in bag_mode_files]
    m_cost = [[load_yaml(goal_file) for goal_file in goal_files] for goal_files in m_goal_files]

    pose_topic = os.path.join(model_name, 'pose_gt')
    thrusters_topics = [os.path.join(model_name, 'thrusters', thruster, 'input') for thruster in ['0', '1', '2', '3', '4', '5']]


    m_trajs = [[pd.read_csv(bag.message_by_topic(pose_topic)) for bag in bags] for bags in m_bags]
    m_thrusters_inputs = [[[pd.read_csv(bag.message_by_topic(thruster)) for thruster in thrusters_topics] for bag in bags] for bags in m_bags]

    pose_entries = ['position.x', 'position.y', 'position.z', 'orientation.x', 'orientation.y', 'orientation.z', 'orientation.w']
    vel_entries = ['linear.x', 'linear.y', 'linear.z', 'angular.x', 'angular.y', 'angular.z']

    m_trajs_pose = [[traj[[f'pose.pose.{e}' for e in pose_entries]].to_numpy() for traj in trajs] for trajs in m_trajs]
    m_trajs_time = [[traj['Time'].to_numpy() for traj in trajs] for trajs in m_trajs]
    m_trajs_vel = [[traj[[f'twist.twist.{e}' for e in vel_entries]].to_numpy() for traj in trajs] for trajs in m_trajs]

    m_trajs_thrust = [[[thruster['data'].to_numpy() for thruster in thrusters_input] for thrusters_input in thrusters_inputs] for thrusters_inputs in m_thrusters_inputs]
    m_trajs_thrust_t = [[[thruster['Time'].to_numpy() for thruster in thrusters_input] for thrusters_input in thrusters_inputs] for thrusters_inputs in m_thrusters_inputs]

    m_trajs_euler = [[traj_to_euler(traj_pose)[:, :6] for traj_pose in trajs_pose] for trajs_pose in m_trajs_pose]

    # PLOT 3D plot

    goal = m_cost[0][0]['goals'][0]

    bbox = (0.1, 0.1, 1., 0.75)
    print("|", " Generating 3D plot")

    fig = plt.figure()
    #fig.suptitle(f"3D Trajectory".upper(), fontsize=title_fontsize)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    ax = fig.add_subplot(111, projection='3d')
    ylim = (-3., 3.)
    zlim = (-53., -47.)
    ax.set_xlim3d(-0.1, 15.1)
    ax.set_ylim3d(ylim[0], ylim[1])
    ax.set_zlim3d(zlim[0], zlim[1])
    
    ax.set_xlabel("\nx [m]", fontsize=label_fontsize)
    ax.set_ylabel("y [m]", fontsize=label_fontsize)
    ax.set_zlabel("\nz [m]", fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_size)

    for c in m_cost[0]:
        for obs in c["obs"]:
            axis = {0: 'x', 1: 'y', 2: "z"}
            p1 = np.array(c["obs"][obs]["p1"])
            p2 = np.array(c["obs"][obs]["p2"])
            r = c["obs"][obs]["r"]

            h = np.abs(p1 - p2)
            foo = np.where(h > 0.)[0]
            a = axis[foo[0]]
            h = h[foo[0]]
            p1, h = trim_data(p1, h, ylim, zlim, a)
            Xc,Yc,Zc = data_for_cylinder(p1[0], p1[1], p1[2], r-0.5, h, a)
            ax.plot_surface(Xc, Yc, Zc, alpha=0.3, color='r')

    modes_label = get_modes_label(modes)

    for trajs_pose, mode in zip(m_trajs_pose, modes_label):
        for traj in trajs_pose:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=mode.upper(), linewidth=2.)
    
    ax.scatter(goal[0], goal[1], goal[2], s=30, color='k')
    ax.set_box_aspect((15.2, 6, 6))
    ax.tick_params(labelsize=tick_size)
    ax.grid(linestyle='--', linewidth=0.5)
    ax.legend(loc='lower left', bbox_to_anchor=bbox, fontsize=legend_fontsize, title="buoyancy".upper(), title_fontsize=legend_title_size)
    plt.tight_layout()
    
    #ax.legend(fontsize=legend_fontsize)


    if save:
        plt.savefig(os.path.join(result_dir, f"obs-3d-plot.pdf"))
    if show:
        plt.show()
    plt.close()

    ##########################
    #      PLANAR PLOTS      #
    ##########################

    print("|", " Generating Coplanar plots ")

    fig, axes = plt.subplots(2, 1, sharex=True)
    #fig.suptitle(f"Trajectories Projection".upper(), fontsize=title_fontsize)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    
    for trajs_pose, mode in zip(m_trajs_pose, modes):
        for traj in trajs_pose:
            axes[0].plot(traj[:, 0], traj[:, 1], label=mode.upper(), linewidth=2.)

    axes[0].scatter(goal[0], goal[1], s=50, color='k', zorder=2)

    for trajs_pose, mode in zip(m_trajs_pose, modes):
        for traj in trajs_pose:
            axes[1].plot(traj[:, 0], traj[:, 2], label=mode.upper(), linewidth=2.)

    axes[1].scatter(goal[0], goal[2], s=50, color='k', zorder=2)

    axes[1].set_xlabel("x [m]", fontsize=label_fontsize+5)
    axes[0].set_ylabel("y [m]", fontsize=label_fontsize+5)
    axes[1].set_ylabel("z [m]", fontsize=label_fontsize+5)
    
    axes[0].tick_params(labelsize=tick_size+7)
    axes[1].tick_params(labelsize=tick_size+7)

    axes[0].grid(linestyle='--', linewidth=0.5)
    axes[1].grid(linestyle='--', linewidth=0.5)

    #axes[0].legend(fontsize=legend_fontsize)
    plt.tight_layout()
    

    if save:
        plt.savefig(os.path.join(result_dir, f"obs-plane-plot.pdf"))
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    root = "/home/pierre/workspace/uuv_ws/src/mppi_ros/log/ocean_experiments/runs/"
    
    
    # import seaborn as sns
    # sns.set()
    # result_dir = os.path.join(root, "..", "results_seaborn")

    result_dir = os.path.join(root, "..", "results")


    ######################
    #   MPPI EXPERIMENT  #
    ######################

    # print("*"*10, " MPPI EXPERIMENTS ", "*"*10)
    # experiments = ["horizon", "filter", "samples", "noise"]
    # legend_entries = [["horizon"], ["filter_seq"], ["samples"], ["noise"]]
    # key_filtering = [None, {"horizon": (25, 25)}, None, {"filter_seq": True}]

    # action = "forward"
    # mode = "buoy"
    # model_name = 'rexrov2'
    # for e, le, kf in zip(experiments, legend_entries, key_filtering):
    #     print(f"Action: {action} | Mode: {mode} | Experiment {e}")
    #     generate_results_mppi(
    #         root=os.path.join(root, e), model_name=model_name,
    #         threshold=0.1, action=action, legend_entries=le, key_filtering=kf,
    #         gifs=False, save=True, show=False, result_dir=os.path.join(result_dir, "abletion")
    #     )


    ######################
    # COMPARISON RESULTS #
    ######################

    # print("*"*10, " COMPARISON EXPERIMENTS ", "*"*10)
    # experiments = "comparaison"
    # model_name = '/rexrov2'
    # modes = ["buoy", "pos", "neg"]
    # actions = ["up", "down", "forward", "back"]
    # actions = ["forward"]
    # controller_list = ["cascade", "pid", "mppi"]

    # for action in actions:
    #     for mode in modes:
    #         print(f"Action: {action} | Mode: {mode}")
    #         generate_restuls_controllers(
    #             root=os.path.join(root, experiments), model_name=model_name, threshold=0.1,
    #             mode=mode, action=action, controller_list=controller_list,
    #             save=True, show=False, result_dir=os.path.join(result_dir, "comparison")
    #         )


    ###################
    # ELLIPSE RESULTS #
    ###################
    
    # print("*"*10, " ELLIPSE EXPERIMENTS ", "*"*10)
    # experiment = "elipse"
    # model_name = '/rexrov2'
    # modes = ["buoy", "pos", "neg"]
    # tg = [True, False]
    # for t in tg:
    #     print(f"Ellispe cost | Tangent: {tg}")
    #     generate_elipse_results(
    #         root=os.path.join(root, experiment), model_name=model_name,
    #         threshold=0.1, modes=modes, tg=t,
    #         save=True, show=False, result_dir=os.path.join(result_dir, "ellipse")
    #     )


    ####################
    # OBSTACLE RESULTS #
    ####################

    experiment = "obstacle"
    model_name = "/rexrov2"
    modes = ["buoy", "pos", "neg"]
    print(f"Obstacle cost")
    generate_obs_results(
        root=os.path.join(root, experiment),  model_name=model_name,
        threshold=0.1, modes=modes, save=True, result_dir=os.path.join(result_dir, "obstacle"),
        show=False
    )
