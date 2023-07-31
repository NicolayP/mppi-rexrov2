#!/usr/bin/env python
import os
import glob
import rospy
import shutil

from std_msgs.msg import String

# We assume a active rosbag file exist. Pick this as the filename to have 
# matching timestamps

def copy_cb(msg):
    if not rospy.has_param('~log'):
        rospy.logerr("Didn't provide path to logdir")

    log_dir = rospy.get_param('~log')
    os.chdir(log_dir)
    bag_file = glob.glob("*.active")[0]
    base_name = os.path.basename(bag_file)
    # remove .active
    base_name = os.path.splitext(base_name)[0]
    # remove .bag
    base_name = os.path.splitext(base_name)[0]

    if rospy.has_param('~cost_file'):
        cost_file = rospy.get_param('~cost_file')
        dst_file = base_name + "_cost.yaml"
        shutil.copyfile(cost_file, dst_file)

    if rospy.has_param("~model_file"):
        model_file = rospy.get_param('~model_file')
        dst_file = base_name + "_model.yaml"
        shutil.copyfile(model_file, dst_file)

    if rospy.has_param("~conf_file"):
        conf_file = rospy.get_param("~conf_file")
        dst_file = base_name + "_conf.yaml"
        shutil.copyfile(conf_file, dst_file)

    if rospy.has_param("~save_param"):
        save_param = rospy.get_param("~save_param")
        if save_param:
            if rospy.has_param("~param_namespace"):
                namespace = rospy.get_param("~param_namespace")
            else:
                namespace = ""
            param_file = base_name + "_params.yaml"
            os.system(f'rosparam dump {param_file} /{namespace}')

    exit()

if __name__ == "__main__":
    rospy.init_node("Copy_cost_routine")

    # waiting for rosbag record signal
    subscriber = rospy.Subscriber("begin_write", String, copy_cb)
    rospy.spin()