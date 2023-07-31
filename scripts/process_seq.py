#!/usr/bin/env python3
import rospy

import numpy as np
import os
from std_msgs.msg import Float32MultiArray, String

import matplotlib.pyplot as plt


class SeqProcess(object):
    def __init__(self):
        self.step = 0
        self.init = False
        self.recording =rospy.Subscriber("begin_write", String, self.start)
        self.seqSub = rospy.Subscriber("mppi/seq", Float32MultiArray, self.receive_cb)
        self.seq_list = []
        self.log_path = "."
        if rospy.has_param("~log_path"):
            self.log_path = rospy.get_param("~log_path")
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.process()
            r.sleep()

    def start(self, msg):
        file = msg.data
        root = os.path.splitext(file)[0]
        self.log_path = root + "_seq_imgs"
        os.mkdir(self.log_path)
        self.init = True

    def receive_cb(self, msg):
        tau = msg.layout.dim[0].size
        aDim = msg.layout.dim[1].size
        seq = np.array(msg.data).reshape((tau, aDim))
        if self.init:
            self.seq_list.append(seq)

    def process(self):
        if len(self.seq_list) == 0:
            return
        seq = self.seq_list[0]
        self.seq_list.pop(0)
        self.plot(seq, self.step)
        self.step += 1

    def plot(self, seq, i):
        fig, axes = plt.subplots(2, 3, sharex=True)
        fig.suptitle(f"Action Sequence: {i}", fontsize=20)
        fig.set_figheight(15)
        fig.set_figwidth(20)

        axes[0, 0].plot(seq[:, 0])
        axes[0, 0].set_title('Fx')
        axes[0, 0].set_ylim(-1, 1)

        axes[0, 1].plot(seq[:, 1])
        axes[0, 1].set_title('Fy')
        axes[0, 1].set_ylim(-1, 1)

        axes[0, 2].plot(seq[:, 2])
        axes[0, 2].set_title('Fz')
        axes[0, 2].set_ylim(-1, 1)
        # Angles
        axes[1, 0].plot(seq[:, 3])
        axes[1, 0].set_title('Tx')
        axes[1, 0].set_ylim(-1, 1)

        axes[1, 1].plot(seq[:, 4])
        axes[1, 1].set_title('Ty')
        axes[1, 1].set_ylim(-1, 1)

        axes[1, 2].plot(seq[:, 5])
        axes[1, 2].set_title('Tz')
        axes[1, 2].set_ylim(-1, 1)

        fig.legend()
        plt.savefig(os.path.join(self.log_path, f"{i:05}" + ".png"))
        plt.close()


if __name__ == "__main__":
    rospy.init_node("Action sequence processing")
    node = SeqProcess()
