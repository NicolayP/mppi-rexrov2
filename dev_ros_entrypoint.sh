#!/bin/bash

source /opt/ros/melodic/setup.bash && \
cd /home/mppi_ws/ && \
catkin_init_workspace && catkin build -j4 && \
source /home/mppi_ws/devel/setup.bash

export TF_CPP_MIN_LOG_LEVEL=2

exec "$@"