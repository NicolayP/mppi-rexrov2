#!/bin/bash

source /opt/ros/melodic/setup.bash
source /home/mppi_ws/devel/setup.bash
export TF_CPP_MIN_LOG_LEVEL=2
exec "$@"
#conda run --no-capture-output -n mppi roslaunch mppi_ros mppi_rexrov_collision.launch