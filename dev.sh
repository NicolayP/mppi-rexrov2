#!/bin/bash

docker run --runtime=nvidia --mount "type=bind,src=$(pwd)/scripts,dst=/home/mppi_ws/src/mppi_ros/scripts" -t mppi-dock