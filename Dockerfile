FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu18.04

RUN apt update && apt install -y --no-install-recommends dirmngr gnupg2 curl software-properties-common && rm -rf /var/lib/apt/lists/*
# setup keys

RUN add-apt-repository universe && add-apt-repository multiverse && apt update
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install ros-melodic-desktop-full -y
RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc

# bootstrap rosdep
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends -y python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential python-catkin-tools python3-vcstool && rm -rf /var/lib/apt/lists/*
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install python-rosdep

RUN rosdep init
RUN rosdep update

RUN apt update && \
    apt install --no-install-recommends -y git openssh-client build-essential wget && \
    apt clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install conda

# Install miniconda
ENV CONDA_DIR /opt/conda

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH


RUN mkdir -p -m 0770 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

# Create catkin workspace
RUN echo ". /opt/ros/melodic/setup.bash" >> ~/.bashrc && \
    . /opt/ros/melodic/setup.sh && \
    mkdir -p  /home/mppi_ws/src/

USER root
WORKDIR /home/mppi_ws/src/
RUN --mount=type=ssh git clone git@github.com:NicolayP/mppi-ros.git && \
    cd mppi-ros && git checkout cleanup
WORKDIR /home/mppi_ws/src/mppi-ros/scripts
RUN --mount=type=ssh git submodule update --init --recursive && \
    cd mppi_tf && git checkout cleanup

RUN . /opt/ros/melodic/setup.sh && \
    cd /home/mppi_ws/ && \
    catkin_init_workspace && catkin build -j4 && \
    echo ". /home/mppi_ws/devel/setup.bash" >> ~/bashrc

COPY environment.yaml .
COPY requirements.txt .
RUN conda env create -f environment.yaml

SHELL ["conda", "run", "-n", "mppi", "/bin/bash", "-c"]

#RUN apt clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
#ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mppi", "roslaunch", "mppi_ros", "mppi_rexrov_collision.launch"]
# setup entrypoint
COPY ./ros_entrypoint.sh /

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]