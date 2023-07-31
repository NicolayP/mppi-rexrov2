# MPPI_ROS

This package contains the necessary ros instruction to run the mppi controller. 

In order to do so the node needs to subscribe to a "state" topic and published to a "force" topic.


## Documentation.

The controller supports multiple angle formats for 6D problems:

    - euler (roll, pitch, yaw) 'XYZ'.
    - quaternion (qx, qy, qz, qw).
    - rotation matrix.

The core components of the controller are the following:

    - Controller class. Implements the core of the algorithm. Computes the new action sequence.
    - Cost class. The cost associated with the current task.
    - Model class. The predictive model used as proxy for the plant.
    - Controller input. Class representing the state, action and other information that
        will be fed to the controller. 
    - ModelState class. Class representing the state inside the model. 