#!/bin/bash
# Extra Credit B: cancel whatever the Driving Node is currently doing.
# An empty CancelGoal request cancels all goals currently being executed.
ros2 service call /drive/_action/cancel_goal action_msgs/srv/CancelGoal "{}"
