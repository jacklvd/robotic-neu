# robotic-neu

ROS 2 (Jazzy) workspace.

## Build

```
cd ~/robotic-neu
colcon build
source install/setup.bash
```

`day3pkg` depends on the generated action in `day3_interfaces`, so build
`day3_interfaces` first if you build packages one at a time:

```
colcon build --packages-select day3_interfaces day3pkg
source install/setup.bash
```

## Projects

<details>
<summary><b>day2pkg — driving over /cmd_vel</b></summary>

```
ros2 run day2pkg node1 --ros-args -p test:=D
```

Tests: `D`, `straight`, `spin`, `halfcircle`, `circle`.
`-p distance:=0.5` sets the distance for the `straight` test.

</details>

<details>
<summary><b>day3pkg — driving a given distance through an Action</b></summary>

The `Drive` action (`day3_interfaces/action/Drive`) carries a distance in
metres and an angle in degrees. One of the two must be 0; a request with both
non-zero, or with either one negative, is rejected. The feedback and the
response are both empty for now.

Two nodes talk over it:

- **Executive Node** (`executive`) — the action client and the user interface.
- **Driving Node** (`driving`) — the action server, which publishes the
  `cmd_vel` messages. A move uses only `linear.x`, a turn only `angular.z`,
  and either one ends with a stop message.

#### Run both with the launch file (Task 4)

```
ros2 launch day3pkg drive_launch.py
```

`ros2 launch` does not hand its nodes a terminal to read from, so the
Executive Node gets its own window. The launch file picks whatever terminal is
installed (xterm, terminator, gnome-terminal or konsole); to choose one
yourself, or to skip the window and start the Executive Node by hand:

```
ros2 launch day3pkg drive_launch.py terminal:=xterm
ros2 launch day3pkg drive_launch.py terminal:=none   # then: ros2 run day3pkg executive
```

#### Executive Node commands

```
m <meters>          move forward, e.g.  m 0.5
t <degrees>         turn,         e.g.  t 90
b <meters> <deg>    send BOTH values (should be rejected, for testing)
8                   drive a figure-8        (extra credit A)
p                   drive a letter P        (extra credit A)
q                   quit
```

#### Cancelling a command (extra credit B)

With both nodes running, start a long move (`m 2`) and then, in another
terminal:

```
./src/day3pkg/cancel_goal.sh
```

The robot stops, the Driving Node reports the goal as canceled, and the
Executive Node prints `Canceled.` and returns to its prompt.

</details>

<details>
<summary><b>Project 5 — odometry accuracy</b></summary>

The scripts in `src/project5` are plain Python, not a colcon package. Source
the workspace first so `day3_interfaces` can be imported, and start the
Driving Node for Tasks 2 and 3 (Task 4 uses the robot's own
`drive_distance` action and does not need it):

```
source install/setup.bash
ros2 run day3pkg driving          # in its own terminal, for Tasks 2 and 3
cd src/project5
```

Each run of `record_run.py` reads `/odom`, drives, waits for the robot to
settle, reads `/odom` again and prints both poses, the odometry distance and
the heading change. Nothing is written to disk; copy the numbers, together
with the ruler measurement, into the task's CSV file.

```
python3 record_run.py           # Task 2: 1 m forward with our Drive action
python3 record_run.py circle    # Task 3: left circle, 0.5 m radius
python3 record_run.py dd        # Task 4: 1 m with the built-in DriveDistance
```

Error table, mean and standard deviation (actual vs. odometry) for the 1 m
runs:

```
python3 stats.py task2.csv
python3 stats.py task4.csv
```

</details>

<details>
<summary><b>Project 6 — lidar occupancy grid</b></summary>

`src/project6/occupancy_grid.py` builds a 200 x 200 grid of 5 cm cells
(10 m x 10 m) from `/scan`, using log-odds and Bresenham ray casting. It needs
`numpy`, `opencv-python` (`cv2`) and, for a saved scan, `pyyaml`. Grid images
are saved as PNG files next to the script; add `--show` to also open each one
in a window (press any key to close it).

```
source install/setup.bash
cd src/project6
```

#### Task 1 — one scan

```
python3 occupancy_grid.py file scan_sample.txt   # offline, from a saved scan -> grid_from_file.png
python3 occupancy_grid.py scan                   # one live scan             -> grid_scan.png
```

To save a new scan (without `--full-length`, `ros2` cuts the ranges list off
after 128 values):

```
ros2 topic echo --once --full-length /scan > scan_sample.txt
```

#### Task 2 — update while driving

```
python3 occupancy_grid.py drive
```

The robot drives 1 m at 0.1 m/s with `DriveDistance` and the grid is updated
every 0.1 m, saving `grid_0.0m.png`, `grid_0.5m.png` and `grid_1.0m.png`.

If no `/scan` message arrives, start the lidar motor:

```
ros2 service call /start_motor std_srvs/srv/Empty
```

</details>
