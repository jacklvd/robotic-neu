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

<details>
<summary><b>Project 7 — mapping out and back</b></summary>

`src/project7/` builds on the Project 6 grid, made bigger: 320 x 320 cells of
5 cm (16 m x 16 m), so a classroom's walls stay on the grid after a 3 m
drive. It needs `numpy` and `opencv-python` (`cv2`).

```
source install/setup.bash
cd src/project7
```

#### Task 1 — a grid each way

```
python3 map_run.py drive            # out 3 m, turn 180 degrees, back 3 m
python3 map_run.py replay [folder]  # rebuild the grids of a saved run (default: newest)
```

The robot drives 3 m at 0.1 m/s with `DriveDistance`, updating a grid every
0.1 m, turns around with `RotateAngle`, then drives 3 m back, building a second
grid the same way. Each grid's frame is the lidar's position at the start of
its leg, with x along the robot's heading then, so the return grid is turned
round relative to the outbound one.

Every scan and odometry pose is saved, so a run can be replayed without the
robot. Each run gets its own folder, `runs/<date-time>/`, holding
`leg_out.json`, `leg_back.json`, `grid_out.png` and `grid_back.png`. Add
`--show` to open the grids in a window.

#### Task 2 — landmarks

```
python3 task2.py [folder]   # default: the newest run in runs/
```

Finds landmarks in both grids of a run: small (at most 15 cm), strongly
filled clusters with no other filled cell within 15 cm, which in an empty
classroom are mostly table and chair legs. Each landmark's position is the
centre of its cluster, weighted by log-odds. RANSAC then matches the two lists
without using odometry: every pair of landmarks in one grid is tried against
every pair at the same distance apart in the other, and the rotation and shift
that line up the most landmarks (within 10 cm) wins, refined by least squares.

It prints, and saves to `task2_report.txt` in the run folder, the transform
from the return grid to the outbound grid (next to odometry's), every return
landmark in the outbound frame, the landmarks found in only one grid, and the
mean position difference of those found in both. It also saves
`landmarks_out.png` and `landmarks_back.png` (landmarks circled: green = in
both grids, with the same number in each; orange / blue = only in that grid)
and `overlay.png` (the return grid moved onto the outbound one).

#### Task 3 — A* path

```
python3 task3.py [folder] [--goal X Y]   # default goal: (3.0, 1.75) m
```

Plans the shortest path with A* on the outbound grid, from where the robot's
centre was at the start to the goal (metres, outbound grid frame; it must be at
least 2 m away). Filled cells are grown by 20 cm (the Turtlebot 4's 17 cm
radius plus a 3 cm margin), so keeping the robot's centre out of them keeps the
whole robot clear, and floor never seen as empty is off limits. A* moves to the
8 neighbouring cells without cutting obstacle corners, with the straight-line
distance to the goal as its heuristic. The default goal is behind the box
ahead and to the left in our run, so the straight line is blocked and the path
bends round it. Saves `task3_path.png` (grown obstacles, the path and the
blocked straight line) and `task3_path.txt` (the path's points) in the run
folder.

Tests (no robot needed):

```
python3 -m pytest test_project7.py
```

</details>

<details>
<summary><b>Project 9 — finding and driving to traffic cones</b></summary>

`src/project9/cones.py` uses the OAK-D camera to tell the cones apart by
colour and the lidar to measure how far away they are. It needs `numpy` and
`opencv-python` (`cv2`); `cv_bridge` is not needed.

Set three cones of different colours in a triangle, at least 1.5 m apart,
with nothing inside it, and put the robot near the middle.

```
source install/setup.bash
cd src/project9
```

#### Tune the colours first

The three colours are hard-coded in `COLORS` at the top of `cones.py`
(default: `orange`, `green`, `blue`, as HSV ranges). Point the robot at each
cone and run:

```
python3 cones.py look
```

This prints the HSV value at the centre of the image and what each colour
detects. It also saves `look.png` and one mask per colour, `look_<color>.png`
(white = matched). Widen or narrow a range until only its cone shows up white.

#### Task 1 and Task 2

```
python3 cones.py
python3 cones.py --image-topic /oakd/rgb/image_raw   # if your camera topic differs (ros2 topic list)
```

**Task 1:** the robot turns 360 degrees in 30 degree steps with `RotateAngle`.
At each stop it looks for every colour and keeps the sighting nearest the
image centre. The lidar then gives each cone's distance. It prints each
colour's bearing and distance, or `NOT FOUND`.

**Task 2:** at the prompt, type a colour (`q` quits). The robot turns towards
that cone, centres it in the camera and drives with `DriveDistance` to 15 cm
short. It then re-centres and creeps up, stopping about an inch (`STOP_GAP`)
from the cone's base. You can type another colour and it drives on to that cone.

Calibration constants at the top of `cones.py`:

- `HFOV_DEG`: the camera's field of view. Lower it if centring overshoots;
  raise it if centring takes many small turns.
- `CONE_BASE_OVER_LIDAR`: how much further out the cone's base reaches than
  the point where the lidar hits it. Raise it if the robot touches the cone;
  lower it if the robot stops too far away.
- `STOP_GAP`: the gap to leave between the bumper and the cone.

If no `/scan` message arrives, start the lidar motor:

```
ros2 service call /start_motor std_srvs/srv/Empty
```

Tests (no robot needed):

```
python3 -m pytest test_project9.py
```

</details>
