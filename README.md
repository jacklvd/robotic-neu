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

## day2pkg — driving over /cmd_vel

```
ros2 run day2pkg node1 --ros-args -p test:=D
```

Tests: `D`, `straight`, `spin`, `halfcircle`, `circle`.
`-p distance:=0.5` sets the distance for the `straight` test.

## day3pkg — driving a given distance through an Action

The `Drive` action (`day3_interfaces/action/Drive`) carries a distance in
metres and an angle in degrees. One of the two must be 0; a request with both
non-zero, or with either one negative, is rejected. The feedback and the
response are both empty for now.

Two nodes talk over it:

- **Executive Node** (`executive`) — the action client and the user interface.
- **Driving Node** (`driving`) — the action server, which publishes the
  `cmd_vel` messages. A move uses only `linear.x`, a turn only `angular.z`,
  and either one ends with a stop message.

### Run both with the launch file (Task 4)

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

### Executive Node commands

```
m <meters>          move forward, e.g.  m 0.5
t <degrees>         turn,         e.g.  t 90
b <meters> <deg>    send BOTH values (should be rejected, for testing)
8                   drive a figure-8        (extra credit A)
p                   drive a letter P        (extra credit A)
q                   quit
```

### Cancelling a command (extra credit B)

With both nodes running, start a long move (`m 2`) and then, in another
terminal:

```
./src/day3pkg/cancel_goal.sh
```

The robot stops, the Driving Node reports the goal as canceled, and the
Executive Node prints `Canceled.` and returns to its prompt.
