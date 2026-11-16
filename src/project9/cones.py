"""Project 9: find three coloured traffic cones, then drive to the one asked for.

Task 1: the robot turns 360 degrees in STEP_DEG steps. After each step it
grabs a camera frame and looks for each colour in COLORS. For every colour it
keeps the sighting closest to the image centre, then uses the lidar near that
bearing to find how far away the cone is. Each cone's position is stored in
the odometry frame.

Task 2: in a loop, you type a colour. The robot turns towards that cone,
centres it in the camera, and drives up with the lidar. It stops a little way
short, re-centres, then creeps up to stop STOP_GAP from the cone's base.

Usage:
  python3 cones.py                 Task 1 then Task 2
  python3 cones.py look            save a camera frame and colour masks (for tuning COLORS)
  --image-topic TOPIC              camera topic (default /oakd/rgb/preview/image_raw)

If no /scan message arrives, start the lidar motor:
  ros2 service call /start_motor std_srvs/srv/Empty
"""
import math
import os
import sys
import time

import cv2
import numpy as np

# The three cone colours as HSV ranges (OpenCV: H 0-179, S and V 0-255).
# Change these to match your cones; tune them with `python3 cones.py look`.
# A colour can have several ranges, which red needs because its hue wraps round 0.
COLORS = {
    'orange': [((5, 120, 120), (20, 255, 255))],
    'green': [((40, 80, 60), (85, 255, 255))],
    'blue': [((95, 120, 60), (130, 255, 255))],
}
MIN_AREA = 150  # pixels; smaller blobs are ignored as noise
IMAGE_TOPIC = '/oakd/rgb/preview/image_raw'
# dev-note: the OAK-D's field of view, which depends on how the preview is cropped.
# If centring overshoots, lower it; if it creeps up in many steps, raise it.
HFOV_DEG = 69.0
STEP_DEG = 30.0  # Task 1 turn per step; keep it below HFOV_DEG / 2 so no cone is missed
TURN_SPEED = 0.5  # rad/s
DRIVE_SPEED = 0.15  # m/s on the way to a cone
CREEP_SPEED = 0.05  # m/s for the last few centimetres
SETTLE_SEC = 0.5  # let the robot and the camera image settle after a move
ALIGN_TOL_DEG = 2.0  # a cone this close to the image centre counts as centred
LIDAR_WINDOW_DEG = 8.0  # look for the cone this far either side of its bearing
# base_link -> rplidar_link on our Turtlebot 4: the lidar is 4 cm behind the
# centre and turned 90 degrees to the left (as in Projects 6 and 7)
LIDAR_IN_BASE = (-0.04, 0.0, math.pi / 2)
ROBOT_RADIUS = 0.17  # m, robot centre to the front of the bumper
# dev-note: the lidar hits the cone higher up, where it is narrower than its base.
# This is how much further out the base reaches. Measure it on your cones.
CONE_BASE_OVER_LIDAR = 0.06
STOP_GAP = 0.025  # m, about an inch between the bumper and the cone's base
APPROACH_MARGIN = 0.15  # m, stop this far short first, then re-centre and creep
HERE = os.path.dirname(os.path.abspath(__file__))


def wrap(a):
    """Wrap an angle in radians to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def find_cone(bgr, color):
    """Return (centre x as a fraction of the image width, area) of the largest blob of a colour, or None."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = color_mask(hsv, color)
    # largest connected blob of that colour; label 0 is the background
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[k, cv2.CC_STAT_AREA])
    if area < MIN_AREA:
        return None
    return centroids[k][0] / bgr.shape[1], area


def color_mask(hsv, color):
    """Binary mask of the pixels inside any of a colour's HSV ranges, with specks removed."""
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in COLORS[color]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    # an opening (erode then dilate) removes single-pixel noise
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def pixel_bearing(cx):
    """Angle (radians, positive = left) from the camera axis to an image column given as a fraction."""
    # right of centre is a turn to the right, i.e. a negative angle
    return math.radians((0.5 - cx) * HFOV_DEG)


def nearest_return(scan, bearing, window=math.radians(LIDAR_WINDOW_DEG), lidar=LIDAR_IN_BASE):
    """Closest lidar hit within +-window of a bearing in the robot's frame, as (x, y) metres in base_link, or None."""
    lx, ly, lyaw = lidar
    c, s = math.cos(lyaw), math.sin(lyaw)
    best = None
    for k, r in enumerate(scan.ranges):
        if not (scan.range_min <= r <= scan.range_max):
            continue
        a = scan.angle_min + k * scan.angle_increment
        # the hit point moved from the lidar's frame into base_link
        x = lx + c * r * math.cos(a) - s * r * math.sin(a)
        y = ly + s * r * math.cos(a) + c * r * math.sin(a)
        if abs(wrap(math.atan2(y, x) - bearing)) <= window and (
                best is None or math.hypot(x, y) < math.hypot(*best)):
            best = (x, y)
    return best


def approach_distance(cone_xy):
    """How far to drive straight ahead so the bumper stops STOP_GAP from a cone at cone_xy (base_link)."""
    # centre-to-cone distance, less the robot's size, the base's overhang and the gap
    return math.hypot(*cone_xy) - ROBOT_RADIUS - CONE_BASE_OVER_LIDAR - STOP_GAP


def to_odom(pose, xy):
    """Move a point from base_link into the odom frame, given the robot's odom pose (x, y, yaw)."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * xy[0] - s * xy[1], y + s * xy[0] + c * xy[1]


def image_to_bgr(msg):
    """Convert a sensor_msgs/Image (bgr8, rgb8 or bgra8) to an OpenCV BGR array, without needing cv_bridge."""
    ch = 4 if msg.encoding in ('bgra8', 'rgba8') else 3
    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * ch]
    img = img.reshape(msg.height, msg.width, ch)[:, :, :3]
    return img[:, :, ::-1].copy() if msg.encoding.startswith('rgb') else img.copy()


def run(mode, image_topic):
    """Connect to the robot and run Task 1 and Task 2, or the `look` tuning mode."""
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from action_msgs.msg import GoalStatus
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image, LaserScan
    from irobot_create_msgs.action import DriveDistance, RotateAngle

    class ConeNode(Node):
        """Keeps the latest scan, odometry and camera frame, and wraps the Create 3 actions."""

        def __init__(self):
            super().__init__('cones')
            self.scan = self.odom = self.image = None
            self.create_subscription(LaserScan, '/scan', lambda m: setattr(self, 'scan', m),
                                     qos_profile_sensor_data)
            self.create_subscription(Odometry, '/odom', lambda m: setattr(self, 'odom', m),
                                     qos_profile_sensor_data)
            self.create_subscription(Image, image_topic, lambda m: setattr(self, 'image', m),
                                     qos_profile_sensor_data)
            self.drive_client = ActionClient(self, DriveDistance, 'drive_distance')
            self.rotate_client = ActionClient(self, RotateAngle, 'rotate_angle')

        def wait_for(self, attr, timeout=10.0):
            """Throw away the stored message and wait for a new one on that topic."""
            setattr(self, attr, None)
            end = time.time() + timeout
            while getattr(self, attr) is None:
                if time.time() > end:
                    sys.exit(f'No {attr} message in {timeout:.0f} s. Check the topic '
                             '(ros2 topic list); for /scan: ros2 service call /start_motor std_srvs/srv/Empty')
                rclpy.spin_once(self, timeout_sec=0.05)
            return getattr(self, attr)

        def frame(self):
            """A camera frame taken after the robot stopped, as a BGR array."""
            # dev-note: skips one frame in case it was captured while the robot was still moving
            self.wait_for('image')
            return image_to_bgr(self.wait_for('image'))

        def pose(self):
            """The robot's current odom pose (x, y, yaw)."""
            o = self.wait_for('odom').pose.pose
            q = o.orientation
            return (o.position.x, o.position.y,
                    math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))

        def act(self, client, goal, what):
            """Send an action goal, wait for it to finish, then let the robot settle."""
            if not client.wait_for_server(timeout_sec=5.0):
                sys.exit(f'No /{what} action server.')
            send = client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send)
            if not send.result().accepted:
                sys.exit(f'{what} goal was rejected.')
            result = send.result().get_result_async()
            rclpy.spin_until_future_complete(self, result)
            if result.result().status != GoalStatus.STATUS_SUCCEEDED:
                print(f'  WARNING: {what} did not succeed (status {result.result().status}); '
                      'was a bumper or cliff sensor triggered?')
            time.sleep(SETTLE_SEC)

        def turn(self, angle):
            """Turn in place by angle radians (positive = left)."""
            goal = RotateAngle.Goal()
            goal.angle = float(angle)
            goal.max_rotation_speed = TURN_SPEED
            self.act(self.rotate_client, goal, 'rotate_angle')

        def forward(self, d, speed):
            """Drive d metres straight ahead."""
            goal = DriveDistance.Goal()
            goal.distance = float(d)
            goal.max_translation_speed = speed
            self.act(self.drive_client, goal, 'drive_distance')

        def centre_on(self, color, tries=5):
            """Turn until the cone of a colour is in the middle of the image; False if it is not in view."""
            for _ in range(tries):
                seen = find_cone(self.frame(), color)
                if seen is None:
                    return False
                offset = pixel_bearing(seen[0])
                if abs(offset) < math.radians(ALIGN_TOL_DEG):
                    return True
                self.turn(offset)
            return True

    def look():
        """Save one camera frame and a mask per colour so the HSV ranges can be tuned."""
        bgr = node.frame()
        cv2.imwrite(os.path.join(HERE, 'look.png'), bgr)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        print(f'image {w}x{h}, HSV at the centre = {tuple(int(v) for v in hsv[h // 2, w // 2])}')
        for color in COLORS:
            cv2.imwrite(os.path.join(HERE, f'look_{color}.png'), color_mask(hsv, color))
            print(f'  {color}: {find_cone(bgr, color)}')
        print(f'saved look.png and look_<color>.png in {HERE}')

    def task1():
        """Turn 360 degrees in steps, find each colour's cone and store its odom position."""
        best = {}  # colour -> (|offset|, odom yaw of the cone's bearing)
        for step in range(round(360 / STEP_DEG)):
            bgr = node.frame()
            _, _, yaw = node.pose()
            seen_now = []
            for color in COLORS:
                seen = find_cone(bgr, color)
                if seen is None:
                    continue
                seen_now.append(color)
                offset = pixel_bearing(seen[0])
                # keep the sighting nearest the image centre, where the bearing is most accurate
                if color not in best or abs(offset) < best[color][0]:
                    best[color] = (abs(offset), wrap(yaw + offset))
            print(f'  step {step + 1}: heading {math.degrees(yaw):+.0f} deg, '
                  f'seen {", ".join(seen_now) or "nothing"}')
            node.turn(math.radians(STEP_DEG))

        cones = {}
        pose = node.pose()
        scan = node.wait_for('scan')
        print('\nCones found:')
        for color in COLORS:
            if color not in best:
                print(f'  {color:8} NOT FOUND')
                continue
            # the lidar gives the distance to the cone that the camera found
            hit = nearest_return(scan, wrap(best[color][1] - pose[2]))
            if hit is None:
                print(f'  {color:8} seen by the camera but no lidar return near it')
                continue
            cones[color] = to_odom(pose, hit)
            print(f'  {color:8} bearing {math.degrees(math.atan2(hit[1], hit[0])):+6.1f} deg, '
                  f'{math.hypot(*hit):.2f} m from the robot centre')
        return cones

    def go_to(color, cones):
        """Turn towards a stored cone, centre it in the camera and drive up to STOP_GAP from it."""
        x, y, yaw = node.pose()
        cx, cy = cones[color]
        node.turn(wrap(math.atan2(cy - y, cx - x) - yaw))
        # two passes: stop APPROACH_MARGIN short, then re-centre and creep up slowly
        for margin, speed in ((APPROACH_MARGIN, DRIVE_SPEED), (0.0, CREEP_SPEED)):
            if not node.centre_on(color):
                print(f'  The {color} cone is not in view; is it still there?')
                return
            hit = nearest_return(node.wait_for('scan'), 0.0)
            if hit is None:
                print('  No lidar return straight ahead.')
                return
            cones[color] = to_odom(node.pose(), hit)  # refresh the stored position
            d = approach_distance(hit) - margin
            print(f'  cone {math.hypot(*hit):.3f} m away, driving {max(d, 0):.3f} m')
            if d > 0.005:
                node.forward(d, speed)
        print(f'  Arrived at the {color} cone.')

    rclpy.init()
    node = ConeNode()
    try:
        if mode == 'look':
            look()
            return
        print('Task 1: turning 360 degrees to find the cones')
        cones = task1()
        if not cones:
            sys.exit('No cones found. Tune COLORS with: python3 cones.py look')
        print('\nTask 2')
        while True:
            try:
                color = input(f'Cone colour ({", ".join(cones)}) or q to quit> ').strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            if color == 'q':
                break
            if color not in cones:
                print(f'  Unknown or unfound colour {color!r}.')
                continue
            go_to(color, cones)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main():
    """Parse the command line and run."""
    args = sys.argv[1:]
    topic = IMAGE_TOPIC
    if '--image-topic' in args:
        i = args.index('--image-topic')
        if i + 1 >= len(args):
            sys.exit('--image-topic needs a topic name.')
        topic = args[i + 1]
        del args[i:i + 2]
    mode = args[0] if args else 'run'
    if mode not in ('run', 'look'):
        sys.exit(__doc__)
    run(mode, topic)


if __name__ == '__main__':
    main()
