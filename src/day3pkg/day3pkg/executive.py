import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus

from day3_interfaces.action import Drive

HELP = """
Commands:
  m <meters>          move forward, e.g.  m 0.5
  t <degrees>         turn,         e.g.  t 90
  b <meters> <deg>    send BOTH values (should be rejected, for testing)
  a <l|r> <radius> <degrees>
                      drive an arc, e.g.  a l 0.5 180  (left, 0.5 m radius, half circle)
  8                   drive a figure-8        (extra credit)
  p                   drive a letter P        (extra credit)
  q                   quit
"""


class ExecutiveNode(Node):
    def __init__(self):
        super().__init__('executive')
        # Client for the action named 'drive', of type Drive
        self._client = ActionClient(self, Drive, 'drive')

    def send(self, distance=0.0, angle=0.0, shape='',
             arc_left=True, radius=0.0, arc_degrees=0.0):
        """Send one request and WAIT until it is finished."""
        if not self._client.wait_for_server(timeout_sec=5.0):
            print('  Driving Node is not running (no action server found).')
            return False

        goal = Drive.Goal()
        goal.distance = float(distance)
        goal.angle = float(angle)
        goal.shape = str(shape)
        goal.arc_left = bool(arc_left)
        goal.radius = float(radius)
        goal.arc_degrees = float(arc_degrees)

        # Wait 1: was the request accepted?
        send_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if not goal_handle.accepted:
            print(f'  REJECTED (distance={distance}, angle={angle}, shape={shape!r}, '
                  f'radius={radius}, arc_degrees={arc_degrees})')
            return False

        if shape:
            print(f'  Accepted, driving a {shape}...')
        elif arc_degrees:
            side = 'left' if arc_left else 'right'
            print(f'  Accepted, driving an arc ({side}, radius={radius} m, {arc_degrees} deg)...')
        else:
            print(f'  Accepted, driving (distance={distance} m, angle={angle} deg)...')

        # Wait 2: the result, when the robot has finished
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        status = result_future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            print('  Done!')
            return True
        if status == GoalStatus.STATUS_CANCELED:
            print('  Canceled.')
        else:
            print(f'  Finished with status {status}.')
        return False


def main(args=None):
    rclpy.init(args=args)
    node = ExecutiveNode()
    print(HELP)

    while True:
        try:
            text = input('Enter command> ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            text = 'q'

        if not text:
            continue
        parts = text.split()
        cmd = parts[0]

        try:
            if cmd == 'q':
                break
            elif cmd == 'm' and len(parts) == 2:
                node.send(distance=float(parts[1]))
            elif cmd == 't' and len(parts) == 2:
                node.send(angle=float(parts[1]))
            elif cmd == 'a' and len(parts) == 4 and parts[1] in ('l', 'r'):
                node.send(arc_left=(parts[1] == 'l'),
                          radius=float(parts[2]), arc_degrees=float(parts[3]))
            elif cmd == '8' and len(parts) == 1:
                node.send(shape='figure8')
            elif cmd == 'p' and len(parts) == 1:
                node.send(shape='p')
            elif cmd == 'b' and len(parts) == 3:
                node.send(distance=float(parts[1]), angle=float(parts[2]))
            else:
                print('  Unknown command.' + HELP)
        except ValueError:
            print('  Please enter a number, e.g.  m 0.5')

    # Quit: destroy the node and shut down rclpy
    print('Quitting.')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()