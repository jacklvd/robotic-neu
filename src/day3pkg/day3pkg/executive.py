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
  q                   quit
"""


class ExecutiveNode(Node):
    def __init__(self):
        super().__init__('executive')
        # Client for the action named 'drive', of type Drive
        self._client = ActionClient(self, Drive, 'drive')

    def send(self, distance=0.0, angle=0.0):
        """Send one request and WAIT until it is finished."""
        if not self._client.wait_for_server(timeout_sec=5.0):
            print('  Driving Node is not running (no action server found).')
            return False

        goal = Drive.Goal()
        goal.distance = float(distance)
        goal.angle = float(angle)

        # Wait 1: was the request accepted?
        send_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if not goal_handle.accepted:
            print(f'  REJECTED (distance={distance}, angle={angle})')
            return False

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