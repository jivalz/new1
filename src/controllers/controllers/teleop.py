import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy


class Teleop(Node):
    def __init__(self):
        super().__init__('teleop')
        self.sub = self.create_subscription(Joy, 'joy', self.joy_callback, 10)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)

    def joy_callback(self, msg):
        cmd = Twist()
        cmd.linear.x = msg.axes[1] * 3.5

        cmd.angular.z = msg.axes[3] * 1.5
        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = Teleop()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
