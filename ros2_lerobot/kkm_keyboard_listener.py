import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pynput import keyboard

class KeyboardListenerNode(Node):
    def __init__(self):
        super().__init__('keyboard_listener_node')
        self.publisher_ = self.create_publisher(String, '/keyboard_command', 10)
        self.get_logger().info("Keyboard listener started. \nPress SPACE (start), LEFT (reset), RIGHT (done), 1 (flatten), 2 (fold)")

        # Start key listener in background
        listener = keyboard.Listener(on_press=self.on_key_press)
        listener.daemon = True
        listener.start()

    def on_key_press(self, key):
        msg = String()
        try:
            if key == keyboard.Key.space:
                msg.data = 'start'
            elif key == keyboard.Key.left:
                msg.data = 'reset'
            elif key == keyboard.Key.right:
                msg.data = 'done'
            elif hasattr(key, 'char') and key.char == '1':
                msg.data = 'flatten'
            elif hasattr(key, 'char') and key.char == '2':
                msg.data = 'fold' 
            elif hasattr(key, 'char') and key.char == 's':
                msg.data = 'shoot'   
            else:
                return  # Ignore other keys

            self.publisher_.publish(msg)
            self.get_logger().info(f"Published: {msg.data}")
        except Exception as e:
            self.get_logger().error(f"Key press error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = KeyboardListenerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()
