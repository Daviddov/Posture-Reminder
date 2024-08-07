import cv2
import numpy as np
import mediapipe as mp
import threading
import time

class HandFootballGame:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.ball_position = np.array([self.frame_width // 2, self.frame_height // 2], dtype=np.float32)
        self.ball_velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.ball_radius = 20
        self.stop_event = threading.Event()
        self.score_left = 0
        self.score_right = 0
        self.start_time = time.time()
        self.last_touch_time = 0

        # Define goal posts
        goal_width = 20
        goal_height = self.frame_height // 3
        self.left_goal = {
            'x': 0,
            'y': (self.frame_height - goal_height) // 2,
            'width': goal_width,
            'height': goal_height
        }
        self.right_goal = {
            'x': self.frame_width - goal_width,
            'y': (self.frame_height - goal_height) // 2,
            'width': goal_width,
            'height': goal_height
        }

    def detect_hands(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb_frame).multi_hand_landmarks

    def check_ball_touch(self, hand_landmarks):
        for handLms in hand_landmarks:
            index_finger_tip = handLms.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            finger_position = np.array([index_finger_tip.x * self.frame_width, index_finger_tip.y * self.frame_height])
            if np.linalg.norm(finger_position - self.ball_position) < self.ball_radius:
                return finger_position
        return None

    def update_ball_position(self):
        self.ball_position += self.ball_velocity

        # Apply friction
        self.ball_velocity *= 0.98

        # Bounce off walls
        if self.ball_position[0] <= self.ball_radius or self.ball_position[0] >= self.frame_width - self.ball_radius:
            self.ball_velocity[0] *= -0.8
        if self.ball_position[1] <= self.ball_radius or self.ball_position[1] >= self.frame_height - self.ball_radius:
            self.ball_velocity[1] *= -0.8

        # Ensure ball stays within frame
        self.ball_position = np.clip(self.ball_position, [self.ball_radius, self.ball_radius],
                                     [self.frame_width - self.ball_radius, self.frame_height - self.ball_radius])

    def check_goal(self):
        if (self.left_goal['x'] <= self.ball_position[0] <= self.left_goal['x'] + self.left_goal['width'] and
                self.left_goal['y'] <= self.ball_position[1] <= self.left_goal['y'] + self.left_goal['height']):
            self.score_right += 1
            return "Right team scored!"
        elif (self.right_goal['x'] <= self.ball_position[0] <= self.right_goal['x'] + self.right_goal['width'] and
              self.right_goal['y'] <= self.ball_position[1] <= self.right_goal['y'] + self.right_goal['height']):
            self.score_left += 1
            return "Left team scored!"
        return None

    def draw_game_info(self, frame):
        cv2.putText(frame, f"Left: {self.score_left} | Right: {self.score_right}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {int(time.time() - self.start_time)}s", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    def handle_ball_touch(self, frame, touch_position):
        current_time = time.time()
        if current_time - self.last_touch_time > 0.1:  # Prevent too rapid touches
            self.last_touch_time = current_time

            # Change ball direction and add velocity based on touch position
            touch_vector = self.ball_position - touch_position  # Reversed to move away from touch
            touch_vector /= np.linalg.norm(touch_vector)  # Normalize the vector
            self.ball_velocity += touch_vector * 10  # Adjust multiplier for desired speed

            # Limit maximum velocity
            max_speed = 20
            current_speed = np.linalg.norm(self.ball_velocity)
            if current_speed > max_speed:
                self.ball_velocity = (self.ball_velocity / current_speed) * max_speed

            cv2.circle(frame, tuple(touch_position.astype(int)), 10, (0, 255, 0), -1)

    def draw_field(self, frame):
        # Draw center line
        cv2.line(frame, (self.frame_width // 2, 0), (self.frame_width // 2, self.frame_height), (255, 255, 255), 2)

        # Draw center circle
        cv2.circle(frame, (self.frame_width // 2, self.frame_height // 2), 70, (255, 255, 255), 2)

        # Draw left goal
        cv2.rectangle(frame, (self.left_goal['x'], self.left_goal['y']),
                      (self.left_goal['x'] + self.left_goal['width'], self.left_goal['y'] + self.left_goal['height']),
                      (0, 0, 255), 2)

        # Draw right goal
        cv2.rectangle(frame, (self.right_goal['x'], self.right_goal['y']),
                      (self.right_goal['x'] + self.right_goal['width'], self.right_goal['y'] + self.right_goal['height']),
                      (0, 0, 255), 2)

    def run(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror the frame for a more intuitive experience
            self.draw_field(frame)
            hand_landmarks = self.detect_hands(frame)

            touch_position = None
            if hand_landmarks:
                touch_position = self.check_ball_touch(hand_landmarks)

            if touch_position is not None:
                self.handle_ball_touch(frame, touch_position)

            self.update_ball_position()

            # Draw ball
            cv2.circle(frame, tuple(self.ball_position.astype(int)), self.ball_radius, (255, 255, 0), -1)

            if hand_landmarks:
                for handLms in hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, handLms, self.mp_hands.HAND_CONNECTIONS)

            self.draw_game_info(frame)

            goal_message = self.check_goal()
            if goal_message:
                cv2.putText(frame, goal_message, (self.frame_width // 4, self.frame_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                # Reset ball position after goal
                self.ball_position = np.array([self.frame_width // 2, self.frame_height // 2], dtype=np.float32)
                self.ball_velocity = np.array([0.0, 0.0], dtype=np.float32)

            cv2.imshow("Hand Football Game", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def start(self):
        threading.Thread(target=self.run, daemon=True).start()

    def stop(self):
        self.stop_event.set()

if __name__ == "__main__":
    game = HandFootballGame()
    game.start()

    print("Press Enter to stop the game.")
    input()

    game.stop()
    print("Game stopped.")