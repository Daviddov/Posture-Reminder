import cv2
import numpy as np
import mediapipe as mp

class FingerDrawing:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open video capture.")
            exit()

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                         min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.canvas = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        self.previous_point = None
        self.drawing_color = (0, 255, 0)  # Green color
        self.drawing_thickness = 5
        self.eraser_thickness = 20  # Thickness for eraser
        self.is_drawing = False  # Flag for drawing mode

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Get index finger tip coordinates
                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_x, index_finger_y = int(index_finger_tip.x * self.frame_width), int(index_finger_tip.y * self.frame_height)

                # Get middle finger tip coordinates
                middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_finger_x, middle_finger_y = int(middle_finger_tip.x * self.frame_width), int(middle_finger_tip.y * self.frame_height)

                # Calculate distance between index and middle finger tips
                distance = np.hypot(middle_finger_x - index_finger_x, middle_finger_y - index_finger_y)

                # Draw or erase on canvas
                if distance < 40:  # Distance threshold for drawing mode
                    self.is_drawing = True
                else:
                    self.is_drawing = False

                if self.previous_point:
                    if self.is_drawing:
                        cv2.line(self.canvas, self.previous_point, (index_finger_x, index_finger_y), self.drawing_color, self.drawing_thickness)
                    else:
                        cv2.line(self.canvas, self.previous_point, (index_finger_x, index_finger_y), (0, 0, 0), self.eraser_thickness)
                self.previous_point = (index_finger_x, index_finger_y)

                # Display coordinates on frame
                cv2.putText(frame, f"X: {index_finger_x} Y: {index_finger_y}", (index_finger_x, index_finger_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            self.previous_point = None

        return frame

    def run(self):
        cv2.namedWindow("Finger Drawing", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Finger Drawing", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror the frame
            frame = self.process_frame(frame)

            # Combine frame and canvas
            combined_image = cv2.addWeighted(frame, 1, self.canvas, 0.5, 0)

            cv2.imshow("Finger Drawing", combined_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.canvas = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            elif key == ord('r'):
                self.drawing_color = (0, 0, 255)  # Red
            elif key == ord('g'):
                self.drawing_color = (0, 255, 0)  # Green
            elif key == ord('b'):
                self.drawing_color = (255, 0, 0)  # Blue
            elif key == ord('+'):
                self.drawing_thickness = min(30, self.drawing_thickness + 1)
            elif key == ord('-'):
                self.drawing_thickness = max(1, self.drawing_thickness - 1)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = FingerDrawing()
    app.run()
