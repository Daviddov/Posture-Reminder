import cv2
import time
import threading
import mediapipe as mp
import numpy as np

class PostureReminder:
    EAR_SHOULDER_DISTANCE_THRESHOLD = 0.05  # Threshold for ear-shoulder distance
    HEAD_FORWARD_THRESHOLD = 0.03  # Threshold for head forward detection

    def __init__(self, interval=5):
        self.interval = interval
        self.camera = cv2.VideoCapture(0)
        self.stop_event = threading.Event()
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        self.good_posture_landmarks = None

    def capture_image(self):
        ret, frame = self.camera.read()
        return frame if ret else None

    def capture_good_posture(self):
        input("Please sit well in front of the computer and press Enter to capture an image of your correct posture.")
        frame = self.capture_image()
        if frame is not None:
            landmarks = self.extract_landmarks(frame)
            if landmarks:
                self.good_posture_landmarks = landmarks
                print("Good posture captured successfully.")
            else:
                print("Failed to capture good posture. Please try again.")
        else:
            print("Failed to capture image. Please try again.")

    def extract_landmarks(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        return results.pose_landmarks.landmark if results.pose_landmarks else None

    def analyze_posture(self, frame):
        landmarks = self.extract_landmarks(frame)
        if not landmarks:
            print("No pose landmarks detected.")
            return frame

        current_posture = {
            'left_shoulder': np.array([landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                       landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]),
            'right_shoulder': np.array([landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                        landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]),
            'left_ear': np.array([landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x,
                                  landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y]),
            'right_ear': np.array([landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                                   landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].y]),
            'nose': np.array([landmarks[self.mp_pose.PoseLandmark.NOSE.value].x,
                              landmarks[self.mp_pose.PoseLandmark.NOSE.value].y])
        }

        print("Current posture coordinates:")
        for key, value in current_posture.items():
            print(f"{key}: {value}")

        if self.good_posture_landmarks:
            good_posture = {
                'left_shoulder': np.array([self.good_posture_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                           self.good_posture_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]),
                'right_shoulder': np.array([self.good_posture_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                            self.good_posture_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]),
                'left_ear': np.array([self.good_posture_landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x,
                                      self.good_posture_landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y]),
                'right_ear': np.array([self.good_posture_landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                                       self.good_posture_landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].y]),
                'nose': np.array([self.good_posture_landmarks[self.mp_pose.PoseLandmark.NOSE.value].x,
                                  self.good_posture_landmarks[self.mp_pose.PoseLandmark.NOSE.value].y])
            }

            print("Good posture coordinates:")
            for key, value in good_posture.items():
                print(f"{key}: {value}")

            posture_issues = []

            # Check head position relative to shoulders
            left_ear_shoulder_distance = np.linalg.norm(current_posture['left_ear'] - current_posture['left_shoulder'])
            right_ear_shoulder_distance = np.linalg.norm(current_posture['right_ear'] - current_posture['right_shoulder'])
            good_left_ear_shoulder_distance = np.linalg.norm(good_posture['left_ear'] - good_posture['left_shoulder'])
            good_right_ear_shoulder_distance = np.linalg.norm(good_posture['right_ear'] - good_posture['right_shoulder'])

            if np.abs(left_ear_shoulder_distance - good_left_ear_shoulder_distance) > self.EAR_SHOULDER_DISTANCE_THRESHOLD:
                posture_issues.append("Head position on the left side differs significantly from good posture")
                self.draw_line(frame, current_posture['left_ear'], current_posture['left_shoulder'], (255, 0, 0))

            if np.abs(right_ear_shoulder_distance - good_right_ear_shoulder_distance) > self.EAR_SHOULDER_DISTANCE_THRESHOLD:
                posture_issues.append("Head position on the right side differs significantly from good posture")
                self.draw_line(frame, current_posture['right_ear'], current_posture['right_shoulder'], (255, 0, 0))

            # Check if the head is too far forward
            nose_shoulder_distance = np.linalg.norm(current_posture['nose'] - ((current_posture['left_shoulder'] + current_posture['right_shoulder']) / 2))
            good_nose_shoulder_distance = np.linalg.norm(good_posture['nose'] - ((good_posture['left_shoulder'] + good_posture['right_shoulder']) / 2))

            if np.abs(nose_shoulder_distance - good_nose_shoulder_distance) > self.HEAD_FORWARD_THRESHOLD:
                posture_issues.append("Head is too far forward")
                self.draw_line(frame, current_posture['nose'], ((current_posture['left_shoulder'] + current_posture['right_shoulder']) / 2), (0, 255, 255))

            if posture_issues:
                print("Bad posture detected!")
                for issue in posture_issues:
                    print(f"- {issue}")
                cv2.putText(frame, "Bad Posture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                print("Posture is good.")
                cv2.putText(frame, "Good Posture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        self.mp_drawing.draw_landmarks(frame, self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Posture Analysis', frame)
        cv2.waitKey(1)

        print("-" * 30)

    @staticmethod
    def draw_line(image, start_point, end_point, color):
        height, width = image.shape[:2]
        start = (int(start_point[0] * width), int(start_point[1] * height))
        end = (int(end_point[0] * width), int(end_point[1] * height))
        cv2.line(image, start, end, color, 3)  # Adjusted thickness for better visibility

    def run(self):
        while not self.stop_event.is_set():
            frame = self.capture_image()
            if frame is not None:
                print("\nAnalyzing posture...")
                self.analyze_posture(frame)
            time.sleep(self.interval)

    def start(self):
        self.capture_good_posture()
        threading.Thread(target=self.run, daemon=True).start()

    def stop(self):
        self.stop_event.set()
        self.camera.release()
        self.pose.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    reminder = PostureReminder(interval=0.1)
    reminder.start()

    print("Posture Reminder is running. Press Enter to stop.")
    input()

    reminder.stop()
    print("Posture Reminder stopped.")
