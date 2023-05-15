import pathlib
import imutils
import time
import dlib
import cv2
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from playsound import playsound


class DrowsinessDetectionApp:
    def __init__(self):
        self.predictor = None
        self.detector = None
        self.project_path = pathlib.Path(__file__).parent
        self.sound_path = self.project_path / 'sounds' / 'alarm.wav'
        self.model_path = 'models/shape_predictor_68_face_landmarks.dat'

        self.load_model()

        self.colors = {
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'red': (0, 0, 255)
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.EYE_AR_THRESH = 0.3
        self.EYE_AR_CONSEC_FRAMES = 30

        self.counter = 0

    def sound_alarm(self, path):
        print("[INFO] Playing sound...")
        playsound(path)
        print("[INFO] Stop sound")

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def load_model(self):
        print("[INFO] Loading facial landmark predictor...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.model_path)
        print("[INFO] Loaded facial landmark predictor")

    def run(self):
        print("[INFO] Starting drowsiness detection...")

        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        print("[INFO] Starting video stream thread...")
        vs = VideoStream(0).start()
        time.sleep(0.5)

        while True:
            frame = vs.read()
            frame = imutils.resize(frame, width=800)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects = self.detector(gray, 0)
            cv2.putText(frame, "PRESS q TO QUIT!", (25, 30),
                        self.font, 0.7, self.colors['blue'], 2)

            for rect in rects:
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)

                if leftEAR < self.EYE_AR_THRESH:
                    cv2.drawContours(frame, [leftEyeHull], -1, self.colors['red'], 1)
                else:
                    cv2.drawContours(frame, [leftEyeHull], -1, self.colors['green'], 1)
                if rightEAR < self.EYE_AR_THRESH:
                    cv2.drawContours(frame, [rightEyeHull], -1, self.colors['red'], 1)
                else:
                    cv2.drawContours(frame, [rightEyeHull], -1, self.colors['green'], 1)

                if ear < self.EYE_AR_THRESH:
                    self.counter += 1
                    if self.counter >= self.EYE_AR_CONSEC_FRAMES:
                        self.counter = 0
                        Thread(target=self.sound_alarm, args=(self.sound_path,), daemon=True, name='backgroundMusicThread').start()
                        cv2.putText(frame, "DROWSINESS ALERT!", (500, 30),
                                    self.font, 0.9, self.colors['red'], 2)
                else:
                    self.counter = 0

                if ear > self.EYE_AR_THRESH:
                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                            self.font, 0.7, self.colors['green'], 2)
                else:
                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                                self.font, 0.7, self.colors['red'], 2)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        print("[INFO] Stoping video stream thread...")
        cv2.destroyAllWindows()
        vs.stop()
        print("[INFO] Stop video stream thread")


if __name__ == "__main__":
    app = DrowsinessDetectionApp()
    app.run()

