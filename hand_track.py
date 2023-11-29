import math
import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode= False, maxHands=2, model_complexity=1, conf_detetion=0.5, conf_track=0.5):
        # Initialize parameters
        self.mode = mode
        self.maxHands = maxHands
        self.complex = model_complexity
        self.conf_detetion = conf_detetion
        self.conf_track = conf_track

        #   Create objects
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.maxHands, self.complex, self.conf_detetion, self.conf_track)
        self.draw = mp.solutions.drawing_utils
        self.tip = [4, 8, 12, 16, 20]

    # Function to find hands
    def findHands(self, frame, draw=True):
        img_color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_color)

        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw:
                    # Draw the conections points
                    self.draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
        return frame

    #Function to find position
    def find_position(self, frame, hand_num=0, draw_points = True, draw_box = True, color = []):
        x_list = []
        y_list = []
        bbox = []
        player = 0
        self.list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_num]
            test = self.results.multi_hand_landmarks
            player = len(test)
            for id, lm in enumerate(my_hand.landmark):
                higth, width, chanels = frame.shape #Get FPS dimensions
                cx, cy = int(lm.x * width), int(lm.y * higth) # Convert data in pixels
                x_list.append(cx)
                y_list.append(cy)
                self.list.append([id, cx, cy])
                if draw_points:
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 0), cv2.FILLED) #Draw a circle

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            bbox = x_min, y_min, x_max, y_max
            if draw_box:
                #Draw rectangule
                cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), color, 2)
        return self.list, bbox, player

    # Funciton to detect and draw up fingers
    def up_fingers(self):
        fingers = []
        if self.list[self.tip[0]][1] > self.list[self.tip[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range (1, 5):
            if self.list[self.tip[id]][2] < self.list[self.tip[id] - 2 ][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    # Function to detect distance between fingers
    def distance(self, p1, p2, frame, draw = True, r=15, t=3):
        x1, y1 = self.list[p1][1:]
        x2, y2 = self.list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), t)
            cv2.circle(frame, (x1, y1), r, (0,0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), r, (0,0, 255), cv2.FILLED)
            cv2.circle(frame, (cx, cy), r, (0,0, 255), cv2.FILLED)
        length = math.hypot(x2-x1, y2-y1)

        return length, fram, [x1, y1, x2, y2, cx, cy]

    # Main function
    def main():
        p_time = 0
        c_time = 0

        # Capute web cam
        cap = cv2.VideoCapture(0)
        # Crete detector object
        detector = handDetector()
        # Do hands detector
        while True:
            ret, frame = cap.read()
            frame = detector.findHands(frame)
            list = bbox = detector.find_position(frame)
            #Show FPS
            c_time = time.time()
            fps = 1 / (c_time - p_time)
            p_time = c_time

            cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

            cv2.imshow("Hands", frame)
            k = cv2.waitKey(1)

            if k == 27:
                break
        cap.realese()
        cv2.destroyAllWindows()
