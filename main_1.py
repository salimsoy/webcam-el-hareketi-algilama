import cv2
import numpy as np
import mediapipe as mp

class HandDetect:
    def __init__(self):
        self.last_rect = None 
        self.start = False
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.color = np.random.randint(0, 255, (100, 3))
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1)  # En fazla 2 el algılansın
        self.mp_draw = mp.solutions.drawing_utils 
    

    def FlowPyrLK(self, prev_gray, next_gray, frame2):
        if self.p0 is None:
            return frame2  # takip noktası yoksa boş işlem
    
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, self.p0, None, **self.lk_params)
        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]
    
            if len(good_new) == 0:
                self.start = False
                return frame2
            
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
                frame2 = cv2.circle(frame2, (int(a), int(b)), 5, self.color[i].tolist(), -1)
            
            self.p0 = good_new.reshape(-1, 1, 2)
            self.img = cv2.add(frame2, self.mask)
            return self.img
        else:
            self.start = False
            return frame2

        
    
    def main(self):
        
        cap = cv2.VideoCapture(0)  
        ret, frame1 = cap.read()
        prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        
        self.mask = np.zeros_like(frame1)
        
        while True:
            ret, frame2 = cap.read()
            if not ret:
                break
            next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            rgb_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb_frame)
            if result.multi_hand_landmarks:
                if not self.start:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    self.mp_draw.draw_landmarks(frame2, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
                    thumb_tip = hand_landmarks.landmark[8]
                    h, w, _ = frame2.shape
                    cx, cy = int(thumb_tip.x * w), int(thumb_tip.y * h)
                    cv2.circle(frame2, (cx, cy), 10, (0, 255, 0), -1)
                    self.p0 = np.array([[[cx, cy]]], dtype=np.float32)
                    self.start = True
                    
                self.img = self.FlowPyrLK(prev_gray, next_gray, frame2)
                cv2.imshow('Hareketli Nesne Takibi', self.img)
            else:
                cv2.imshow('Hareketli Nesne Takibi', frame2)
        
            if cv2.waitKey(1) & 0xFF == 27:
                break
            prev_gray = next_gray.copy()
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':

    process = HandDetect()
    process.main()
