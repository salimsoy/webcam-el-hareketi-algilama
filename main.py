import cv2
import numpy as np

class HandDetect:
    def __init__(self):
        self.last_rect = None 
        self.start = False
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.color = np.random.randint(0, 255, (100, 3))
        
    def Farneback(self, prev_gray, next_gray):
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray,
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        threshold = 7.5
        motion_area = mag > threshold
        
        return motion_area
      
    
    def conturs(self, motion_area, frame2):
        contours, _ = cv2.findContours(motion_area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        

        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.last_rect = (x, y)
    
        if self.last_rect:
            x, y = self.last_rect
            cv2.line(frame2, (x + 20, y + 20), (x + 20, y + 20), (0, 0, 255), 10)
    
        
    def FlowPyrLK(self, prev_gray, next_gray, frame2):
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, self.p0, None, **self.lk_params)
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
            frame2 = cv2.circle(frame2, (int(a), int(b)), 5, self.color[i].tolist(), -1)
        
        self.img = cv2.add(frame2, self.mask)
        self.p0 = good_new.reshape(-1, 1, 2)
        
    
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
            
            if not self.start:
            
                motion_area = self.Farneback(prev_gray, next_gray)
                self.conturs(motion_area, frame2)
              
                cv2.imshow("Hareketli Nesne Takibi", frame2)
                
                if cv2.waitKey(1) & 0xFF == ord('m'):
                    x, y = self.last_rect
                    self.p0 = np.array([[[x + 20, y + 20]]], dtype=np.float32)
                    self.start = True
            
            
            elif self.start:
                
                self.FlowPyrLK(prev_gray, next_gray, frame2)
                cv2.imshow('Hareketli Nesne Takibi', self.img)
            
            if cv2.waitKey(30) & 0xFF == 27:
                break
            prev_gray = next_gray.copy()
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':

    process = HandDetect()
    process.main()
