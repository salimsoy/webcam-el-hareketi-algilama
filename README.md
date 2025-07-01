# Webcam ile Hareketli Nesne Takip Sistemi
Bu proje, OpenCV kullanarak kamera görüntüsünde hareketli nesneleri tespit edip bu nesneleri takip eden bir uygulamadır ve burada nesne olarak işaret parmağını seçeriz.

-  **Hareketli nesne tespiti** – Farneback Optical Flow algoritması kullanılarak, **Nesne takibi** – Lucas-Kanade yöntemi ile yapılır.

-  **MediaPipe** ile eldeki parmak ucunu algılayarak **Lucas-Kanade Optical Flow** yöntemi ile hareketini takip eder.

## Lucas-Kanade
Lucas-Kanade, optik akışın en çok bilinen yöntemlerinden biridir. Yalnızca bazı önemli noktaların (örneğin köşe veya kenar gibi dikkat çeken yerlerin) hareketini takip eder. Bu yüzden seyrek optik akış denir.
Videodaki belirli noktaları seçer. Bir sonraki karede bu noktaların nerede olduğunu tahmin eder. Hızlı ve hesaplaması kolaydır. Küçük hareketlerde başarılıdır ancak Tüm görüntü yerine sadece seçilen noktaları takip eder.

## Dense Optik Akış (Farneback)
Lucas-Kanade yöntemi, seyrek bir özellik kümesi için optik akış hesaplar. Buna karşılık OpenCV, tüm piksellerin hareketini tahmin edebilen Gunnar Farneback algoritmasını da sunar. 
Dense optik akış, bir sahnedeki nesnelerle kamera arasındaki bağıl harekete bağlı olarak, ardışık video karelerinde görülen nesne hareketlerini tahmin etme yöntemidir. 

**Temel Mantık: 1**
- Kamera Açılır İlk Kare Alınır Farneback ile Hareket Tespiti
- Hareketli Nesne Dikdörtgen ile Belirlenir ve sol üst köşesine nokta koyulur.
- Kullanıcı `m` harfine tıkladığında takip noktası ayarlanır
- Bu noktadan Lucas-Kanade Optical Flow ile Takip Başlar
- Nesnenin Hareketi Renkli Çizgilerle İzlenir
- `esc` tuşuna basıldığında uygulama sonlandırılır.

**Temel Mantık: 2**
- Kamera görüntüsü RGB formatına dönüştürülür.
- MediaPipe ile eldeki landmark noktalar bulunur.
- İşaret parmağının ucu (`landmark[8]`) başlangıç takip noktası olarak alınır.
- Takip noktası sonraki karede nereye gittiği Lucas-Kanade ile hesaplanır.
- Her karede hareketin izi renkli çizgilerle gösterilir.

**Avantajları: 1**
- El, top, kutu gibi hareket eden herhangi bir cismi tespit eder.
- Spesifik nesne tanımına gerek yok.
- Kullanıcı, hangi nesnenin takip edileceğini manuel olarak belirler.

**Dezavantajları: 1**
- Arka planda küçük bir hareket bile tespit edilir, istenmeyen dikdörtgenler çizilebilir.
- İlk tespit yanlışa dayalıysa takip de yanlış olur.

**Avantajları: 2**
- MediaPipe, doğrudan eli ve landmark noktalarını tespit eder.
- El dışında başka nesneye odaklanmaz, daha temiz çalışır.

**Dezavantajları: 2**
- Kurulumu biraz daha zahmetli olabilir.
- Özellikle düşük donanımlı sistemlerde performans düşebilir.

# Webcam ile El Takibi Uygulaması
### `main_1.py`
- MediaPipe ile el algılanır.
- İşaret parmağının ucu (landmark 8) takip başlangıç noktası olarak alınır.
- Her karede bu noktanın hareketi Lucas-Kanade ile takip edilir.
- Takip noktası ekranda çizgi ve daireyle görselleştirilir.

Aşağıda Python kodu ve açıklamaları yer almaktadır.
```python
import cv2
import numpy as np
import mediapipe as mp

class HandDetect:
    def __init__(self):
        # Son tespit edilen el noktasının koordinatları
        self.last_rect = None 
        
        # Takip işlemi başladı mı?
        self.start = False

        # Lucas-Kanade Optical Flow parametreleri
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Takip noktaları için rastgele renkler (iz çizgileri)
        self.color = np.random.randint(0, 255, (100, 3))

        # MediaPipe el algılama modülü
        self.mp_hands = mp.solutions.hands

        # Tek bir el için el algılayıcıyı başlat
        self.hands = self.mp_hands.Hands(max_num_hands=1)

        # El çizim yardımcı sınıfı (landmark çizmek için)
        self.mp_draw = mp.solutions.drawing_utils 

    # Lucas-Kanade Optical Flow ile takip işlemi
    def FlowPyrLK(self, prev_gray, next_gray, frame2):
        # Eğer takip edilecek nokta yoksa, görüntüyü direkt döndür
        if self.p0 is None:
            return frame2
    
        # Optical flow ile bir sonraki karedeki yeni konumu bul
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, self.p0, None, **self.lk_params)

        # Takip noktaları varsa işleme devam et
        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]
    
            # Eğer geçerli nokta kalmadıysa takip durdurulsun
            if len(good_new) == 0:
                self.start = False
                return frame2
            
            # Her bir takip edilen nokta için çizim yapılır
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                # Hareket çizgisi çiz
                self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
                # Takip edilen nokta üzerine daire çiz
                frame2 = cv2.circle(frame2, (int(a), int(b)), 5, self.color[i].tolist(), -1)
            
            # Yeni noktaları bir sonraki kare için güncelle
            self.p0 = good_new.reshape(-1, 1, 2)

            # Orijinal görüntü ile çizilen maskeyi birleştir
            self.img = cv2.add(frame2, self.mask)
            return self.img
        else:
            # Takip başarısızsa durdur
            self.start = False
            return frame2

    # Ana döngü: kamera açılır, el algılama ve takip yapılır
    def main(self):
        cap = cv2.VideoCapture(0)  # Kamera başlatılır
        ret, frame1 = cap.read()   # İlk kare alınır
        prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # Gri tonlamaya çevrilir

        # Çizim için boş bir maske oluştur (aynı boyutta)
        self.mask = np.zeros_like(frame1)
        
        while True:
            ret, frame2 = cap.read()
            if not ret:
                break

            # Görüntüler gri ve RGB formatına çevrilir
            next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            rgb_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

            # MediaPipe ile elleri algıla
            result = self.hands.process(rgb_frame)

            # Eğer bir el algılandıysa
            if result.multi_hand_landmarks:
                # Henüz takip başlamadıysa
                if not self.start:
                    # İlk elin landmark'larını al
                    hand_landmarks = result.multi_hand_landmarks[0]

                    # Landmark'ları çiz
                    self.mp_draw.draw_landmarks(frame2, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # İşaret parmağı ucu (landmark 8) alınır
                    thumb_tip = hand_landmarks.landmark[8]

                    # Piksel koordinatına çevrilir
                    h, w, _ = frame2.shape
                    cx, cy = int(thumb_tip.x * w), int(thumb_tip.y * h)

                    # Parmağın ucuna yeşil daire çiz
                    cv2.circle(frame2, (cx, cy), 10, (0, 255, 0), -1)

                    # Takip başlangıç noktası olarak ayarlanır
                    self.p0 = np.array([[[cx, cy]]], dtype=np.float32)
                    self.start = True

                # Takip işlemini uygula
                self.img = self.FlowPyrLK(prev_gray, next_gray, frame2)

                # Takip edilen görüntüyü göster
                cv2.imshow('Hareketli Nesne Takibi', self.img)
            else:
                # El algılanmazsa sadece görüntüyü göster
                cv2.imshow('Hareketli Nesne Takibi', frame2)
        
            # ESC tuşuna basılırsa döngüden çık
            if cv2.waitKey(1) & 0xFF == 27:
                break

            # Gelecek kare için önceki kare güncellenir
            prev_gray = next_gray.copy()
        
        # Kaynaklar serbest bırakılır
        cap.release()
        cv2.destroyAllWindows()

# Program başlatıldığında çalıştırılacak ana kısım
if __name__ == '__main__':
    process = HandDetect()
    process.main()
```
### `main.py`
- Farneback yöntemiyle görüntüdeki hareketli bölgeleri tespit eder.
- Hareketli nesnelerin çevresine dikdörtgen çizer.
- m tuşuna basılınca dikdörtgenin ortası takip noktası olarak alınır.
- Lucas-Kanade ile bu nokta kareler arası takip edilir.
- Takip noktalarının izleri renkli çizgilerle çizilir.

Aşağıda Python kodu ve açıklamaları yer almaktadır.
```python
import cv2
import numpy as np

# Hareketli nesne tespiti ve takibi yapan sınıf
class HandDetect:
    def __init__(self):
        self.last_rect = None  # Son tespit edilen dikdörtgenin sol üst köşesi
        self.start = False     # Takip başlatıldı mı?
        
        # Lucas-Kanade Optical Flow parametreleri
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Her takip noktası için rastgele renkler oluştur
        self.color = np.random.randint(0, 255, (100, 3))

    # Farneback Optical Flow ile hareketli alanları tespit eder
    def Farneback(self, prev_gray, next_gray):
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray,
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Hareket vektörlerini büyüklük ve açıya ayır
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Büyüklüğü belli bir eşikten büyük olanları hareketli kabul et
        threshold = 7.5
        motion_area = mag > threshold
        
        return motion_area

    # Hareketli alanların konturlarını bul ve dikdörtgen çiz
    def conturs(self, motion_area, frame2):
        # Konturları tespit et
        contours, _ = cv2.findContours(motion_area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Küçük alanları yoksay
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                # Dikdörtgeni çiz
                cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Son tespit edilen konumu kaydet
                self.last_rect = (x, y)
        
        # Takip başlatılacak nokta (dikdörtgenin ortası)
        if self.last_rect:
            x, y = self.last_rect
            cv2.line(frame2, (x + 20, y + 20), (x + 20, y + 20), (0, 0, 255), 10)

    # Lucas-Kanade yöntemi ile hareketli noktanın takibini yapar
    def FlowPyrLK(self, prev_gray, next_gray, frame2):
        # Takip noktalarının yeni konumunu hesapla
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, self.p0, None, **self.lk_params)
        
        if p1 is not None:
            # Başarılı takip noktalarını ayıkla
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]
        
        # Her başarılı eşleşme için çizim yap
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            # Hareket yönünü gösteren çizgi
            self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
            # Güncel konuma daire çiz
            frame2 = cv2.circle(frame2, (int(a), int(b)), 5, self.color[i].tolist(), -1)
        
        # Maskeyi görüntüye ekle
        self.img = cv2.add(frame2, self.mask)
        
        # Bir sonraki karede kullanılmak üzere takip noktalarını güncelle
        self.p0 = good_new.reshape(-1, 1, 2)

    # Ana program döngüsü
    def main(self):
        cap = cv2.VideoCapture(0)  # Kamerayı başlat
        
        ret, frame1 = cap.read()  # İlk kareyi al
        prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # Griye çevir
        
        self.mask = np.zeros_like(frame1)  # Takip çizgilerini çizeceğimiz boş görüntü
        
        while True:
            ret, frame2 = cap.read()
            if not ret:
                break

            next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)  # Yeni kareyi griye çevir

            if not self.start:
                # Takip başlamadıysa önce hareketli nesneyi bul
                motion_area = self.Farneback(prev_gray, next_gray)
                self.conturs(motion_area, frame2)
                
                # Görüntüyü göster
                cv2.imshow("Hareketli Nesne Takibi", frame2)
                
                # 'm' tuşuna basılırsa takip başlasın
                if cv2.waitKey(1) & 0xFF == ord('m'):
                    x, y = self.last_rect
                    # Takip başlangıç noktası belirlenir
                    self.p0 = np.array([[[x + 20, y + 20]]], dtype=np.float32)
                    self.start = True

            elif self.start:
                # Takip başladıysa Lucas-Kanade ile izlemeye devam et
                self.FlowPyrLK(prev_gray, next_gray, frame2)
                # Güncel görüntüyü göster
                cv2.imshow('Hareketli Nesne Takibi', self.img)

            # ESC tuşuna basılırsa çıkış yap
            if cv2.waitKey(30) & 0xFF == 27:
                break

            # Önceki kareyi güncelle (bir sonraki iterasyon için)
            prev_gray = next_gray.copy()
        
        # Kaynakları serbest bırak
        cap.release()
        cv2.destroyAllWindows()

# Uygulamayı çalıştır
if __name__ == '__main__':
    process = HandDetect()
    process.main()

```


