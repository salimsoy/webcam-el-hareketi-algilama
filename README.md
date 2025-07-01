# Webcam ile Hareketli Nesne Takip Sistemi
Bu proje, OpenCV kullanarak kamera görüntüsünde hareketli nesneleri tespit edip bu nesneleri takip eden bir uygulamadır ve burada nesne olarak işaret parmağını seçeriz.
1- **Hareketli nesne tespiti** – Farneback Optical Flow algoritması kullanılarak, **Nesne takibi** – Lucas-Kanade (Pyramidal LK) yöntemi ile yapılır.
2- **MediaPipe** ile eldeki parmak ucunu algılayarak **Lucas-Kanade Optical Flow** yöntemi ile hareketini takip eder.

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

**Temel Mantık: 1**
