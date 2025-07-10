# ADAS_EC
Bu bir araç adas sistemidir. Şerit takip uyarı sistemi, Trafik ışık ve tabela tanıma sistemi, Arka kamera ve Navigasyon içerir. GPS daha eklenmemiştir. 

<img width="2560" height="1440" alt="Screenshot from 2000-01-01 09-44-53" src="https://github.com/user-attachments/assets/bb049530-285f-4ea1-9d62-df04882e16bd" />

Darknet YOLOv4 mimarisi kullanılarak bir model eğitilmiş ve sisteme entegre edilmiştir. Model ağırlık dosyası ve gerekli dosyalar: https://drive.google.com/drive/folders/1zg17FC5uFDRiNPHP2B5R4005A0Cv4Y5P?usp=sharing

Sistem NVIDIA Jetson AGX Xavier üzerinde çalıştırılmıştır. Kamera ve buzzer kontrolü için JETGPIO kütüphanesindeki uyumlu pinler kullanılmıştır.

![xavier](https://github.com/user-attachments/assets/8efa963f-0a2e-4dce-9207-279c319ebba4)

