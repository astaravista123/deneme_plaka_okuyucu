import cv2
import pytesseract
import numpy as np
from collections import Counter

# Tesseract'ın bulunduğu dizin (Kendi bilgisayarında tesseract'ın yolunu belirt)
pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# OCR için yapılandırma
config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -c tessedit_char_blacklist=!?@#$%^&*()_+=<>[]'

# Görüntü ön işleme fonksiyonu
def preprocess_plate_region(plate_region):
    plate_gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
    # CLAHE ile kontrast artırma
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    plate_clahe = clahe.apply(plate_gray)
    # Adaptive Thresholding
    preprocessed_plate = cv2.adaptiveThreshold(plate_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return preprocessed_plate

# OCR sonrası sonuç üzerinde manuel düzeltme
def correct_ocr_output(ocr_text):
    corrections = {
        '0': 'O',  # Sıfır yerine O harfi
        'O': '0',  # O harfi yerine sıfır
        'I': '1',  # I harfi yerine 1
        '1': 'I',  # 1 yerine I harfi
        'S': '5',  # S harfi yerine 5
        '5': 'S',  # 5 yerine S harfi
        'B': '8',  # B harfi yerine 8
        '8': 'B',  # 8 yerine B harfi
        'D': '0',  # D harfi yerine sıfır
        '0': 'D',  # Sıfır yerine D harfi
        'Z': '2',  # Z harfi yerine 2
        '2': 'Z',  # 2 yerine Z harfi
        'P': 'A',  # P yerine A harfi
        'A': 'P'   # A yerine P harfi
    }
    for wrong_char, correct_char in corrections.items():
        ocr_text = ocr_text.replace(wrong_char, correct_char)
    return ocr_text

# Türk plakalarının formatına uygun olup olmadığını kontrol eden fonksiyon
def is_valid_turkish_plate(plate_text):
    # Türk plakaları genellikle en az 6 ve en fazla 8 karakterden oluşur
    if len(plate_text) < 6 or len(plate_text) > 8:
        return False
    # Plakanın ilk iki karakteri sayı, kalan kısmı alfanumerik olmalıdır
    if not (plate_text[:2].isdigit() and plate_text[2:].isalnum()):
        return False
    return True

# Bir karede tespit edilen plakaların OCR sonuçlarını toplama
def get_most_frequent_ocr(results_list):
    if results_list:
        return Counter(results_list).most_common(1)[0][0]
    return ""

# Video işlemesi
video_path = 'C:/Intel/Pyhon2/traffic_test2.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video açılamadı.")

# Haar Cascade kullanarak plaka tespiti
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    plate_ocr_results = []

    for (x, y, w, h) in plates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        plate_region = frame[y:y + h, x:x + w]
        preprocessed_plate = preprocess_plate_region(plate_region)

        # Plaka çözünürlüğünü artırma
        plate_resized = cv2.resize(preprocessed_plate, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        # OCR işlemi
        plate_text = pytesseract.image_to_string(plate_resized, config=config)
        plate_text = ''.join(e for e in plate_text if e.isalnum())  # Sadece alfanumerik karakterler
        plate_text = correct_ocr_output(plate_text)  # OCR düzeltmesi

        # Tespit edilen plaka terminale yazdırılıyor
        print(f"Tespit edilen plaka: {plate_text}")

        # Plaka geçerli Türk plakası formatına uygun mu kontrol et
        if is_valid_turkish_plate(plate_text):
            plate_ocr_results.append(plate_text)

        # Her plakayı tespit ettikten sonra doğrudan video üzerine yazdır
        if plate_text:
            cv2.putText(frame, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Video gösterimi
    cv2.imshow('Plaka Tespiti', frame)
    
    # Videoyu durdurmak için 'q' tuşuna basılması yeterli
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
