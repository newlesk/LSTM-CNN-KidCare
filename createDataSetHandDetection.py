import cv2
import mediapipe as mp
import pandas as pd

# สร้างโมดูล Mediapipe สำหรับการตรวจจับมือ
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# สร้าง DataFrame สำหรับเก็บข้อมูล
columns = ['frame_id', 'hand_id'] + [f'x{i},y{i},z{i}' for i in range(21)]  # 21 จุดของมือ (x, y, z)
data = []

# เปิดการอ่านวิดีโอจากไฟล์หรือกล้อง
cap = cv2.VideoCapture('video_file.mp4')  # หรือใช้ 0 เพื่อใช้กล้อง

frame_id = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # เปลี่ยนภาพเป็น RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # ตรวจจับมือในภาพ
    result = hands.process(frame_rgb)
    
    if result.multi_hand_landmarks:
        for hand_id, hand_landmarks in enumerate(result.multi_hand_landmarks):
            # ดึงข้อมูลจุดต่าง ๆ ของมือ
            row = [frame_id, hand_id]  # เก็บ frame_id และ hand_id
            for landmark in hand_landmarks.landmark:
                row.extend([landmark.x, landmark.y, landmark.z])
            
            data.append(row)  # เพิ่มข้อมูลลงใน DataFrame

            # วาดผลลัพธ์บนภาพ
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    frame_id += 1

    # แสดงภาพ
    cv2.imshow('Hand Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# สร้าง DataFrame และบันทึกเป็นไฟล์ CSV
df = pd.DataFrame(data, columns=columns)
df.to_csv('hand_data_for_cnn_lstm.csv', index=False)

print("Data set ถูกสร้างและบันทึกเป็นไฟล์ CSV ชื่อ 'hand_data_for_cnn_lstm.csv'")
