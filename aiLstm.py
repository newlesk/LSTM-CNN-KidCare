import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# โหลดข้อมูลจากไฟล์ CSV
dataset = pd.read_csv('emg_dataset_example.csv')

# กำหนดค่าพารามิเตอร์
timesteps = 10  # จำนวนเฟรมใน 1 วินาที
num_features = 63  # จำนวนคุณลักษณะ (21 จุด x 3 แกน)
num_outputs = 5  # จำนวนตำแหน่ง EMG ที่ต้องการทำนาย

# แยกข้อมูลอินพุตและเอาต์พุต
X = dataset.iloc[:, :-num_outputs].values.reshape(-1, timesteps, num_features)
y = dataset.iloc[:, -num_outputs:].values

# แบ่งข้อมูลเป็นชุดเทรนและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(timesteps, num_features)))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(50, activation='relu'))
model.add(Dense(num_outputs))  # ทำนายค่า EMG 5 ตำแหน่ง

# คอมไพล์โมเดล
model.compile(optimizer='adam', loss='mean_squared_error')

# ฝึกโมเดล
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2)

# ทดสอบโมเดลและคำนวณค่า Mean Squared Error (MSE)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse:.4f}')

# คำนวณ Accuracy แบบง่าย ๆ ด้วยการใช้ค่า MSE
accuracy = 100 - mse
print(f'Accuracy: {accuracy:.2f}%')

# แสดงผลลัพธ์ตัวอย่างการทำนายค่า EMG
print("ตัวอย่างผลลัพธ์จากการทำนายค่า EMG:")
for i in range(5):
    print(f"ตัวอย่างที่ {i+1}: ค่าจริง: {y_test[i]}, ค่าที่ทำนาย: {y_pred[i]}")
