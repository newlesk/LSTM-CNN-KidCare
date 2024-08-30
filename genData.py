import pandas as pd
import numpy as np

# กำหนดค่าพารามิเตอร์สำหรับ Data Set
num_samples = 100  # จำนวนตัวอย่าง
timesteps = 10  # จำนวนเฟรมใน 1 วินาที
num_features = 63  # จำนวนคุณลักษณะ (21 จุด x 3 แกน)
num_outputs = 5  # จำนวนตำแหน่ง EMG ที่ต้องการทำนาย

# สุ่มข้อมูลอินพุต (Input) ที่มีค่าในช่วง [0, 1]
X = np.random.rand(num_samples, timesteps, num_features)

# สุ่มข้อมูลเอาต์พุต (Output) ที่มีค่าในช่วง [0, 0.3]
y = np.random.rand(num_samples, num_outputs)

# แปลงข้อมูลเป็น DataFrame เพื่อบันทึกเป็น CSV
X_flat = X.reshape(num_samples, -1)  # ทำให้เป็นข้อมูลแบนเพื่อบันทึกง่าย
y_flat = y

# สร้าง DataFrame สำหรับอินพุตและเอาต์พุต
df_X = pd.DataFrame(X_flat, columns=[f'feature_{i}' for i in range(X_flat.shape[1])])
df_y = pd.DataFrame(y_flat, columns=[f'emg_output_{i}' for i in range(y_flat.shape[1])])

# รวม DataFrame อินพุตและเอาต์พุต
df = pd.concat([df_X, df_y], axis=1)

# บันทึก Data Set ลงในไฟล์ CSV
df.to_csv('emg_dataset_example.csv', index=False)

print("Data Set ถูกสร้างและบันทึกลงไฟล์ 'emg_dataset_example.csv' เรียบร้อยแล้ว!")
