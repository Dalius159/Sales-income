# Import thư viện cần thiết
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Giả sử dữ liệu đã có sẵn trong file CSV "sales_data.csv"
# Tải dữ liệu
data = pd.read_csv("sales_data.csv")

# Xem trước dữ liệu
print(data.head())

# Kiểm tra giá trị bị thiếu
print(data.isnull().sum())

# Xử lý các giá trị bị thiếu nếu có
data = data.dropna()  # Hoặc có thể dùng các cách khác tùy yêu cầu dữ liệu

# Chọn các cột đầu vào (X) và đầu ra (y)
X = data[['Ad_Cost', 'Inventory', 'Potential_Customers']]
y = data['Sales']  # Cột dự đoán là "Sales" (doanh số bán hàng)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình hồi quy tuyến tính
model = LinearRegression()

# Huấn luyện mô hình với tập huấn luyện
model.fit(X_train, y_train)

# Dự đoán doanh số trên tập kiểm tra
y_pred = model.predict(X_test)

# Tính toán độ lỗi bình phương trung bình (Mean Squared Error) và hệ số R^2
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)


# Vẽ biểu đồ so sánh giữa y_test (giá trị thực tế) và y_pred (giá trị dự đoán)
plt.scatter(y_test, y_pred)
plt.xlabel("Giá trị thực tế")
plt.ylabel("Giá trị dự đoán")
plt.title("Biểu đồ so sánh giữa giá trị thực tế và dự đoán")
plt.show()
