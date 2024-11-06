# chạy file này để tạo file sales_data.csv (mẫu thử để train cho file model sales_volume.py)
# lệnh để chạy: python data_genarator.py
import pandas as pd
import numpy as np

# Tạo dữ liệu giả lập cho file CSV
np.random.seed(42)
num_samples = 1000  # Số lượng mẫu (muốn tăng số lượng mẫu thửu thì sửa chỗ này)
# để nhiều nhất 10000 mẫu thôi không chạy lâu lắm

# Tạo các cột dữ liệu: chi phí quảng cáo, số lượng tồn kho, số khách hàng tiềm năng, và doanh số bán hàng
data = {
    "Ad_Cost": np.random.uniform(1000, 5000, num_samples),
    "Inventory": np.random.uniform(50, 300, num_samples).astype(int),  # Chuyển thành số nguyên
    "Potential_Customers": np.random.uniform(200, 1000, num_samples).astype(int),  # Chuyển thành số nguyên
    "Sales": lambda df: df["Ad_Cost"] * 0.1 + df["Inventory"] * 0.5 + df["Potential_Customers"] * 0.2 + np.random.normal(0, 10, num_samples)
}


# Chuyển dữ liệu thành DataFrame và tính toán doanh số dựa vào công thức trên
df = pd.DataFrame(data)
df["Sales"] = data["Sales"](df)

# Lưu dữ liệu vào file CSV
output_path = "sales_data.csv"
df.to_csv(output_path, index=False)

output_path
