# 🧠 AMPerfume ROCK Clustering App

Ứng dụng web dùng thuật toán **ROCK Clustering** để phân cụm dữ liệu hỗn hợp (số + phân loại), có giao diện trực quan bằng React.

---

## 📁 Cấu trúc dự án

```
AMPerfume-Cluster/
├── backend/                 # Flask API xử lý phân cụm
│   ├── app.py              # Server chính
│   ├── rock.py             # ROCK hỗn hợp (numeric + categorical)
│   ├── rock_basic.py       # ROCK đơn giản (chỉ categorical)
│   ├── rock_fast.py        # ROCK sample nhanh
├── frontend/               # React app giao diện người dùng
│   ├── App.tsx             # Routing trang
│   ├── pages/              # /rock-basic, /rock-hybrid, /rock-fast
│   ├── components/         # ClusterForm xử lý chính
├── README.md
├── requirements.txt        # Python packages
```

---

## ⚙️ Cài đặt

### Backend (Python)
```bash
cd backend
python -m venv venv
source venv/bin/activate  # hoặc venv\Scripts\activate nếu Windows
pip install -r requirements.txt
python app.py
```

### Frontend (React)
```bash
cd frontend
npm install
npm run dev
```

---

## 🌐 Các trang chính

- `/rock-basic` → ROCK cổ điển (categorical-only)
- `/rock-hybrid` → ROCK hỗn hợp (categorical + numeric)
- `/rock-fast` → Chạy nhanh trên mẫu

---

## 🎯 Tính năng

- Upload file .csv/.xlsx
- Chọn cột & tham số K, theta
- Hiển thị cụm, scatter plot, pie chart
- Tải file `.xlsx`

---

## ✅ Ghi chú
- Thư viện Python cần thiết nằm trong `requirements.txt`

---

