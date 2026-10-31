# Hướng Dẫn Cài Đặt & Sử Dụng Hệ Thống Chess System (Chess960 Arena & Benchmark)

Hệ thống hỗ trợ thi đấu đối kháng nhiều bàn (Arena), chạy benchmark chấm điểm, lưu trữ PGN lịch sử ván đấu và xem lại tương tác trên giao diện Web.

---

## 1. Yêu cầu hệ thống
- Hệ điều hành: Windows, macOS hoặc Linux.
- Đã cài đặt [Git](https://git-scm.com/).
- Đã cài đặt [Python 3.11+](https://www.python.org/) hoặc [uv](https://docs.astral.sh/uv/).

---

## 2. Cách chạy nhanh trên Windows (Khuyên dùng cho thành viên nhóm)

Chỉ cần clone project về máy và thực hiện:

### Bước 1: Clone repo
```bash
git clone https://github.com/dinhtu2007bn/chess_system.git
cd chess_system
```

### Bước 2: Chạy hệ thống
- Click đúp vào file **`CHAY_HET_THONG.bat`**.
- File `.bat` sẽ:
  1. Tự động kiểm tra môi trường ảo `.venv`. Nếu chưa có, hệ thống sẽ tự động tạo và cài đặt đầy đủ thư viện.
  2. Tự động mở trình duyệt web tại địa chỉ: **`http://127.0.0.1:8888`**.
  3. Khởi chạy máy chủ FastAPI & WebSocket.

---

## 3. Cách chạy thủ công bằng dòng lệnh (Dành cho macOS / Linux hoặc nếu không dùng file .bat)

### Bước 1: Tạo môi trường ảo & Cài đặt thư viện
```bash
# Cách 1: Sử dụng uv (cực nhanh)
uv sync

# Cách 2: Sử dụng pip thông thường
python -m venv .venv
source .venv/bin/activate  # Trên Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Bước 2: Khởi chạy máy chủ
```bash
python -m uvicorn app.server:app --host 0.0.0.0 --port 8888
```
Sau đó mở trình duyệt truy cập: `http://127.0.0.1:8888`

---

## 4. Các tính năng chính trên giao diện Web
1. **Thi đấu Bot vs Bot (Arena):**
   - Chọn 2 bot bất kỳ trong danh sách (V0 đến V11, Bot Test, Bot Son...).
   - Chạy đồng thời nhiều bàn cờ (hỗ trợ 1 đến 8 bàn cùng lúc).
   - Chọn thể thức thời gian (30s, 60s, 180s, 360s...) và số ván đấu.
2. **Benchmark & Chấm điểm:**
   - Chấm điểm tối ưu của bot theo 3 tiêu chí: Win rate, Time Efficiency, Timeout Penalty.
3. **Lịch sử ván đấu (Match History):**
   - Toàn bộ ván đấu được tự động lưu thành file PGN trong thư mục `history_matches/`.
   - Xem lại từng nước đi, đồ thị thời gian, đồ thị điểm số trực tiếp trên web.
4. **Chia sẻ cho người khác xem online:**
   - Khi đang bật `CHAY_HET_THONG.bat`, click đúp file **`CHIA_SE_ONLINE.bat`**.
   - Copy đường link `https://xxxx.trycloudflare.com` gửi cho đồng đội để họ truy cập từ xa.
