# AppBank - Phân tích giao dịch ngân hàng bằng AI

Ứng dụng phân tích và quản lý giao dịch ngân hàng từ văn bản được chia sẻ, sử dụng Google Gemini AI và Supabase.

## Tính năng

- 🤖 Phân tích văn bản giao dịch bằng Google Gemini AI
- 💾 Lưu trữ dữ liệu trên Supabase
- 📊 Thống kê giao dịch theo thời gian thực
- 🔍 Trích xuất thông tin: số tài khoản, số tiền, nội dung, loại giao dịch
- 📱 Giao diện web responsive đơn giản

## Cài đặt

1. **Clone dự án:**

```bash
git clone <repo-url>
cd AppBank
```

2. **Cài đặt dependencies:**

```bash
npm install
```

3. **Cấu hình environment variables:**
   Tạo file `.env` và điền thông tin:

```env
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
GEMINI_API_KEY=your_gemini_api_key
PORT=3000
```

4. **Setup Supabase Database:**

- Tạo project mới trên [Supabase](https://supabase.com)
- Chạy script SQL trong file `database/schema.sql`

5. **Lấy Google Gemini API Key:**

- Truy cập [Google AI Studio](https://makersuite.google.com/app/apikey)
- Tạo API key mới

6. **Chạy ứng dụng:**

```bash
npm run dev
```

## Sử dụng

1. Mở trình duyệt và truy cập `http://localhost:3000`
2. Copy/paste nội dung giao dịch từ app ngân hàng
3. Click "Phân tích giao dịch"
4. Xem kết quả phân tích và thống kê

## API Endpoints

### POST /api/transactions/process

Xử lý và phân tích văn bản giao dịch

**Request:**

```json
{
  "transactionText": "Nội dung giao dịch từ app ngân hàng..."
}
```

**Response:**

```json
{
  "success": true,
  "message": "Giao dịch đã được xử lý thành công",
  "data": {
    "transaction": { ... },
    "extracted_info": { ... }
  }
}
```

### GET /api/transactions

Lấy danh sách giao dịch

**Query Parameters:**

- `account_number`: Lọc theo số tài khoản
- `limit`: Giới hạn số bản ghi (mặc định: 100)

### GET /api/transactions/stats

Lấy thống kê giao dịch

## Cấu trúc Database

### bank_accounts

- `id`: UUID (Primary key)
- `account_number`: VARCHAR(50) (Unique)
- `account_name`: VARCHAR(255)
- `bank_name`: VARCHAR(100)
- `created_at`: TIMESTAMP

### transactions

- `id`: UUID (Primary key)
- `account_id`: UUID (Foreign key)
- `transaction_type`: VARCHAR(20) ('SEND'|'RECEIVE')
- `from_account`: VARCHAR(50)
- `to_account`: VARCHAR(50)
- `amount`: DECIMAL(15,2)
- `content`: TEXT
- `transaction_date`: TIMESTAMP
- `raw_text`: TEXT
- `created_at`: TIMESTAMP

### ai_extractions

- `id`: UUID (Primary key)
- `transaction_id`: UUID (Foreign key)
- `raw_input`: TEXT
- `extracted_data`: JSONB
- `confidence_score`: DECIMAL(3,2)
- `created_at`: TIMESTAMP

## Mở rộng

- Thêm hỗ trợ nhiều ngân hàng khác nhau
- Phân loại giao dịch tự động
- Xuất báo cáo Excel/PDF
- Tích hợp webhook cho real-time updates
- Mobile app với React Native

## Lưu ý bảo mật

- Không lưu trữ thông tin nhạy cảm như PIN, password
- Mã hóa dữ liệu nhạy cảm trước khi lưu
- Sử dụng HTTPS trong production
- Giới hạn rate limiting cho API

## Support

Nếu gặp vấn đề, vui lòng tạo issue trên GitHub hoặc liên hệ qua email.
