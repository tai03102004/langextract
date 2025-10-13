-- Bảng lưu thông tin tài khoản ngân hàng
CREATE TABLE bank_accounts (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  account_number VARCHAR(50) UNIQUE NOT NULL,
  account_name VARCHAR(255),
  bank_name VARCHAR(100),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Bảng lưu thông tin giao dịch
CREATE TABLE transactions (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  account_id UUID REFERENCES bank_accounts(id),
  other_account_number VARCHAR(50) NOT NULL,-- Số tài khoản đối tác
  other_account_name VARCHAR(255), -- Tên tài khoản đối tác
  other_bank_name VARCHAR(100), -- Tên ngân hàng đối tác
  amount DECIMAL(15,2) NOT NULL, -- Số tiền (âm = chuyển đi, dương = nhận vào)
  content TEXT, -- Nội dung giao dịch
  transaction_date TIMESTAMP WITH TIME ZONE,
  raw_text TEXT, -- Dữ liệu thô từ sao kê
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Bảng lưu lịch sử phân tích AI
CREATE TABLE ai_extractions (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  transaction_id UUID REFERENCES transactions(id),
  raw_input TEXT NOT NULL,
  extracted_data JSONB,
  confidence_score DECIMAL(3,2),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index để tối ưu truy vấn
CREATE INDEX idx_transactions_account_id ON transactions(account_id);
CREATE INDEX idx_transactions_date ON transactions(transaction_date);
CREATE INDEX idx_transactions_amount ON transactions(amount);
CREATE INDEX idx_other_amount ON transactions(other_account_number);
