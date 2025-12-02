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

-- 1. Enable pgvector extension (để lưu embeddings)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Thêm cột embedding vào bảng transactions
-- Lưu vector 1536 chiều từ OpenAI text-embedding-3-small
ALTER TABLE transactions 
ADD COLUMN IF NOT EXISTS embedding vector(1536);

-- 3. Thêm metadata cho AI extraction (nếu chưa có)
ALTER TABLE transactions
ADD COLUMN IF NOT EXISTS extraction_model VARCHAR(50),
ADD COLUMN IF NOT EXISTS extraction_confidence NUMERIC(3, 2),
ADD COLUMN IF NOT EXISTS response_time INTEGER;

-- 4. Tạo vector search index (HNSW = Hierarchical Navigable Small World)
-- Tăng tốc độ tìm kiếm vector similarity lên 100x
CREATE INDEX IF NOT EXISTS idx_transactions_embedding 
ON transactions USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 5. Bảng lưu lịch sử RAG queries
CREATE TABLE IF NOT EXISTS rag_queries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Câu hỏi từ user
    query TEXT NOT NULL,
    query_embedding VECTOR(1536), -- Vector của câu hỏi
    
    -- Kết quả tìm kiếm
    retrieved_transaction_ids UUID[], -- Mảng IDs của transactions tìm được
    num_results INTEGER, -- Số lượng kết quả
    
    -- Câu trả lời từ GPT
    generated_answer TEXT,
    model_used VARCHAR(50), -- 'gpt-4o-mini', 'gpt-4', etc.
    response_time INTEGER, -- Thời gian xử lý (ms)
    
    -- Feedback từ user (optional - dùng để improve model)
    user_rating INTEGER CHECK (user_rating BETWEEN 1 AND 5),
    user_feedback TEXT
);

-- 6. Indexes cho rag_queries
CREATE INDEX IF NOT EXISTS idx_rag_queries_date 
ON rag_queries(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_rag_queries_embedding 
ON rag_queries USING hnsw (query_embedding vector_cosine_ops);

-- 7. Function: Tìm kiếm transactions tương tự bằng vector
CREATE OR REPLACE FUNCTION search_similar_transactions(
    query_embedding VECTOR(1536),
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    raw_text TEXT,
    other_account_number VARCHAR(50),
    other_account_name VARCHAR(255),
    other_bank_name VARCHAR(100),
    amount DECIMAL(15,2),
    content TEXT,
    transaction_date TIMESTAMP WITH TIME ZONE,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        t.id,
        t.raw_text,
        t.other_account_number,
        t.other_account_name,
        t.other_bank_name,
        t.amount,
        t.content,
        t.transaction_date,
        1 - (t.embedding <=> query_embedding) AS similarity
    FROM transactions t
    WHERE t.embedding IS NOT NULL
    ORDER BY t.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- 8. Comments để documentation
COMMENT ON COLUMN transactions.embedding IS 'OpenAI text-embedding-3-small vector (1536 dimensions) for semantic search';
COMMENT ON TABLE rag_queries IS 'Log của RAG queries và responses để tracking và improvement';
COMMENT ON FUNCTION search_similar_transactions IS 'Tìm kiếm transactions bằng cosine similarity trên vector embeddings';

-- 9. Verify setup
DO $$ 
BEGIN
    RAISE NOTICE '==============================================';
    RAISE NOTICE '✅ RAG extensions added successfully!';
    RAISE NOTICE '==============================================';
    RAISE NOTICE 'Tables created:';
    RAISE NOTICE '  - rag_queries';
    RAISE NOTICE 'Columns added to transactions:';
    RAISE NOTICE '  - embedding (vector 1536D)';
    RAISE NOTICE '  - extraction_model';
    RAISE NOTICE '  - extraction_confidence';
    RAISE NOTICE '  - response_time';
    RAISE NOTICE 'Indexes created:';
    RAISE NOTICE '  - idx_transactions_embedding (HNSW)';
    RAISE NOTICE '  - idx_rag_queries_embedding (HNSW)';
    RAISE NOTICE 'Functions created:';
    RAISE NOTICE '  - search_similar_transactions()';
    RAISE NOTICE '==============================================';
END $$;