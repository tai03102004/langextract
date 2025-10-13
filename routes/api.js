const express = require('express');
const router = express.Router();
const transactionController = require('../controllers/transactionController');

// POST /api/transactions/process - Xử lý văn bản giao dịch
router.post('/transactions/process', transactionController.processTransaction);

// GET /api/transactions - Lấy danh sách giao dịch
router.get('/transactions', transactionController.getTransactions);

// GET /api/transactions/stats - Lấy thống kê giao dịch
router.get('/transactions/stats', transactionController.getTransactionStats);

module.exports = router;