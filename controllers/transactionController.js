const geminiService = require('../services/gemini.service');
const supabaseService = require('../services/supabase.service');
const langextractService = require("../services/langExtract.service");

class TransactionController {
    // async processTransaction(req, res) {
    //     try {
    //         const {
    //             transactionText
    //         } = req.body;

    //         if (!transactionText) {
    //             return res.status(400).json({
    //                 error: 'Vui lòng cung cấp nội dung giao dịch'
    //             });
    //         }

    //         // Bước 1: Trích xuất thông tin bằng AI
    //         console.log('Extracting transaction info...');
    //         const extractedData = await geminiService.extractTransactionInfo(transactionText);

    //         // Bước 2: Validate và cải thiện dữ liệu
    //         const validatedData = await geminiService.validateAndEnhance(extractedData, transactionText);

    //         // Bước 3: Tìm hoặc tạo tài khoản ngân hàng
    //         const bankAccount = await supabaseService.findOrCreateBankAccount(
    //             validatedData.account_number,
    //             'Unknown', // Account name sẽ được cập nhật sau
    //             validatedData.bank_name
    //         );

    //         // Bước 4: Lưu giao dịch
    //         const transactionRecord = await supabaseService.saveTransaction({
    //                 ...validatedData,
    //                 raw_text: transactionText
    //             },
    //             bankAccount.id
    //         );

    //         // Bước 5: Lưu lịch sử AI extraction
    //         await supabaseService.saveAIExtraction(
    //             transactionRecord.id,
    //             transactionText,
    //             validatedData,
    //             validatedData.confidence || 0.8
    //         );

    //         res.json({
    //             success: true,
    //             message: 'Giao dịch đã được xử lý thành công',
    //             data: {
    //                 transaction: transactionRecord,
    //                 extracted_info: validatedData
    //             }
    //         });

    //     } catch (error) {
    //         console.error('Error processing transaction:', error);
    //         res.status(500).json({
    //             error: 'Lỗi xử lý giao dịch',
    //             details: error.message
    //         });
    //     }
    // }
    async processTransaction(req, res) {
        try {
            const {
                transactionText
            } = req.body;

            if (!transactionText) {
                return res.status(400).json({
                    error: 'Vui lòng cung cấp nội dung giao dịch'
                });
            }

            const extractedData = await langextractService.extractTransactionInfo(transactionText);
            console.log("Extracted Data:", extractedData);


            // // Bước 1: Trích xuất thông tin bằng AI
            // console.log('Extracting transaction info...');
            // const extractedData = await langextractService.extractTransactionInfo(transactionText);

            // // Bước 2: Validate và cải thiện dữ liệu
            // const validatedData = await langextractService.validateAndEnhance(extractedData, transactionText);

            // // Bước 3: Tìm hoặc tạo tài khoản ngân hàng
            // const bankAccount = await supabaseService.findOrCreateBankAccount(
            //     validatedData.account_number,
            //     'Unknown', // Account name sẽ được cập nhật sau
            //     validatedData.bank_name
            // );

            // // Bước 4: Lưu giao dịch
            // const transactionRecord = await supabaseService.saveTransaction({
            //         ...validatedData,
            //         raw_text: transactionText
            //     },
            //     bankAccount.id
            // );

            // // Bước 5: Lưu lịch sử AI extraction
            // await supabaseService.saveAIExtraction(
            //     transactionRecord.id,
            //     transactionText,
            //     validatedData,
            //     validatedData.confidence || 0.8
            // );

            // res.json({
            //     success: true,
            //     message: 'Giao dịch đã được xử lý thành công',
            //     data: {
            //         transaction: transactionRecord,
            //         extracted_info: validatedData
            //     }
            // });

        } catch (error) {
            console.error('Error processing transaction:', error);
            res.status(500).json({
                error: 'Lỗi xử lý giao dịch',
                details: error.message
            });
        }
    }

    async getTransactions(req, res) {
        try {
            const {
                account_number,
                limit
            } = req.query;

            let transactions;
            if (account_number) {
                transactions = await supabaseService.getTransactionsByAccount(
                    account_number,
                    parseInt(limit) || 50
                );
            } else {
                transactions = await supabaseService.getAllTransactions(
                    parseInt(limit) || 100
                );
            }

            res.json({
                success: true,
                data: transactions
            });

        } catch (error) {
            console.error('Error getting transactions:', error);
            res.status(500).json({
                error: 'Lỗi lấy danh sách giao dịch',
                details: error.message
            });
        }
    }

    async getTransactionStats(req, res) {
        try {
            // Lấy thống kê cơ bản
            const allTransactions = await supabaseService.getAllTransactions(1000);

            const stats = {
                total_transactions: allTransactions.length,
                total_sent: allTransactions.filter(t => t.transaction_type === 'SEND').length,
                total_received: allTransactions.filter(t => t.transaction_type === 'RECEIVE').length,
                total_amount_sent: allTransactions
                    .filter(t => t.transaction_type === 'SEND')
                    .reduce((sum, t) => sum + parseFloat(t.amount), 0),
                total_amount_received: allTransactions
                    .filter(t => t.transaction_type === 'RECEIVE')
                    .reduce((sum, t) => sum + parseFloat(t.amount), 0)
            };

            res.json({
                success: true,
                data: stats
            });

        } catch (error) {
            console.error('Error getting stats:', error);
            res.status(500).json({
                error: 'Lỗi lấy thống kê',
                details: error.message
            });
        }
    }
}

module.exports = new TransactionController();