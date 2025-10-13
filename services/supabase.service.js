const {
    createClient
} = require('@supabase/supabase-js');

class SupabaseService {
    constructor() {
        this.supabase = createClient(
            process.env.SUPABASE_URL,
            process.env.SUPABASE_ANON_KEY
        );
    }

    async findOrCreateBankAccount(accountNumber, accountName, bankName) {
        try {
            // Tìm tài khoản đã tồn tại
            let {
                data: existingAccount
            } = await this.supabase
                .from('bank_accounts')
                .select('*')
                .eq('account_number', accountNumber)
                .single();

            if (existingAccount) {
                return existingAccount;
            }

            // Tạo tài khoản mới
            const {
                data: newAccount,
                error
            } = await this.supabase
                .from('bank_accounts')
                .insert([{
                    account_number: accountNumber,
                    account_name: accountName,
                    bank_name: bankName
                }])
                .select()
                .single();

            if (error) throw error;
            return newAccount;
        } catch (error) {
            console.error('Error with bank account:', error);
            throw error;
        }
    }

    async saveTransaction(transactionData, accountId) {
        try {
            const {
                data,
                error
            } = await this.supabase
                .from('transactions')
                .insert([{
                    account_id: accountId,
                    transaction_type: transactionData.transaction_type,
                    from_account: transactionData.from_account,
                    to_account: transactionData.to_account,
                    amount: parseFloat(transactionData.amount),
                    content: transactionData.content,
                    transaction_date: transactionData.transaction_date,
                    raw_text: transactionData.raw_text
                }])
                .select()
                .single();

            if (error) throw error;
            return data;
        } catch (error) {
            console.error('Error saving transaction:', error);
            throw error;
        }
    }

    async saveAIExtraction(transactionId, rawInput, extractedData, confidence) {
        try {
            const {
                data,
                error
            } = await this.supabase
                .from('ai_extractions')
                .insert([{
                    transaction_id: transactionId,
                    raw_input: rawInput,
                    extracted_data: extractedData,
                    confidence_score: confidence
                }])
                .select()
                .single();

            if (error) throw error;
            return data;
        } catch (error) {
            console.error('Error saving AI extraction:', error);
            throw error;
        }
    }

    async getTransactionsByAccount(accountNumber, limit = 50) {
        try {
            const {
                data,
                error
            } = await this.supabase
                .from('transactions')
                .select(`
                    *,
                    bank_accounts!inner(account_number, account_name, bank_name)
                `)
                .eq('bank_accounts.account_number', accountNumber)
                .order('transaction_date', {
                    ascending: false
                })
                .limit(limit);

            if (error) throw error;
            return data;
        } catch (error) {
            console.error('Error getting transactions:', error);
            throw error;
        }
    }

    async getAllTransactions(limit = 100) {
        try {
            const {
                data,
                error
            } = await this.supabase
                .from('transactions')
                .select(`
                    *,
                    bank_accounts(account_number, account_name, bank_name)
                `)
                .order('created_at', {
                    ascending: false
                })
                .limit(limit);

            if (error) throw error;
            return data;
        } catch (error) {
            console.error('Error getting all transactions:', error);
            throw error;
        }
    }
}

module.exports = new SupabaseService();