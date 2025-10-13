const axios = require("axios");

class LangExtractService {
    constructor() {
        this.baseURL = "http://localhost:8000";
        this.models = [
            'gemini-2.5-flash', // Online model
            'gemma2:2b', // Local Ollama
            'llama3.2:1b', // Local Ollama
            'llama3.2:3b', // Local Ollama
            'qwen2.5:3b', // Local Ollama
            'phi3:3.8b' // Local Ollama
        ]
    }
    async extractTransactionInfo(transactionText, modelId = 'gemini-2.5-flash') {
        const startTime = Date.now();
        try {
            const response = await axios.post(`${this.baseURL}/extract`, {
                text: transactionText,
                model_id: modelId
            });
            const responseTime = Date.now() - startTime;
            const converted = this.convertLangExtractResponse(response.data.results);

            return {
                model: modelId,
                response_time: responseTime,
                data: converted,
                raw_response: response.data,
                service: 'langextract'
            }
        } catch (error) {
            const responseTime = Date.now() - startTime;
            console.error(`LangExtract Error with model ${modelId}:`, error.message);
            return {
                model: modelId,
                response_time: responseTime,
                error: error.message,
                data: null,
                service: 'langextract'
            };
        }
    }

    convertLangExtractResponse(extractions) {
        const result = {
            account_number: null,
            other_account_number: null,
            other_account_name: null,
            other_bank_name: null,
            amount: null,
            content: null,
            transaction_date: null,
            bank_name: null,
            _extraction_method: 'langextract',
            _confidence: 0.8
        };
        extractions.forEach(extraction => {
            switch (extraction.extraction_class) {
                case 'account_number':
                    result.account_number = extraction.extraction_text;
                    break;
                case 'other_account_number':
                    result.other_account_number = extraction.extraction_text;
                    break;
                case 'other_account_name':
                    result.other_account_name = extraction.extraction_text;
                    break;
                case 'other_bank_name':
                    result.other_bank_name = extraction.extraction_text;
                    break;
                case 'amount':
                    result.amount = extraction.extraction_text;
                    break;
                case 'content':
                    result.content = extraction.extraction_text;
                    break;
                case 'transaction_date':
                    result.transaction_date = extraction.extraction_text;
                    break;
                case 'bank_name':
                    result.bank_name = extraction.extraction_text;
                    break;
            }
        });

        return result;
    }

    async checkConnection() {
        try {
            const response = await axios.get(`${this.baseURL}/health`);
            console.log('✅ LangExtract server is running');
            return true;
        } catch (error) {
            console.log('❌ LangExtract server is not running. Please start: uvicorn app:app --reload');
            return false;
        }
    }
}

module.exports = new LangExtractService();