const langExtractService = require('./langExtract.service');
const fs = require('fs');

class LangExtractBenchmarkService {
    constructor() {
        this.loadTransactionSamples();

        this.models = [{
                name: 'Gemini 2.5 Flash',
                model_id: 'gemini-2.5-flash',
                type: 'online'
            },
            {
                name: 'Gemma2 2B',
                model_id: 'gemma2:2b',
                type: 'local'
            },
            {
                name: 'Llama3.2 1B',
                model_id: 'llama3.2:1b',
                type: 'local'
            },
            {
                name: 'Llama3.2 3B',
                model_id: 'llama3.2:3b',
                type: 'local'
            },
            {
                name: 'Qwen2.5 3B',
                model_id: 'qwen2.5:3b',
                type: 'local'
            },
            {
                name: 'Phi3 3.8B',
                model_id: 'phi3:3.8b',
                type: 'local'
            }
        ];
        this.experimentConfig = {
            iterations: 3,
            delay_between_requests: 1000, // ms
            timeout_per_request: 30000 // ms
        };
    }

    loadTransactionSamples() {
        try {
            if (fs.existsSync('data/transaction-sample.json')) {
                this.testTransactions = JSON.parse(fs.readFileSync('data/transaction-sample.json', 'utf-8'));
                console.log(`📊 Loaded ${this.testTransactions.length} transaction samples`);
            } else {
                console.error('❌ File data/transaction-sample.json not found');
                this.testTransactions = [];
            }
        } catch (error) {
            console.error('❌ Error loading data:', error.message);
            this.testTransactions = [];
        }
    }

    async runBenchmark() {
        console.log('🚀 Starting LangExtract Benchmark...');
        const isConnect = await langExtractService.checkConnection();
        if (!isConnect) {
            throw new Error('LangExtract service is not reachable. Please check your configuration.');
        }

        const results = [];

        // ✅ Sửa: dùng model object, không phải modelId
        for (const model of this.models) {
            console.log(`\n🔬 Testing model: ${model.name} (${model.model_id})`);

            const modelResult = {
                model: model.name,
                model_id: model.model_id,
                model_type: model.type,
                tests: [],
                summary: {}
            };

            const testTransactions = this.testTransactions.slice(0, 3);
            for (let i = 0; i < testTransactions.length; i++) {
                const transaction = testTransactions[i];
                console.log(`  📝 Testing transaction ${i + 1}/${testTransactions.length}`);

                try {
                    // ✅ Sửa: dùng model.model_id
                    const result = await langExtractService.extractTransactionInfo(transaction, model.model_id);
                    const success = !result.error && result.data && result.data.account_number;

                    modelResult.tests.push({
                        transaction_id: i + 1,
                        success: success,
                        response_time: result.response_time,
                        error: result.error || null,
                        extracted_fields: success ? Object.keys(result.data).filter(k => result.data[k] !== null).length : 0
                    });

                } catch (error) {
                    modelResult.tests.push({
                        transaction_id: i + 1,
                        success: false,
                        response_time: 0,
                        error: error.message,
                        extracted_fields: 0
                    });
                }

                await new Promise(resolve => setTimeout(resolve, 1000));
            }

            this.calculateSummary(modelResult);
            results.push(modelResult);
        }

        this.displayResults(results);
        this.saveResults(results);
        return results;
    }

    calculateSummary(modelResult) {
        const tests = modelResult.tests;
        const successful = tests.filter(t => t.success);

        modelResult.summary = {
            total_tests: tests.length,
            successful: successful.length,
            success_rate: ((successful.length / tests.length) * 100).toFixed(1),
            avg_response_time: Math.round(tests.reduce((sum, t) => sum + t.response_time, 0) / tests.length),
            avg_extracted_fields: Math.round(tests.reduce((sum, t) => sum + t.extracted_fields, 0) / tests.length)
        };
    }

    displayResults(results) {
        console.log('\n' + '='.repeat(80));
        console.log('📊 LANGEXTRACT BENCHMARK RESULTS');
        console.log('='.repeat(80));

        // Tạo bảng đơn giản
        const table = results.map(r => ({
            'Model': r.model,
            'Success Rate': r.summary.success_rate + '%',
            'Tests': `${r.summary.successful}/${r.summary.total_tests}`,
            'Avg Time (ms)': r.summary.avg_response_time,
            'Avg Fields': r.summary.avg_extracted_fields
        }));

        console.table(table);

        const bestModel = results.sort((a, b) => parseFloat(b.summary.success_rate) - parseFloat(a.summary.success_rate))[0];
        console.log(`\n🏆 Best Model: ${bestModel.model} (${bestModel.summary.success_rate}% success rate)`);

        const fastestModel = results.sort((a, b) => a.summary.avg_response_time - b.summary.avg_response_time)[0];
        console.log(`⚡ Fastest Model: ${fastestModel.model} (${fastestModel.summary.avg_response_time}ms avg)`);
    }

    saveResults(results) {
        if (!fs.existsSync('benchmark_results')) {
            fs.mkdirSync('benchmark_results');
        }

        const filename = `benchmark_results/langextract_${Date.now()}.json`;
        fs.writeFileSync(filename, JSON.stringify(results, null, 2));
        console.log(`\n💾 Results saved to: ${filename}`);
    }
}

module.exports = new LangExtractBenchmarkService();