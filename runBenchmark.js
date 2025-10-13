require('dotenv').config();
const langExtractBenchmark = require('./services/langExtractBenchmark.service');

async function main() {
    console.log('🚀 LANGEXTRACT SIMPLE BENCHMARK');
    console.log('================================');
    console.log('Testing LangExtract framework with different models');
    console.log('Vietnamese banking transaction extraction');

    try {
        const results = await langExtractBenchmark.runBenchmark();

        if (results) {
            console.log('\n✅ Benchmark completed!');
            console.log('📁 Check benchmark_results/ folder for detailed data');
        }

    } catch (error) {
        console.error('❌ Benchmark failed:', error.message);
    }
}

main().catch(console.error);