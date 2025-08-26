# Performance Benchmarking Guide with HTML Reporting

This guide explains how to run performance benchmarks for predict-otron-9000 and generate HTML reports for easy visualization and analysis.

## Overview

The predict-otron-9000 system consists of three main components:

1. **predict-otron-9000**: The main server that integrates the other components
2. **embeddings-engine**: Generates text embeddings using the Nomic Embed Text v1.5 model
3. **inference-engine**: Handles text generation using various Gemma models

We have two benchmark scripts that test these components under different conditions:
- `performance_test_embeddings.sh`: Tests embedding generation with different input sizes
- `performance_test_inference.sh`: Tests text generation with different prompt sizes

This guide extends the existing benchmarking functionality by adding HTML report generation for better visualization and sharing of results.

## Prerequisites

- Rust 1.70+ with 2024 edition support
- Cargo package manager
- Node.js 16+ (for HTML report generation)
- Basic understanding of the system architecture
- The project built with `cargo build --release`

## Step 1: Installing Required Tools

First, you'll need to install the necessary tools for HTML report generation:

```bash
# Install Chart.js for visualizations
npm install -g chart.js

# Install a simple HTTP server to view reports locally
npm install -g http-server
```

## Step 2: Running Performance Tests

The benchmarking process has two phases: running the tests and generating HTML reports from the results.

### Start the Server

```bash
# Start the server in a terminal window
./run_server.sh
```

Wait for the server to fully initialize (look for "server listening" message).

### Run Embedding Performance Tests

In a new terminal window:

```bash
# Run the embeddings performance test
./performance_test_embeddings.sh
```

Note the temporary directory path where results are stored. You'll need this for the HTML generation.

### Run Inference Performance Tests

```bash
# Run the inference performance test
./performance_test_inference.sh
```

Again, note the temporary directory path where results are stored.

## Step 3: Generating HTML Reports

Now you'll convert the test results into HTML reports. Use the script below to transform the benchmark data.

Create a file named `generate_benchmark_report.sh` in the project root:

```bash
#!/bin/bash

# Create a new benchmark report script
cat > generate_benchmark_report.sh << 'EOF'
#!/bin/bash

# Script to generate HTML performance reports from benchmark results

# Check if results directory was provided
if [ -z "$1" ]; then
    echo "Error: Please provide the directory containing benchmark results."
    echo "Usage: $0 /path/to/results/directory"
    exit 1
fi

RESULTS_DIR="$1"
OUTPUT_DIR="benchmark_reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_DIR="${OUTPUT_DIR}/${TIMESTAMP}"

# Create output directories
mkdir -p "${REPORT_DIR}"

# Function to extract data from results files
extract_data() {
    local test_type="$1"
    local data_file="${REPORT_DIR}/${test_type}_data.js"
    
    echo "// ${test_type} benchmark data" > "$data_file"
    echo "const ${test_type}Labels = [];" >> "$data_file"
    echo "const ${test_type}Times = [];" >> "$data_file"
    
    # Find all result files for this test type
    for result_file in "${RESULTS_DIR}"/*_results.txt; do
        if [ -f "$result_file" ]; then
            # Extract test size/name
            size=$(basename "$result_file" | sed 's/_results.txt//')
            
            # Extract average time
            avg_time=$(grep "Average time for $size" "$result_file" | awk '{print $6}')
            
            if [ -n "$avg_time" ]; then
                echo "${test_type}Labels.push('$size');" >> "$data_file"
                echo "${test_type}Times.push($avg_time);" >> "$data_file"
            fi
        fi
    done
}

# Generate the HTML report
create_html_report() {
    local html_file="${REPORT_DIR}/index.html"
    
    cat > "$html_file" << HTML
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>predict-otron-9000 Performance Benchmark Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .report-header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #eee;
        }
        .chart-container {
            margin: 30px 0;
            height: 400px;
        }
        .metrics-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            flex: 1;
            min-width: 250px;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            background-color: #f9f9f9;
        }
        .raw-data {
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: monospace;
            white-space: pre;
            margin-top: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
    </style>
</head>
<body>
    <div class="report-header">
        <h1>predict-otron-9000 Performance Benchmark Report</h1>
        <p>Generated on: $(date)</p>
    </div>
    
    <h2>Summary</h2>
    <p>
        This report shows performance benchmarks for the predict-otron-9000 system,
        measuring both embedding generation and text inference capabilities across
        different input sizes.
    </p>
    
    <div class="metrics-container">
        <div class="metric-card">
            <h3>Embeddings Performance</h3>
            <p>Average response times for generating embeddings with different input sizes.</p>
        </div>
        <div class="metric-card">
            <h3>Inference Performance</h3>
            <p>Average response times for text generation with different prompt sizes.</p>
        </div>
    </div>
    
    <h2>Embeddings Engine Performance</h2>
    <div class="chart-container">
        <canvas id="embeddingsChart"></canvas>
    </div>
    
    <h2>Inference Engine Performance</h2>
    <div class="chart-container">
        <canvas id="inferenceChart"></canvas>
    </div>
    
    <h2>Detailed Results</h2>
    
    <h3>Embeddings Performance by Input Size</h3>
    <table id="embeddingsTable">
        <tr>
            <th>Input Size</th>
            <th>Average Response Time (s)</th>
        </tr>
        <!-- Table will be populated by JavaScript -->
    </table>
    
    <h3>Inference Performance by Prompt Size</h3>
    <table id="inferenceTable">
        <tr>
            <th>Prompt Size</th>
            <th>Average Response Time (s)</th>
        </tr>
        <!-- Table will be populated by JavaScript -->
    </table>
    
    <h2>System Information</h2>
    <div class="metrics-container">
        <div class="metric-card">
            <h3>Hardware</h3>
            <p>$(uname -s) $(uname -m)</p>
            <p>CPU: $(grep 'model name' /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2 || sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")</p>
        </div>
        <div class="metric-card">
            <h3>Software</h3>
            <p>Rust Version: $(rustc --version)</p>
            <p>predict-otron-9000 Version: $(grep 'version' Cargo.toml | head -1 | cut -d'"' -f2 || echo "Unknown")</p>
        </div>
    </div>
    
    <script src="embeddings_data.js"></script>
    <script src="inference_data.js"></script>
    <script>
        // Embeddings Chart
        const embeddingsCtx = document.getElementById('embeddingsChart').getContext('2d');
        new Chart(embeddingsCtx, {
            type: 'bar',
            data: {
                labels: embeddingsLabels,
                datasets: [{
                    label: 'Average Response Time (s)',
                    data: embeddingsTimes,
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Time (seconds)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Input Size'
                        }
                    }
                }
            }
        });
        
        // Inference Chart
        const inferenceCtx = document.getElementById('inferenceChart').getContext('2d');
        new Chart(inferenceCtx, {
            type: 'bar',
            data: {
                labels: inferenceLabels,
                datasets: [{
                    label: 'Average Response Time (s)',
                    data: inferenceTimes,
                    backgroundColor: 'rgba(255, 99, 132, 0.5)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Time (seconds)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Prompt Size'
                        }
                    }
                }
            }
        });
        
        // Populate tables
        function populateTable(tableId, labels, times) {
            const table = document.getElementById(tableId);
            for (let i = 0; i < labels.length; i++) {
                const row = table.insertRow(-1);
                const sizeCell = row.insertCell(0);
                const timeCell = row.insertCell(1);
                sizeCell.textContent = labels[i];
                timeCell.textContent = times[i].toFixed(3);
            }
        }
        
        // Populate tables when page loads
        window.onload = function() {
            populateTable('embeddingsTable', embeddingsLabels, embeddingsTimes);
            populateTable('inferenceTable', inferenceLabels, inferenceTimes);
        };
    </script>
</body>
</html>
HTML

    echo "Created HTML report at: ${html_file}"
}

# Extract data for each test type
echo "Extracting embeddings benchmark data..."
extract_data "embeddings"

echo "Extracting inference benchmark data..."
extract_data "inference"

# Create the HTML report
echo "Generating HTML report..."
create_html_report

echo "Benchmark report generated successfully!"
echo "Open the report with: http-server ${REPORT_DIR} -o"
EOF

# Make the script executable
chmod +x generate_benchmark_report.sh
```

After creating this script, make it executable:

```bash
chmod +x generate_benchmark_report.sh
```

## Step 4: Using the Report Generator

After running the benchmark tests, use the newly created script to generate an HTML report:

```bash
# Generate HTML report from test results
./generate_benchmark_report.sh /path/to/results/directory
```

Replace `/path/to/results/directory` with the temporary directory path that was output by the benchmark scripts.

## Step 5: Viewing the Report

After generating the report, you can view it in your browser:

```bash
# Start a local web server to view the report
cd benchmark_reports/<timestamp>
http-server -o
```

This will open your default browser and display the HTML benchmark report.

## HTML Report Features

The generated HTML report includes:

1. **Summary overview** of all benchmark results
2. **Interactive charts** visualizing performance across different input sizes
3. **Detailed tables** with exact timing measurements
4. **System information** to provide context for the benchmark results
5. **Raw data** available for further analysis

## Customizing Benchmarks

You can customize the benchmark tests by modifying the existing script parameters:

### Embeddings Benchmark Customization

Edit `performance_test_embeddings.sh` to change:
- Number of iterations
- Test input sizes
- Server URL/port

### Inference Benchmark Customization

Edit `performance_test_inference.sh` to change:
- Number of iterations
- Test prompt sizes
- Maximum token generation
- Model selection

## Interpreting Results

When analyzing the benchmark results, consider:

1. **Response Time Scaling**: How does performance scale with input size?
2. **Consistency**: Are response times consistent across iterations?
3. **Hardware Utilization**: Check CPU/memory usage during tests
4. **Bottlenecks**: Identify which operations take the most time

## Sharing Results

The HTML reports are self-contained and can be shared with team members by:
- Copying the benchmark_reports directory
- Hosting the report on an internal web server
- Converting to PDF if needed

## Troubleshooting

If you encounter issues:

1. **Empty reports**: Ensure the benchmark tests completed successfully
2. **Missing charts**: Check for JavaScript errors in the browser console
3. **Script errors**: Verify Node.js and required packages are installed

## Conclusion

Regular performance benchmarking helps track system performance over time, identify regressions, and measure the impact of optimizations. By generating HTML reports, you can more easily visualize and share performance data with your team.

For more detailed performance analysis, see [PERFORMANCE.md](PERFORMANCE.md) and [OPTIMIZATIONS.md](OPTIMIZATIONS.md).