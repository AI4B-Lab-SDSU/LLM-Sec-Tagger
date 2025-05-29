# LLM-Driven Document Clustering: Improving Real-Time Security Intelligence Extraction and Threat Analysis

This project compares two different approaches for multi-label document classification and topic modeling across multiple academic paper datasets. Developed as part of research at San Diego State University (SDSU), it evaluates:

1. **LLM-based approach** using GPT-4o for keyword extraction and label generation
2. **Traditional approach** using BERTopic with sentence transformers

The system supports analysis of multiple datasets including APT(Advanced Persistent Threat) conference papers and other academic collections.

## Project Structure

```
LLMTaggerSDSU-main/
├── apt_2023_data/           # Additional APT 2023 dataset
├── DS1_Paper/               # Dataset 1: Paper collection
│   ├── pdfs/               # PDF documents for Dataset 1: MLSCHOLAR
│   └── comparison_results/ # DS1 comparison outputs
├── DS2_APT/                # Dataset 2: APT 2015 collection
│   ├── APT2015/            # PDF documents for DS2
│   └── comparison_results/ # DS2 comparison outputs
├── DS3_APT2/               # Dataset 3: Additional APT collection
│   ├── APT/                # PDF documents for DS3
│   └── comparison_results/ # DS3 comparison outputs
├── logs/                   # Application logs
├── nltk/                   # NLTK data (stopwords)
├── old/                    # Archived/old versions
├── Results/                # General results directory
├── test_pdf/               # Test PDF files
├── script1-llm.py         # LLM: Extract keywords from PDFs
├── script2-llm.py         # LLM: Generate labels from keywords
├── script4_LLM_multi-label_metrics.py  # LLM: Evaluate results
├── 1-traditional.py       # Traditional: BERTopic clustering
├── 3-traditional_metrics_multilabel.py # Traditional: Evaluate results
└── compare.py             # Compare both approaches
```

## Workflow Overview

The project supports analysis of multiple datasets (DS1_Paper, DS2_APT, DS3_APT2) using the same pipeline:

### LLM Pipeline

1. **Keyword Extraction** (`script1-llm.py`) → `llm_1_extracted_keywords.json`
2. **Label Generation** (`script2-llm.py`) → `llm_2_generated_labels.json`
3. **Evaluation** (`script4_LLM_multi-label_metrics.py`) → `llm_3_evaluation_results.json`

### Traditional Pipeline

1. **Multi-label Clustering** (`1-traditional.py`) → `trad-1-multilabel_bertopic_results.json`
2. **Evaluation** (`3-traditional_metrics_multilabel.py`) → `trad-2-evaluation_results.json`

### Comparison

- **Final Comparison** (`compare.py`) → HTML report and visualizations in `{dataset}/comparison_results/`

**Note**: Configure the dataset paths in each script's `main()` function to switch between DS1, DS2, or DS3.

## Prerequisites

### Python Dependencies

```bash
pip install -R Requirements.txt
```

### Environment Setup

Important!!!
Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

## Usage Instructions

### Option 1: Run Complete Comparison

```bash
# Run LLM pipeline
python script1-llm.py
python script2-llm.py
python script4_LLM_multi-label_metrics.py

# Run traditional pipeline
python 1-traditional.py
python 3-traditional_metrics_multilabel.py

# Generate comparison
python compare.py
```

### Option 2: Run Individual Components

#### LLM Approach

```bash
# Step 1: Extract keywords using GPT-4
python script1-llm.py

# Step 2: Generate labels from keywords
python script2-llm.py

# Step 3: Evaluate LLM results
python script4_LLM_multi-label_metrics.py
```

#### Traditional Approach

```bash
# Step 1: Multi-label topic modeling with BERTopic
python 1-traditional.py

# Step 2: Evaluate traditional results
python 3-traditional_metrics_multilabel.py
```

#### Comparison

```bash
# Generate comparison report and visualizations
python compare.py
```

## Configuration Options

### Dataset Selection

Update the paths in each script's `main()` function to select your dataset:

#### For DS1_Paper dataset:

```python
FOLDER_PATH = r'DS1_Paper/pdfs'
OUTPUT_FILE = r'DS1_Paper/llm_1_extracted_keywords.json'
```

#### For DS2_APT dataset (default in provided code):

```python
FOLDER_PATH = r'DS2_APT/APT2015'
OUTPUT_FILE = r'DS2_APT/llm_1_extracted_keywords.json'
```

#### For DS3_APT2 dataset:

```python
FOLDER_PATH = r'DS3_APT2/APT'
OUTPUT_FILE = r'DS3_APT2/llm_1_extracted_keywords.json'
```

### LLM Settings (`script1-llm.py`, `script2-llm.py`)

- `MODEL`: OpenAI model to use (default: `gpt-4o`)
- `FOLDER_PATH`: Path to PDF documents (update for your chosen dataset)
- `OUTPUT_FILE`: Path for results (update for your chosen dataset)

### Traditional Settings (`1-traditional.py`)

- `MODEL_NAME`: Sentence transformer model (default: `all-MiniLM-L6-v2`)
- `N_CLUSTERS`: Number of clusters for KMeans
- `SIMILARITY_THRESHOLD`: Threshold for multi-label assignment (default: 0.3)
- `MAX_LABELS`: Maximum labels per document (default: 3)
- `PDF_FOLDER`: Update to match your chosen dataset
- `OUTPUT_JSON`: Update to match your chosen dataset

### Evaluation Settings

Both evaluation scripts use the same metrics but require path updates:

- Update `LABELS_JSON_PATH`, `PDF_FOLDER`, and `EVAL_OUTPUT_JSON` to match your dataset
- Label density (semantic coherence within labels)
- Topic coherence (C_V, C_UCI, C_NPMI)
- Overlap quality (how well multi-label assignments work)
- Topic distinctiveness (Jensen-Shannon divergence)
- Label balance (distribution entropy)

## Output Files

### Generated Data Files (per dataset)

Files are generated in the respective dataset directory (DS1_Paper/, DS2_APT/, or DS3_APT2/):

- `llm_1_extracted_keywords.json`: Keywords extracted by LLM per document
- `llm_2_generated_labels.json`: Labels and document assignments from LLM
- `llm_3_evaluation_results.json`: Evaluation metrics for LLM approach
- `trad-1-multilabel_bertopic_results.json`: BERTopic clustering results
- `trad-2-evaluation_results.json`: Evaluation metrics for traditional approach

### Comparison Results

Generated in `{dataset}/comparison_results/`:

- `model_comparison_report.html`: Detailed HTML comparison report
- `metrics_comparison.png`: Bar chart comparing key metrics
- `radar_comparison.png`: Radar chart of normalized metrics
- `wins_pie_chart.png`: Distribution of which model performs better

### Additional Outputs

- `logs/`: Detailed execution logs with timestamps
- `Results/`: General results archive
- Individual dataset metadata files (when generated)

## Evaluation Metrics

The project evaluates both approaches using:

1. **Overall Density**: Average semantic similarity within labels
2. **Topic Coherence**: Linguistic coherence of topic words (C_V, C_UCI, C_NPMI)
3. **Overlap Quality Index**: How well documents sharing labels are similar
4. **Topic Distinctiveness**: Jensen-Shannon divergence between topics
5. **Label Entropy**: Balance of label size distribution
6. **Average Labels per Document**: Multi-label assignment statistics

## Understanding Results

### High-Quality Results Indicators

- **High density**: Documents within labels are semantically similar
- **High coherence**: Topic words make linguistic sense together
- **High overlap quality**: Multi-label assignments are meaningful
- **High distinctiveness**: Topics are well-separated
- **Balanced entropy**: Labels are reasonably sized

### Comparison Output

The final HTML report includes:

- Executive summary with winner determination
- Detailed metric comparisons
- Visualizations (bar charts, radar plots, pie charts)
- Top labels and representative words for each approach
- Recommendations based on performance

## Troubleshooting

### Common Issues

1. **OpenAI API errors**: Check API key in `.env` file and account credits
2. **PDF extraction failures**: Ensure PDFs contain extractable text
3. **NLTK download errors**: Run `nltk.download('stopwords')` manually (downloads to `nltk/` directory)

### Log Files

All scripts generate detailed logs in the `logs/` directory for debugging.

### File Path Issues

Update file paths in each script's main() function to match your chosen dataset:

- `FOLDER_PATH` / `PDF_FOLDER`: Location of PDF documents (DS1_Paper/pdfs, DS2_APT/APT2015, or DS3_APT2/APT)
- `OUTPUT_FILE` / `*_JSON_PATH`: Output file locations (update dataset prefix)
- `OUTPUT_DIR`: Comparison results directory (update to match dataset)

## Customization

### Adding New Evaluation Metrics

Extend the evaluation scripts by adding functions to compute additional metrics and including them in the results dictionary.

### Changing LLM Prompts

Modify the prompts in `script1-llm.py` and `script2-llm.py` to adjust keyword extraction and label generation behavior.

### Adjusting Traditional Model

Modify clustering parameters in `1-traditional.py`:

- Change sentence transformer model
- Adjust clustering algorithms or parameters
- Modify multi-label assignment logic

## Research Applications

This framework is suitable for:

- Academic paper classification
- Classification of Security Documents
- Document organization systems
- Topic modeling research
- Comparing ML approaches
- Multi-label classification evaluation
