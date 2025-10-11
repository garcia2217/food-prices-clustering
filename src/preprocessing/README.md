# Preprocessing Module

This module provides modular components for preprocessing raw food price data from Excel files into a consolidated format suitable for analysis.

## Components

### 1. Configuration (`config.py`)

-   `ConsolidationConfig`: Pydantic model for type-safe configuration
-   Validates parameters like provinces, years, commodities
-   Supports auto-population of empty lists from available data

### 2. Data Loader (`data_loader.py`)

-   `DataLoader`: Handles file discovery and loading
-   Discovers Excel files matching configuration criteria
-   Loads and transforms data from wide to long format
-   Adds metadata columns (city, year, province)

### 3. Data Cleaner (`cleaner.py`)

-   `DataCleaner`: Handles data cleaning operations
-   Missing value imputation using forward/backward fill
-   Data type conversion (dates, prices, categories)
-   Outlier removal (IQR or Z-score methods)
-   Date range validation

### 4. Data Consolidator (`consolidator.py`)

-   `DataConsolidator`: Orchestrates the complete pipeline
-   Combines loading and cleaning operations
-   Handles logging setup and error management
-   Exports results to multiple formats (CSV, Excel, JSON metadata)

## Usage

### Basic Usage

```python
from src.preprocessing import ConsolidationConfig, DataConsolidator
from pathlib import Path

# Create configuration
config = ConsolidationConfig(
    enable_file_logging=True,
    log_level="INFO",
    provinces=[],  # Empty = process all
    years=[],      # Empty = process all
    commodities=[  # Or use default 10 commodities
        "Beras", "Telur Ayam", "Daging Ayam", "Daging Sapi",
        "Bawang Merah", "Bawang Putih", "Cabai Merah", "Cabai Rawit",
        "Minyak Goreng", "Gula Pasir"
    ]
)

# Run pipeline
consolidator = DataConsolidator(config)
results = consolidator.run_full_pipeline(
    raw_data_path=Path("data/raw"),
    output_dir=Path("data/processed"),
    remove_outliers=False
)

if results['success']:
    print(f"✅ Processed {results['validation_results']['total_rows']} records")
    consolidated_df = results['consolidated_df']
else:
    print(f"❌ Failed: {results['error']}")
```

### Advanced Usage

```python
# Custom configuration
config = ConsolidationConfig(
    provinces=["Jawa Barat", "DKI Jakarta"],  # Specific provinces
    years=["2022", "2023", "2024"],           # Recent years only
    commodities=["Beras", "Minyak Goreng"],   # Specific commodities
    log_level="DEBUG"
)

# Step-by-step processing
consolidator = DataConsolidator(config)

# 1. Load raw data
raw_data_path = Path("data/raw")
consolidated_df = consolidator.consolidate_data(raw_data_path, remove_outliers=True)

# 2. Export to custom location
export_paths = consolidator.export_consolidated_data(
    consolidated_df,
    output_dir=Path("custom/output/path")
)
```

### Individual Components

```python
from src.preprocessing import DataLoader, DataCleaner, ConsolidationConfig

config = ConsolidationConfig()

# Use data loader independently
loader = DataLoader(config)
files = loader.discover_data_files(Path("data/raw"))
df = loader.load_and_transform_file(files[0])

# Use data cleaner independently
cleaner = DataCleaner(config)
cleaned_df = cleaner.clean_dataframe(df, remove_outliers=True)
```

## Configuration Options

### Data Selection

-   `provinces`: List of province names to process (empty = all)
-   `years`: List of years to process (empty = all available)
-   `commodities`: List of commodities to include (default = all 10)

### Processing Options

-   `columns_to_drop`: Column names to remove from raw data
-   `missing_value_indicators`: Values to treat as missing/null
-   `date_format`: Expected date format in Excel files
-   `commodity_column_name`: Name of commodity column in Excel

### Logging Options

-   `enable_file_logging`: Save logs to file (default: False)
-   `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Output Format

The consolidated DataFrame has the following structure:

| Column    | Type     | Description          |
| --------- | -------- | -------------------- |
| Commodity | category | Food commodity name  |
| City      | category | City name            |
| Year      | category | Year of data         |
| Province  | category | Province name        |
| Date      | datetime | Date of price record |
| Price     | float64  | Price in IDR         |

## Error Handling

The module includes comprehensive error handling:

-   File loading errors are logged and skipped
-   Data validation errors are reported
-   Configuration validation prevents invalid setups
-   Quality issues are detected and reported

## Testing

Run tests with:

```bash
python -m pytest tests/test_preprocessing.py -v
```

## Dependencies

-   pandas: Data manipulation
-   numpy: Numerical operations
-   pydantic: Configuration validation
-   openpyxl: Excel file reading
-   pathlib: Path operations
