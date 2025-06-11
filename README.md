# Retail Sales Analysis & Forecast

This Streamlit application helps analyze retail sales data and predict future sales using machine learning.

## Features

- Data upload and analysis
- Sales trend visualization
- Store performance metrics
- Machine learning-based sales prediction
- Historical sales comparison

## Deployment Instructions

1. Fork this repository to your GitHub account
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your forked repository
6. Set the main file path as `app.py`
7. Click "Deploy"

## Local Development

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Project Structure

- `app.py`: Main Streamlit application
- `random_forest_generator.py`: Machine learning model implementation
- `data/`: Directory containing sales data
  - `sales.tsv`: Sales data file
  - `store_combined.tsv`: Store information file

## Important Notes

- The machine learning models are generated at runtime when you first run the application
- No pre-trained models are included in the repository
- The application will automatically train new models when needed

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt 