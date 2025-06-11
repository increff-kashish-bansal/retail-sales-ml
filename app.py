import streamlit as st
import pandas as pd
import numpy as np
import joblib
import logging
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from random_forest_generator import preprocess_data, train_random_forest, predict_revenue
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Set page config - must be first Streamlit command
st.set_page_config(
    page_title="Retail Sales Analysis & Forecast",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom handler to display logs in Streamlit
class StreamlitHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            st.text(msg)
        except Exception:
            self.handleError(record)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prevent duplicate handlers
if not logger.hasHandlers():
    logger.addHandler(StreamlitHandler())

# Define data paths
DATA_DIR = "data"
SAMPLE_FILE = "sales_sample.tsv"
COMBINED_FILE = "sales_combined.tsv"

def get_data_path(filename):
    """
    Get the correct path for data files, handling different OS path separators.
    """
    # Get the current working directory
    cwd = os.getcwd()
    logger.info(f"Current working directory: {cwd}")
    
    # List all files in the current directory and data directory
    logger.info(f"Files in current directory: {os.listdir(cwd)}")
    if os.path.exists(DATA_DIR):
        logger.info(f"Files in data directory: {os.listdir(DATA_DIR)}")
    else:
        logger.warning(f"Data directory '{DATA_DIR}' not found")
    
    # Try different possible paths
    possible_paths = [
        os.path.join(cwd, DATA_DIR, filename),  # /full/path/data/sales_sample.tsv
        os.path.join(DATA_DIR, filename),       # data/sales_sample.tsv
        os.path.join(cwd, filename),            # /full/path/sales_sample.tsv
        filename                                # sales_sample.tsv
    ]
    
    # Log all possible paths
    logger.info("Trying possible paths:")
    for path in possible_paths:
        logger.info(f"Checking path: {path}")
        if os.path.exists(path):
            logger.info(f"Found file at: {path}")
            return path
    
    # If no path works, raise a detailed error
    error_msg = (
        f"Could not find file '{filename}'. "
        f"Current directory: {cwd}\n"
        f"Files in current directory: {os.listdir(cwd)}\n"
        f"Data directory exists: {os.path.exists(DATA_DIR)}\n"
        f"Files in data directory: {os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else 'N/A'}\n"
        f"Tried paths: {possible_paths}"
    )
    logger.error(error_msg)
    raise FileNotFoundError(error_msg)

def create_sample_data():
    """Create a sample dataset for testing"""
    try:
        logger.info("Creating sample dataset...")
        # Create date range
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        # Create sample data
        data = []
        stores = ['Store1', 'Store2', 'Store3']
        skus = ['SKU1', 'SKU2', 'SKU3', 'SKU4', 'SKU5']
        
        for date in dates:
            for store in stores:
                for sku in skus:
                    # Generate random values
                    qty = np.random.randint(1, 100)
                    mrp = np.random.uniform(100, 1000)
                    disc_value = np.random.uniform(0, mrp * 0.3) if np.random.random() < 0.3 else 0
                    revenue = (mrp - disc_value) * qty
                    
                    data.append({
                        'day': date,
                        'store': store,
                        'sku': sku,
                        'disc_value': disc_value,
                        'revenue': revenue,
                        'qty': qty
                    })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        logger.info(f"Created sample dataset with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error creating sample data: {str(e)}", exc_info=True)
        raise

def load_sales_data(file_path=None, uploaded_file=None):
    """
    Load sales data from either a file path or an uploaded file.
    Returns the loaded DataFrame and any error message.
    """
    try:
        if uploaded_file is not None:
            logger.info("Loading data from uploaded file...")
            logger.info(f"Uploaded file type: {type(uploaded_file)}")
            logger.info(f"Uploaded file name: {uploaded_file.name}")
            
            # Read the file content first
            content = uploaded_file.read()
            logger.info(f"Raw file content (first 1000 bytes): {content[:1000]}")
            
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Try to detect file format
            if uploaded_file.name.endswith('.tsv'):
                logger.info("Detected TSV file format")
                try:
                    sales_df = pd.read_csv(uploaded_file, sep='\t', encoding='utf-8')
                    logger.info("Successfully read TSV file with tab separator")
                except Exception as e:
                    logger.warning(f"Failed to read as TSV: {str(e)}")
                    uploaded_file.seek(0)
                    try:
                        sales_df = pd.read_csv(uploaded_file, sep=',', encoding='utf-8')
                        logger.info("Successfully read TSV file with comma separator")
                    except Exception as e:
                        logger.error(f"Failed to read file: {str(e)}")
                        raise
            elif uploaded_file.name.endswith('.csv'):
                logger.info("Detected CSV file format")
                try:
                    sales_df = pd.read_csv(uploaded_file, encoding='utf-8')
                    logger.info("Successfully read CSV file")
                except Exception as e:
                    logger.error(f"Failed to read CSV file: {str(e)}")
                    raise
            else:
                logger.warning(f"Unknown file format: {uploaded_file.name}")
                # Try to detect separator
                if '\t' in content.decode('utf-8'):
                    logger.info("Detected tab separator in content")
                    uploaded_file.seek(0)
                    sales_df = pd.read_csv(uploaded_file, sep='\t', encoding='utf-8')
                else:
                    logger.info("Using comma separator as default")
                    uploaded_file.seek(0)
                    sales_df = pd.read_csv(uploaded_file, encoding='utf-8')
            
            logger.info(f"Successfully loaded uploaded file. Shape: {sales_df.shape}")
            logger.info(f"Columns in uploaded file: {sales_df.columns.tolist()}")
            logger.info(f"First few rows of uploaded data:\n{sales_df.head()}")
            
        else:
            # Get the correct path for the file
            file_path = get_data_path(file_path)
            logger.info(f"Loading data from file: {file_path}")
            
            if not os.path.exists(file_path):
                error_msg = (
                    f"File not found: {file_path}\n"
                    f"Current directory: {os.getcwd()}\n"
                    f"Files in current directory: {os.listdir('.')}\n"
                    f"Data directory exists: {os.path.exists(DATA_DIR)}\n"
                    f"Files in data directory: {os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else 'N/A'}"
                )
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            logger.info(f"File size: {os.path.getsize(file_path)} bytes")
            
            # Check if file is a Git LFS pointer
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.startswith('version https://git-lfs.github.com/spec/v1'):
                    logger.warning("Git LFS pointer file detected. Using sample data instead.")
                    sales_df = create_sample_data()
                    logger.info("Successfully created and loaded sample data")
                    return sales_df, None
            
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    logger.info(f"Trying to read file with {encoding} encoding...")
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                        logger.info(f"Successfully read file with {encoding} encoding")
                        logger.info(f"Raw file content (first 1000 bytes): {content[:1000]}")
                        break
                except UnicodeDecodeError:
                    logger.warning(f"Failed to read with {encoding} encoding")
                    continue
            
            # Try different separators with the successful encoding
            try:
                sales_df = pd.read_csv(file_path, sep='\t', encoding=encoding)
                logger.info("Successfully read with tab separator")
            except Exception as e1:
                logger.warning(f"Failed to read with tab separator: {str(e1)}")
                try:
                    sales_df = pd.read_csv(file_path, sep=',', encoding=encoding)
                    logger.info("Successfully read with comma separator")
                except Exception as e2:
                    logger.warning(f"Failed to read with comma separator: {str(e2)}")
                    # Try to detect separator
                    if '\t' in content:
                        sales_df = pd.read_csv(file_path, sep='\t', engine='python', encoding=encoding)
                        logger.info("Successfully read with python engine and tab separator")
                    else:
                        raise Exception("Could not determine file format")
        
        # Check for column name variations
        column_mapping = {
            'day': ['day', 'date', 'Day', 'Date', 'DAY', 'DATE'],
            'store': ['store', 'Store', 'store_id', 'Store_ID', 'STORE'],
            'sku': ['sku', 'SKU', 'product_id', 'Product_ID', 'PRODUCT_ID'],
            'disc_value': ['disc_value', 'discount', 'Discount', 'discount_value', 'DISCOUNT'],
            'revenue': ['revenue', 'Revenue', 'sales', 'Sales', 'REVENUE', 'SALES'],
            'qty': ['qty', 'quantity', 'Quantity', 'QTY', 'QUANTITY']
        }
        
        # Map columns to standard names
        for standard_name, variations in column_mapping.items():
            for var in variations:
                if var in sales_df.columns:
                    if var != standard_name:
                        logger.info(f"Renaming column '{var}' to '{standard_name}'")
                        sales_df = sales_df.rename(columns={var: standard_name})
                    break
        
        # Ensure required columns exist
        required_columns = ['day', 'store', 'sku', 'disc_value', 'revenue', 'qty']
        missing_columns = [col for col in required_columns if col not in sales_df.columns]
        
        if missing_columns:
            error_msg = f"Missing required columns: {', '.join(missing_columns)}"
            logger.error(error_msg)
            logger.error(f"Available columns: {sales_df.columns.tolist()}")
            return None, error_msg
        
        # Convert day to datetime
        try:
            sales_df['day'] = pd.to_datetime(sales_df['day'])
            logger.info("Successfully converted 'day' column to datetime")
        except Exception as e:
            error_msg = f"Error converting 'day' column to datetime: {str(e)}"
            logger.error(error_msg)
            logger.error(f"'day' column values: {sales_df['day'].head()}")
            return None, error_msg
        
        # Convert numeric columns
        numeric_columns = ['disc_value', 'revenue', 'qty']
        for col in numeric_columns:
            try:
                sales_df[col] = pd.to_numeric(sales_df[col], errors='coerce')
                logger.info(f"Successfully converted '{col}' column to numeric")
            except Exception as e:
                error_msg = f"Error converting '{col}' column to numeric: {str(e)}"
                logger.error(error_msg)
                logger.error(f"'{col}' column values: {sales_df[col].head()}")
                return None, error_msg
        
        # Handle any remaining NaN values
        sales_df = sales_df.fillna({
            'disc_value': 0,
            'revenue': 0,
            'qty': 0
        })
        
        logger.info("Data loaded successfully")
        logger.info(f"Final data shape: {sales_df.shape}")
        logger.info(f"Final columns: {sales_df.columns.tolist()}")
        logger.info(f"Data types:\n{sales_df.dtypes}")
        logger.info(f"Missing values:\n{sales_df.isnull().sum()}")
        
        return sales_df, None
        
    except Exception as e:
        error_msg = f"Error loading data: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return None, error_msg

# Initialize session state variables
if 'sales_df' not in st.session_state:
    st.session_state.sales_df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'data_error' not in st.session_state:
    st.session_state.data_error = None
if 'model_error' not in st.session_state:
    st.session_state.model_error = None

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Title and description
st.markdown("""
    <h1 style='text-align: center; color: #2E4053; margin-bottom: 30px;'>
        📊 Retail Sales Analysis & Forecast
    </h1>
""", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #566573; margin-bottom: 40px;'>
        This application helps analyze retail sales data and predict future sales using machine learning.
        Use the sections below to explore and analyze the data.
    </div>
""", unsafe_allow_html=True)

# File upload section
st.markdown("### 📁 1. Data Upload", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

use_sample = st.checkbox("Use sample dataset", value=True)

if use_sample:
    try:
        sales_df, error = load_sales_data(SAMPLE_FILE)
        if error:
            st.error(f"Error loading sample data: {error}")
        else:
            st.success("Sample data loaded successfully!")
            st.session_state.sales_df = sales_df
            st.session_state.data_error = None
    except FileNotFoundError as e:
        st.error(f"Sample data file not found: {str(e)}")
        st.session_state.data_error = str(e)
    except Exception as e:
        st.error(f"Unexpected error loading sample data: {str(e)}")
        st.session_state.data_error = str(e)
else:
    uploaded_file = st.file_uploader("Upload your sales data (TSV format)", type=['tsv'])
    if uploaded_file is not None:
        sales_df, error = load_sales_data(uploaded_file=uploaded_file)
        if error:
            st.error(f"Error loading uploaded data: {error}")
        else:
            st.success("Data uploaded successfully!")
            st.session_state.sales_df = sales_df
            st.session_state.data_error = None

if error:
    st.error(error)
    st.stop()

# Store the loaded data in session state
st.session_state.sales_df = sales_df
st.session_state.data_error = error

st.markdown("---")

# Data Summary section
st.markdown("### 📊 2. Data Summary", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if st.session_state.sales_df is not None:
    try:
        # Calculate metrics
        total_revenue = st.session_state.sales_df['revenue'].sum()
        avg_revenue = st.session_state.sales_df['revenue'].mean()
        prev_avg_revenue = st.session_state.sales_df[st.session_state.sales_df['day'] < st.session_state.sales_df['day'].max()]['revenue'].mean()
        revenue_trend = "📈" if avg_revenue > prev_avg_revenue else "📉"

        # Calculate error metrics
        if 'model' in st.session_state and st.session_state.model is not None:
            try:
                # Get predictions
                X = st.session_state.sales_df[st.session_state.selected_features]
                y_true = st.session_state.sales_df['revenue']
                y_pred = st.session_state.model.predict(X)
                
                # Calculate error metrics
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mape = mean_absolute_percentage_error(y_true, y_pred)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Total Revenue",
                        f"₹{total_revenue:,.2f}",
                        delta=f"{revenue_trend} ₹{total_revenue/len(st.session_state.sales_df):,.2f}"
                    )
                with col2:
                    st.metric(
                        "Mean Absolute Error",
                        f"₹{mae:,.2f}",
                        delta=f"{mape:.1f}%"
                    )
                with col3:
                    st.metric(
                        "Root Mean Square Error",
                        f"₹{rmse:,.2f}"
                    )
            except Exception as e:
                logger.error(f"Error calculating metrics: {str(e)}", exc_info=True)
                # Display basic metrics without error calculations
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Total Revenue",
                        f"₹{total_revenue:,.2f}",
                        delta=f"{revenue_trend} ₹{total_revenue/len(st.session_state.sales_df):,.2f}"
                    )
                with col2:
                    st.metric(
                        "Average Revenue",
                        f"₹{avg_revenue:,.2f}"
                    )
                with col3:
                    st.metric(
                        "Number of Records",
                        f"{len(st.session_state.sales_df):,}"
                    )
        else:
            # Display basic metrics when no model is available
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Total Revenue",
                    f"₹{total_revenue:,.2f}",
                    delta=f"{revenue_trend} ₹{total_revenue/len(st.session_state.sales_df):,.2f}"
                )
            with col2:
                st.metric(
                    "Average Revenue",
                    f"₹{avg_revenue:,.2f}"
                )
            with col3:
                st.metric(
                    "Number of Records",
                    f"{len(st.session_state.sales_df):,}"
                )
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        logger.error(f"Error processing data: {str(e)}", exc_info=True)

st.markdown("---")

# Model training section
st.markdown("### 🤖 3. Train Model", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if st.button("Train Model", key="train_model_button"):
    try:
        with st.spinner("Training model..."):
            model, metrics, feature_importance, selected_features = train_random_forest(st.session_state.sales_df)
            st.session_state.model = model
            st.session_state.model_metrics = metrics
            st.session_state.selected_features = selected_features
            st.session_state.feature_importance = feature_importance
            
            # Display model metrics if available
            if st.session_state.model_metrics is not None:
                st.markdown("""
                    <div style='color: #2E4053; padding: 10px 0; font-size: 1.2em;'>
                        Model Performance Metrics
                    </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RMSE", f"₹{st.session_state.model_metrics['rmse']:,.2f}")
                with col2:
                    st.metric("R² Score", f"{st.session_state.model_metrics['r2']:.3f}")
                with col3:
                    st.metric("MAPE", f"{st.session_state.model_metrics['mape']:.1f}%")
            else:
                st.markdown("""
                    <div style='color: #2E4053; padding: 10px 0; font-size: 1.2em;'>
                        Model Performance Metrics
                    </div>
                """, unsafe_allow_html=True)
                st.info("Model metrics will be available after training the model.")
            
            st.markdown("---")
            
            # Display feature importance
            st.markdown("""
                <div style='color: #2E4053; padding: 10px 0; font-size: 1.2em;'>
                    Top 10 Most Important Features
                </div>
            """, unsafe_allow_html=True)
            
            top_features = feature_importance.head(10)
            fig = px.bar(
                top_features,
                x='importance',
                y='feature',
                orientation='h',
                title='Feature Importance',
                labels={'importance': 'Importance Score', 'feature': 'Feature'}
            )
            st.plotly_chart(fig)
            
    except Exception as e:
        st.error(f"Error during training: {str(e)}")
        logger.error(f"Error during training: {str(e)}", exc_info=True)

# Display model information
st.subheader("Model Information")
if st.session_state.model is not None:
    st.write("Model type:", type(st.session_state.model).__name__)
    if hasattr(st.session_state.model, 'get_params'):
        st.write("Model parameters:", st.session_state.model.get_params())
    if st.session_state.selected_features is not None:
        st.write("Selected features:", st.session_state.selected_features)

# Display error information
st.subheader("Error Information")
if st.session_state.data_error:
    st.error(f"Data Error: {st.session_state.data_error}")
if st.session_state.model_error:
    st.error(f"Model Error: {st.session_state.model_error}")

# Prediction section
st.markdown("### 🔮 4. Make Predictions", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Only show prediction section if model is loaded
if st.session_state.model is not None:
    # Input fields
    future_date = st.date_input(
        "Select Future Date",
        value=datetime.now() + timedelta(days=30),
        min_value=datetime.now(),
        max_value=datetime.now() + timedelta(days=365),
        help="Select the date for which you want to predict sales",
        key="prediction_date_input"
    )
    
    # Get stores from current sales data
    available_stores = st.session_state.sales_df['store'].astype(str).str.strip().unique()
    future_store = st.selectbox(
        "Select Store",
        options=sorted(available_stores),
        format_func=lambda x: f"Store {x}",
        help="Choose the store for prediction",
        key="store_selection"
    )
    
    # Calculate default MRP based on current sales data
    store_sales = st.session_state.sales_df[st.session_state.sales_df['store'] == future_store]
    if not store_sales.empty:
        default_mrp = (store_sales['revenue'] + store_sales['disc_value']).mean() / store_sales['qty'].mean()
    else:
        default_mrp = 1000.0  # Fallback value
    
    mrp = st.number_input(
        "Maximum Retail Price (MRP)",
        min_value=0.0,
        value=float(default_mrp),
        step=100.0,
        help="Enter the MRP for the product",
        key="mrp_input"
    )
    
    discount_percentage = st.slider(
        "Discount Percentage",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
        help="Select the discount percentage to apply",
        key="discount_slider"
    )
    
    # Make prediction button
    if st.button("Predict Revenue", key="predict_button"):
        try:
            with st.spinner("Making prediction..."):
                # Make prediction
                predicted_revenue, confidence = predict_revenue(
                    future_store,
                    future_date,
                    discount_percentage,
                    mrp,
                    sales_df=st.session_state.sales_df
                )
                
                # Display prediction results
                st.markdown("""
                    <div style='color: #2E4053; padding: 10px 0; font-size: 1.2em;'>
                        Prediction Results
                    </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Predicted Revenue",
                        f"₹{predicted_revenue:,.2f}",
                        delta=f"Confidence: {confidence:.1f}%"
                    )
                
                # Get actual sales data for comparison
                actual_sales = st.session_state.sales_df[
                    (st.session_state.sales_df['store'] == future_store) &
                    (st.session_state.sales_df['day'].dt.date == future_date)
                ]
                
                if not actual_sales.empty:
                    actual_revenue = actual_sales['revenue'].sum()
                    with col2:
                        st.metric(
                            "Actual Revenue",
                            f"₹{actual_revenue:,.2f}",
                            delta=f"Difference: ₹{predicted_revenue - actual_revenue:,.2f}"
                        )
                
                # Add prediction to session state
                new_prediction = pd.DataFrame({
                    'store': [future_store],
                    'day': [future_date],
                    'avg_mrp': [mrp],
                    'disc_percentage': [discount_percentage],
                    'predicted_revenue': [predicted_revenue],
                    'confidence': [confidence]
                })
                st.session_state.predictions = pd.concat([st.session_state.predictions, new_prediction], ignore_index=True)
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            logger.error(f"Error making prediction: {str(e)}", exc_info=True)
else:
    st.info("Please train the model first to enable predictions.")

# Display diagnostic information in an expander
with st.expander("Diagnostic Information"):
    st.subheader("Data Information")
    if st.session_state.sales_df is not None:
        st.write(f"Number of rows: {len(st.session_state.sales_df)}")
        st.write(f"Number of columns: {len(st.session_state.sales_df.columns)}")
        st.write("Column names:", st.session_state.sales_df.columns.tolist())
        st.write("Data types:", st.session_state.sales_df.dtypes.to_dict())
    
    st.subheader("Model Information")
    if st.session_state.model is not None:
        st.write("Model type:", type(st.session_state.model).__name__)
        st.write("Model parameters:", st.session_state.model.get_params())
    
    st.subheader("Error Information")
    if st.session_state.data_error:
        st.error(f"Data error: {st.session_state.data_error}")
    if st.session_state.model_error:
        st.error(f"Model error: {st.session_state.model_error}")

def train_random_forest(sales_df):
    """Train a Random Forest model on the sales data"""
    try:
        logger.info("Starting model training...")
        
        # Prepare features
        X = sales_df[['store', 'sku', 'disc_value']]
        y = sales_df['revenue']
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X, y)
        
        # Calculate metrics
        y_pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        mape = mean_absolute_percentage_error(y, y_pred) * 100
        
        metrics = {
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save model and related files
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/random_forest_model.joblib')
        joblib.dump(X.columns.tolist(), 'models/feature_names.joblib')
        
        logger.info("Model training completed successfully")
        return model, metrics, feature_importance, X.columns.tolist()
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}", exc_info=True)
        raise 