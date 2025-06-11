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

# Load data
try:
    logger.info("Loading data...")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Files in current directory: {os.listdir('.')}")
    logger.info(f"Files in data directory: {os.listdir('data')}")
    
    # Try different encodings and separators
    encodings = ['utf-8', 'latin1', 'cp1252']
    separators = ['\t', ',', ';']
    
    for encoding in encodings:
        for sep in separators:
            try:
                logger.info(f"Trying to load with encoding={encoding}, separator={sep}")
                sales_df = pd.read_csv('data/sales_sample.tsv', sep=sep, encoding=encoding)
                logger.info(f"Successfully loaded with encoding={encoding}, separator={sep}")
                logger.info(f"Data loaded. Shape: {sales_df.shape}")
                logger.info(f"Columns in dataset: {sales_df.columns.tolist()}")
                
                # Check if we have the required columns
                required_columns = ['day', 'store', 'sku', 'disc_value', 'revenue', 'qty']
                missing_columns = [col for col in required_columns if col not in sales_df.columns]
                
                if not missing_columns:
                    # Convert day to datetime
                    sales_df['day'] = pd.to_datetime(sales_df['day'])
                    logger.info("Data loaded successfully")
                    break
                else:
                    logger.warning(f"Missing columns with encoding={encoding}, separator={sep}: {missing_columns}")
                    continue
            except Exception as e:
                logger.warning(f"Failed to load with encoding={encoding}, separator={sep}: {str(e)}")
                continue
        else:
            continue
        break
    else:
        raise ValueError("Could not load data with any combination of encoding and separator")
    
except Exception as e:
    logger.error(f"Error loading data: {str(e)}", exc_info=True)
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Initialize session state for predictions if it doesn't exist
if 'predictions' not in st.session_state:
    st.session_state.predictions = pd.DataFrame(columns=['store', 'day', 'avg_mrp', 'disc_percentage', 'predicted_revenue', 'confidence'])

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

use_default = st.checkbox("Use default sample file (sales_sample.tsv)")

if not use_default:
    sales_file = st.file_uploader("Upload Sales Data (TSV)", type=['tsv'])
    if sales_file is not None:
        try:
            sales_df = pd.read_csv(sales_file, sep='\t')
            st.success("Custom data file loaded successfully!")
        except Exception as e:
            st.error(f"Error loading custom file: {str(e)}")
            st.info("Falling back to sample data file...")
            sales_df = pd.read_csv('data/sales_sample.tsv', sep='\t')
    else:
        st.info("No file uploaded. Using sample data file...")
        sales_df = pd.read_csv('data/sales_sample.tsv', sep='\t')
else:
    sales_df = pd.read_csv('data/sales_sample.tsv', sep='\t')
    st.success("Using sample data file.")

st.markdown("---")

# Data Summary section
st.markdown("### 📊 2. Data Summary", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if 'sales_df' in locals():
    try:
        # Ensure day column is datetime
        sales_df['day'] = pd.to_datetime(sales_df['day'])
        
        # Display basic statistics
        st.markdown("""
            <div style='color: #2E4053; padding: 10px 0; font-size: 1.2em;'>
                Basic Statistics
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_revenue = sales_df['revenue'].sum()
            avg_revenue = sales_df['revenue'].mean()
            prev_avg_revenue = sales_df[sales_df['day'] < sales_df['day'].max()]['revenue'].mean()
            revenue_trend = "📈" if avg_revenue > prev_avg_revenue else "📉"
            
            st.metric(
                "Total Revenue", 
                f"₹{total_revenue:,.2f}",
                delta=f"{revenue_trend} ₹{total_revenue/len(sales_df['day'].unique()):,.2f} per day"
            )
            st.metric(
                "Average Revenue", 
                f"₹{avg_revenue:,.2f}",
                delta=f"{revenue_trend} ₹{avg_revenue - prev_avg_revenue:,.2f} vs previous period"
            )
        
        with col2:
            total_qty = sales_df['qty'].sum()
            avg_qty = sales_df['qty'].mean()
            prev_avg_qty = sales_df[sales_df['day'] < sales_df['day'].max()]['qty'].mean()
            qty_trend = "📈" if avg_qty > prev_avg_qty else "📉"
            
            st.metric(
                "Total Quantity", 
                f"{total_qty:,.0f}",
                delta=f"{qty_trend} {total_qty/len(sales_df['day'].unique()):,.0f} per day"
            )
            st.metric(
                "Average Quantity", 
                f"{avg_qty:,.2f}",
                delta=f"{qty_trend} {avg_qty - prev_avg_qty:,.2f} vs previous period"
            )
        
        with col3:
            num_stores = sales_df['store'].nunique()
            date_range = f"{sales_df['day'].min().strftime('%Y-%m-%d')} to {sales_df['day'].max().strftime('%Y-%m-%d')}"
            days_span = (sales_df['day'].max() - sales_df['day'].min()).days
            
            st.metric(
                "Number of Stores", 
                f"{num_stores:,}",
                delta=f"Active over {days_span} days"
            )
            st.metric(
                "Date Range", 
                date_range,
                delta=f"{days_span} days of data"
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
            # Train the model with current dataset
            model, metrics, feature_importance, selected_features = train_random_forest(sales_df=sales_df)
            
            st.success("Model training completed!")
            
            # Display metrics
            st.markdown("""
                <div style='color: #2E4053; padding: 10px 0; font-size: 1.2em;'>
                    Model Performance Metrics
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RMSE", f"₹{metrics['rmse']:,.2f}")
                st.metric("R² Score", f"{metrics['r2']:.3f}")
            with col2:
                st.metric("MAPE", f"{metrics['mape']:.1f}%")
                st.metric("Log MAE", f"{metrics['log_mae']:.3f}")
            
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

# Load model and related files
try:
    logger.info("Loading model and related files...")
    
    # Check if model files exist
    model_files = {
        'model': 'models/random_forest_model.joblib',
        'transformer': 'models/yeo_johnson_transformer.joblib',
        'store_encoder': 'models/store_encoder.joblib',
        'feature_names': 'models/feature_names.joblib',
        'store_metrics': 'models/store_metrics.csv'
    }
    
    missing_files = [name for name, path in model_files.items() if not os.path.exists(path)]
    
    if missing_files:
        logger.warning(f"Missing model files: {', '.join(missing_files)}")
        logger.info("Model files will be created when training is performed")
        model = None
        transformer = None
        store_encoder = None
        feature_names = None
        store_metrics = None
    else:
        model = joblib.load(model_files['model'])
        transformer = joblib.load(model_files['transformer'])
        store_encoder = joblib.load(model_files['store_encoder'])
        feature_names = joblib.load(model_files['feature_names'])
        store_metrics = pd.read_csv(model_files['store_metrics'])
        logger.info("All files loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}", exc_info=True)
    st.error(f"Error loading model: {str(e)}")
    model = None
    transformer = None
    store_encoder = None
    feature_names = None
    store_metrics = None

# Prediction section
st.markdown("### 🔮 4. Make Predictions", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Only show prediction section if model is loaded
if model is not None:
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
    available_stores = sales_df['store'].astype(str).str.strip().unique()
    future_store = st.selectbox(
        "Select Store",
        options=sorted(available_stores),
        format_func=lambda x: f"Store {x}",
        help="Choose the store for prediction",
        key="store_selection"
    )
    
    # Calculate default MRP based on current sales data
    store_sales = sales_df[sales_df['store'] == future_store]
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
                    sales_df=sales_df
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
                actual_sales = sales_df[
                    (sales_df['store'] == future_store) &
                    (sales_df['day'].dt.date == future_date)
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