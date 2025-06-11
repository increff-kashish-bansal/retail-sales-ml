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
    sales_df = pd.read_csv('data/sales.tsv', sep='\t')
    logger.info("Data loaded successfully")
except Exception as e:
    logger.error(f"Error loading data: {str(e)}", exc_info=True)
    st.error(f"Error loading data: {str(e)}")
    st.stop()

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

use_default = st.checkbox("Use default files (sales_combined.tsv)")

if not use_default:
    sales_file = st.file_uploader("Upload Sales Data (TSV)", type=['tsv'])
else:
    sales_file = None

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
            # Train the model
            model, metrics, feature_importance, selected_features = train_random_forest()
            
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
    model = joblib.load('models/random_forest_model.joblib')
    transformer = joblib.load('models/yeo_johnson_transformer.joblib')
    store_encoder = joblib.load('models/store_encoder.joblib')
    feature_names = joblib.load('models/feature_names.joblib')
    store_metrics = pd.read_csv('models/store_metrics.csv')
    logger.info("All files loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}", exc_info=True)
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Prediction section
st.markdown("### 🔮 4. Make Predictions", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Input fields
future_date = st.date_input(
    "Select Future Date",
    value=datetime.now() + timedelta(days=30),
    min_value=datetime.now(),
    max_value=datetime.now() + timedelta(days=365),
    help="Select the date for which you want to predict sales",
    key="prediction_date_input"
)

# Get stores from sales data
available_stores = sales_df['store'].astype(str).str.strip().unique()
future_store = st.selectbox(
    "Select Store",
    options=sorted(available_stores),
    format_func=lambda x: f"Store {x}",
    help="Choose the store for prediction",
    key="store_selection"
)

future_disc_perc = st.slider(
    "Discount Percentage",
    min_value=0,
    max_value=100,
    value=0,
    step=5,
    help="Set the discount percentage for the prediction",
    key="discount_slider"
)

# Calculate default average MRP from sales data
store_sales = sales_df[sales_df['store'].astype(str).str.strip() == str(future_store).strip()]
if not store_sales.empty:
    default_avg_mrp = (store_sales['revenue'] / store_sales['qty']).mean()
else:
    default_avg_mrp = 1000.0  # Fallback value

st.write(f"Estimated Average MRP: ₹{default_avg_mrp:,.2f}")
future_avg_mrp = st.number_input(
    "Average MRP (₹)",
    min_value=0.0,
    value=float(default_avg_mrp),
    help="Average Maximum Retail Price for the store",
    key="mrp_input"
)

if st.button("Predict Revenue", key="make_prediction_button", type="primary"):
    try:
        with st.spinner("Making prediction..."):
            # Make prediction
            prediction, confidence_intervals = predict_revenue(
                store=future_store,
                future_date=future_date,
                discount_percentage=future_disc_perc,
                mrp=future_avg_mrp
            )
            
            # Display prediction results with better styling
            st.markdown("""
                <h2 style='color: #2E4053; border-bottom: 2px solid #2E4053; padding-bottom: 10px; margin-top: 40px;'>
                    Prediction Results
                </h2>
            """, unsafe_allow_html=True)
            
            # Get historical sales data
            sales_df = pd.read_csv('sales_combined.tsv', sep='\t')
            sales_df['day'] = pd.to_datetime(sales_df['day'])
            store_sales = sales_df[sales_df['store'].astype(str).str.strip() == str(future_store)]
            
            # Get actual sales for the same day and month in previous years
            actual_sales = store_sales[
                (store_sales['day'].dt.day == future_date.day) & 
                (store_sales['day'].dt.month == future_date.month)
            ].groupby(['store', 'day']).agg({
                'revenue': 'sum',
                'qty': 'sum'
            }).reset_index()
            
            # Create metrics row with better styling
            st.markdown("<div style='margin: 20px 0;'>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Predicted Revenue",
                    value=f"₹{prediction:,.2f}",
                    delta=None
                )
            
            if not actual_sales.empty:
                with col2:
                    actual_revenue = actual_sales['revenue'].iloc[-1]
                    trend = "📈" if prediction > actual_revenue else "📉"
                    st.metric(
                        label="Actual Revenue (Last Year)",
                        value=f"₹{actual_revenue:,.2f}",
                        delta=f"{trend} ₹{prediction - actual_revenue:,.2f}"
                    )
            
            if confidence_intervals:
                with col3:
                    st.metric(
                        label="Upper Bound (95% CI)",
                        value=f"₹{confidence_intervals[1]:,.2f}",
                        delta=None
                    )
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Display historical sales in a table with better styling
            if not actual_sales.empty:
                st.markdown("""
                    <h3 style='color: #2E4053; margin-top: 30px;'>
                        Historical Sales for Same Day/Month
                    </h3>
                """, unsafe_allow_html=True)
                
                # Create a formatted table of historical sales
                historical_data = []
                for _, row in actual_sales.iterrows():
                    historical_data.append({
                        'Date': row['day'].strftime('%Y-%m-%d'),
                        'Revenue': f"₹{row['revenue']:,.2f}",
                        'Quantity': f"{row['qty']:,.0f}"
                    })
                
                # Add average row
                avg_revenue = actual_sales['revenue'].mean()
                avg_qty = actual_sales['qty'].mean()
                historical_data.append({
                    'Date': 'Average',
                    'Revenue': f"₹{avg_revenue:,.2f}",
                    'Quantity': f"{avg_qty:,.0f}"
                })
                
                # Display the table with custom styling
                st.markdown("""
                    <style>
                        .dataframe {
                            width: 100%;
                            border-collapse: collapse;
                            margin: 20px 0;
                        }
                        .dataframe th {
                            background-color: #2E4053;
                            color: white;
                            padding: 12px;
                            text-align: left;
                        }
                        .dataframe td {
                            padding: 12px;
                            border-bottom: 1px solid #ddd;
                        }
                        .dataframe tr:last-child {
                            font-weight: bold;
                            background-color: #f8f9fa;
                        }
                    </style>
                """, unsafe_allow_html=True)
                
                st.table(pd.DataFrame(historical_data))
                
                # Add comparison metrics with better styling
                st.markdown("<div style='margin: 20px 0;'>", unsafe_allow_html=True)
                trend = "📈" if prediction > avg_revenue else "📉"
                st.metric(
                    label="Prediction vs Historical Average",
                    value=f"₹{prediction:,.2f}",
                    delta=f"{trend} ₹{prediction - avg_revenue:,.2f}"
                )
                st.markdown("</div>", unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        logger.error(f"Error in main: {str(e)}", exc_info=True) 