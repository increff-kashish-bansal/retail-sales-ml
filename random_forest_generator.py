import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error, explained_variance_score, mean_absolute_percentage_error
import os
import logging
from datetime import datetime
from sklearn.cluster import KMeans
import json
from sklearn.preprocessing import StandardScaler
import scipy.stats
from sklearn.preprocessing import PowerTransformer
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prevent duplicate handlers
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler())

def setup_logger():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"model_training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

def preprocess_data(sales_df):
    """
    Preprocess the sales data for model training.
    
    Args:
        sales_df (pd.DataFrame): Sales data
    
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target variable
    """
    try:
        logger.info("Starting data preprocessing...")
        
        # Convert store to string type
        sales_df['store'] = sales_df['store'].astype(str)
        
        # Filter out low revenue rows
        sales_df = sales_df[sales_df['revenue'] >= 100]
        logger.info(f"Filtered out low revenue rows. New shape: {sales_df.shape}")
        
        # Calculate MRP from revenue and quantity
        sales_df['mrp'] = (sales_df['revenue'] + sales_df['disc_value']) / sales_df['qty']
        logger.info("Calculated MRP using (disc_value + revenue) / qty")
        
        # Aggregate sales at store-day level
        daily_sales = sales_df.groupby(['store', 'day']).agg({
            'revenue': 'sum',
            'qty': 'sum',
            'disc_value': 'sum',
            'mrp': 'mean'
        }).reset_index()
        logger.info(f"Aggregated sales at store-day level. New shape: {daily_sales.shape}")
        
        # Calculate store metrics
        store_metrics = daily_sales.groupby('store').agg({
            'revenue': ['mean', 'std'],
            'mrp': 'mean',
            'qty': 'mean'
        }).reset_index()
        
        store_metrics.columns = ['store', 'avg_revenue', 'revenue_std', 'avg_mrp', 'avg_qty']
        store_metrics['revenue_volatility'] = store_metrics['revenue_std'] / store_metrics['avg_revenue']
        store_metrics['qty_volatility'] = daily_sales.groupby('store')['qty'].std() / daily_sales.groupby('store')['qty'].mean()
        
        # Add store metrics to daily sales
        daily_sales = daily_sales.merge(store_metrics, on='store', how='left')
        
        # Add date features
        daily_sales['day'] = pd.to_datetime(daily_sales['day'])
        daily_sales['dayofweek'] = daily_sales['day'].dt.dayofweek
        daily_sales['month'] = daily_sales['day'].dt.month
        daily_sales['year'] = daily_sales['day'].dt.year
        daily_sales['day_of_year'] = daily_sales['day'].dt.dayofyear
        daily_sales['day_is_weekend'] = daily_sales['dayofweek'].isin([5, 6]).astype(int)
        daily_sales['quarter'] = daily_sales['day'].dt.quarter
        
        # Add discount day signal
        daily_sales['discount_day_signal'] = (daily_sales['disc_value'] > 0).astype(int)
        
        # Encode store
        store_encoder = LabelEncoder()
        daily_sales['store_encoded'] = store_encoder.fit_transform(daily_sales['store'])
        
        # Save store encoder
        joblib.dump(store_encoder, 'models/store_encoder.joblib')
        
        # Save store metrics
        store_metrics.to_csv('models/store_metrics.csv', index=False)
        
        # Prepare feature matrix
        feature_cols = [
            'store_encoded', 'dayofweek', 'month', 'year', 'day_of_year',
            'day_is_weekend', 'quarter', 'disc_value', 'discount_day_signal',
            'avg_revenue', 'revenue_volatility', 'avg_mrp', 'avg_qty',
            'qty_volatility'
        ]
        
        X = daily_sales[feature_cols]
        y = daily_sales['revenue']
        
        # Save feature names
        joblib.dump(feature_cols, 'models/feature_names.joblib')
        
        logger.info("Data preprocessing completed successfully")
        return X, y
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}", exc_info=True)
        raise

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate MAPE after filtering out low revenue rows.
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        
    Returns:
        float: MAPE value
    """
    try:
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Filter out rows where actual revenue is less than 100
        mask = y_true >= 100
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        
        # Check if we have any data left after filtering
        if len(y_true_filtered) == 0:
            logger.warning("No data points left after filtering low revenue rows")
            return np.nan
        
        # Calculate MAPE on filtered data
        mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
        
        logger.info(f"MAPE calculated on {len(y_true_filtered)} rows after filtering")
        return mape
        
    except Exception as e:
        logger.error(f"Error calculating MAPE: {str(e)}")
        return np.nan

def analyze_revenue_distribution(y):
    """Analyze the distribution of revenue values"""
    try:
        # Calculate log of revenue
        y_log = np.log1p(y)  # Using log1p to handle zeros
        
        # Calculate statistics
        stats = {
            'mean': np.mean(y_log),
            'median': np.median(y_log),
            'std': np.std(y_log),
            'skew': scipy.stats.skew(y_log),
            'kurtosis': scipy.stats.kurtosis(y_log)
        }
        
        # Create histogram data
        hist, bins = np.histogram(y_log, bins=50)
        
        return stats, hist, bins
    except Exception as e:
        logger.error(f"Error analyzing revenue distribution: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        raise

def train_random_forest(sales_df=None):
    """
    Train a Random Forest model on the preprocessed data.
    
    Args:
        sales_df (pd.DataFrame, optional): Current sales data. If None, loads from file.
    
    Returns:
        tuple: (model, metrics, feature_importance, selected_features)
    """
    try:
        logger.info("Starting model training...")
        
        # Use provided sales_df or load from file
        if sales_df is None:
            sales_df = pd.read_csv('data/sales_sample.tsv', sep='\t')
            logger.info("Using sample data file for training")
        else:
            logger.info("Using provided dataset for training")
            
        logger.info(f"Sales data loaded. Shape: {sales_df.shape}")
        
        # Preprocess data
        X, y = preprocess_data(sales_df)
        
        # Apply Yeo-Johnson transformation to target variable
        transformer = PowerTransformer(method='yeo-johnson')
        y_transformed = transformer.fit_transform(y.values.reshape(-1, 1)).ravel()
        
        # Save transformer
        joblib.dump(transformer, 'models/yeo_johnson_transformer.joblib')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_transformed, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(model, 'models/random_forest_model.joblib')
        
        # Evaluate model
        metrics, feature_importance, selected_features = evaluate_model(model, X_test, y_test, transformer)
        
        logger.info("Model training completed successfully")
        return model, metrics, feature_importance, selected_features
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}", exc_info=True)
        raise

def predict_revenue(store, future_date, discount_percentage, mrp, sales_df=None):
    """
    Predict revenue for a store on a future date.
    
    Args:
        store (str): Store ID
        future_date (str): Future date in YYYY-MM-DD format
        discount_percentage (float): Discount percentage (0-100)
        mrp (float): Mean retail price
        sales_df (pd.DataFrame, optional): Current sales data. If None, loads from file.
    
    Returns:
        tuple: (predicted_revenue, confidence_intervals)
    """
    try:
        logger.info(f"Starting prediction for store {store}")
        logger.info(f"Input parameters: date={future_date}, discount={discount_percentage}%, mrp={mrp}")
        
        # Load required files
        model = joblib.load('models/random_forest_model.joblib')
        transformer = joblib.load('models/yeo_johnson_transformer.joblib')
        feature_names = joblib.load('models/feature_names.joblib')
        store_encoder = joblib.load('models/store_encoder.joblib')
        store_metrics = pd.read_csv('models/store_metrics.csv')
        
        # Use provided sales_df or load from file
        if sales_df is None:
            sales_df = pd.read_csv('data/sales_sample.tsv', sep='\t')
        
        logger.info("All required files loaded successfully")
        
        # Convert store to string and strip whitespace
        store = str(store).strip()
        
        # Find store in sales data
        logger.info(f"Looking for store {store} in sales data")
        store_sales = sales_df[sales_df['store'].astype(str).str.strip() == store]
        
        if len(store_sales) == 0:
            available_stores = sales_df['store'].astype(str).str.strip().unique()
            logger.warning(f"Store {store} not found in sales data. Available stores: {available_stores}")
            raise ValueError(f"No sales data found for store {store}")
            
        logger.info(f"Found {len(store_sales)} sales records for store {store}")
        
        # Convert future date to datetime
        future_date = pd.to_datetime(future_date)
        logger.info(f"Future date converted to: {future_date}")
        
        # Convert sales data dates to datetime
        store_sales['day'] = pd.to_datetime(store_sales['day'])
        
        # Analyze historical sales for same day and month
        historical_sales = store_sales[
            (store_sales['day'].dt.day == future_date.day) & 
            (store_sales['day'].dt.month == future_date.month) &
            (store_sales['day'].dt.year < future_date.year)  # Only include previous years
        ].copy()
        
        if len(historical_sales) > 0:
            logger.info("\nHistorical sales for same day and month:")
            logger.info("----------------------------------------")
            historical_sales['year'] = historical_sales['day'].dt.year
            historical_sales = historical_sales.sort_values('year')
            
            # Group by year to get daily totals
            yearly_sales = historical_sales.groupby('year').agg({
                'revenue': 'sum',
                'qty': 'sum'
            }).reset_index()
            
            for _, row in yearly_sales.iterrows():
                logger.info(f"Year: {row['year']}, Revenue: ₹{row['revenue']:,.2f}, Quantity: {row['qty']:,.2f}")
            
            avg_historical_revenue = yearly_sales['revenue'].mean()
            logger.info(f"\nAverage historical revenue for this day/month: ₹{avg_historical_revenue:,.2f}")
            logger.info("----------------------------------------\n")
        else:
            logger.info("\nNo historical sales data found for this day/month in previous years")
            logger.info("----------------------------------------\n")
        
        # Calculate daily sales
        daily_sales = store_sales.groupby('day').agg({
            'revenue': 'sum',
            'qty': 'sum'
        }).reset_index()
        
        # Calculate MRP from revenue and quantity
        daily_sales['mrp'] = daily_sales['revenue'] / daily_sales['qty']
        
        # Calculate store metrics from daily sales
        store_metrics = {
            'avg_revenue': daily_sales['revenue'].mean(),
            'revenue_std': daily_sales['revenue'].std(),
            'avg_mrp': daily_sales['mrp'].mean(),
            'avg_qty': daily_sales['qty'].mean(),
            'revenue_volatility': daily_sales['revenue'].std() / daily_sales['revenue'].mean() if daily_sales['revenue'].mean() > 0 else 0,
            'qty_volatility': daily_sales['qty'].std() / daily_sales['qty'].mean() if daily_sales['qty'].mean() > 0 else 0
        }
        
        logger.info("Store metrics calculated:")
        logger.info(f"avg_revenue: {store_metrics['avg_revenue']:,.2f}")
        logger.info(f"std_revenue: {store_metrics['revenue_std']:,.2f}")
        logger.info(f"avg_mrp: {store_metrics['avg_mrp']:,.2f}")
        logger.info(f"avg_quantity: {store_metrics['avg_qty']:,.2f}")
        logger.info(f"revenue_volatility: {store_metrics['revenue_volatility']:,.2f}")
        logger.info(f"qty_volatility: {store_metrics['qty_volatility']:,.2f}")
        
        # Create feature vector
        features = {}
        
        # Store features
        features['store_encoded'] = store_encoder.transform([store])[0]
        
        # Date features
        features['dayofweek'] = future_date.dayofweek
        features['month'] = future_date.month
        features['year'] = future_date.year
        features['day_of_year'] = future_date.dayofyear
        features['day_is_weekend'] = int(future_date.dayofweek in [5, 6])
        features['quarter'] = future_date.quarter
        
        # Discount features
        features['disc_value'] = mrp * (discount_percentage / 100)
        features['discount_day_signal'] = int(discount_percentage > 0)
        
        # Store metrics
        features['avg_revenue'] = float(store_metrics['avg_revenue'])
        features['revenue_volatility'] = float(store_metrics['revenue_volatility'])
        features['avg_mrp'] = float(store_metrics['avg_mrp'])
        features['avg_qty'] = float(store_metrics['avg_qty'])
        features['qty_volatility'] = float(store_metrics['qty_volatility'])
        
        # Log features for debugging
        logger.info("Features created:")
        for name, value in features.items():
            logger.info(f"{name}: {value}")
        
        # Create feature matrix as numpy array
        X = np.array([[features[col] for col in feature_names]])
        logger.info("Feature matrix created as numpy array")
        
        # Make prediction
        raw_prediction = model.predict(X)[0]
        logger.info(f"Raw prediction from model: {raw_prediction}")
        
        # Transform prediction back to original scale
        transformed_prediction = transformer.inverse_transform([[raw_prediction]])[0][0]
        logger.info(f"Transformed prediction: {transformed_prediction:,.2f}")
        
        # Calculate confidence intervals if available
        if hasattr(model, 'estimators_'):
            predictions = []
            for estimator in model.estimators_:
                pred = estimator.predict(X)[0]
                predictions.append(transformer.inverse_transform([[pred]])[0][0])
            
            lower_ci = np.percentile(predictions, 2.5)
            upper_ci = np.percentile(predictions, 97.5)
            logger.info(f"Confidence intervals: [{lower_ci:,.2f}, {upper_ci:,.2f}]")
            return transformed_prediction, (lower_ci, upper_ci)
        
        return transformed_prediction, None
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}", exc_info=True)
        raise

def load_data():
    """Load and prepare data for training"""
    try:
        # Load data
        sales_df = pd.read_csv('data/sales_sample.tsv', sep='\t')
        
        # Convert date columns to datetime
        sales_df['day'] = pd.to_datetime(sales_df['day'])
        
        # Ensure store column is string type in both dataframes
        sales_df['store'] = sales_df['store'].astype(str)
        
        return sales_df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test, transformer):
    """Evaluate model performance and return metrics"""
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Transform predictions and actual values back to original scale
        y_pred = transformer.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        y_test = transformer.inverse_transform(y_test.reshape(-1, 1)).ravel()
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        log_mae = mean_absolute_error(np.log1p(y_test), np.log1p(y_pred))
        
        # Calculate corrected metrics
        y_pred_corrected = np.maximum(y_pred, 0)  # Ensure non-negative predictions
        rmse_corrected = np.sqrt(mean_squared_error(y_test, y_pred_corrected))
        r2_corrected = r2_score(y_test, y_pred_corrected)
        mape_corrected = mean_absolute_percentage_error(y_test, y_pred_corrected) * 100
        log_mae_corrected = mean_absolute_error(np.log1p(y_test), np.log1p(y_pred_corrected))
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top features
        selected_features = feature_importance.head(10)['feature'].tolist()
        
        metrics = {
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'log_mae': log_mae,
            'rmse_corrected': rmse_corrected,
            'r2_corrected': r2_corrected,
            'mape_corrected': mape_corrected,
            'log_mae_corrected': log_mae_corrected
        }
        
        return metrics, feature_importance, selected_features
        
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        raise 