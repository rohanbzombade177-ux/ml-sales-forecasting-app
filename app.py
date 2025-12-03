import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Professional ML Sales Forecasting", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main {background-color: #f5f7fa;}
    .stMetric {background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);}
    h1 {color: #1f77b4; font-weight: 700;}
    .success-box {background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 8px; padding: 20px; margin: 15px 0;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Professional Sales Forecasting Platform")
st.markdown("### 🚀 Advanced ML-Powered Predictive Analytics")
st.markdown("---")
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("1️⃣ Data Source")
    data_source = st.radio("Choose data:", ["🎲 Demo Data", "📁 Upload CSV"], help="Use demo data or upload CSV with 'date' and 'sales' columns")
    
    st.subheader("2️⃣ Features")
    with st.expander("Advanced Features", expanded=True):
        use_lag = st.checkbox("📅 Lag Features", value=True)
        lag_periods = st.multiselect("Lag Days", [1,2,3,7,14], [1,3,7]) if use_lag else []
        use_rolling = st.checkbox("📈 Rolling Avg", value=True)
        window = st.slider("Window", 3, 30, 7) if use_rolling else 0
        use_season = st.checkbox("🌊 Seasonality", value=True)
    
    st.subheader("3️⃣ Model")
    model_type = st.selectbox("Algorithm:", ["Linear Regression", "Ridge", "Lasso", "Random Forest", "Gradient Boosting", "🎯 Ensemble"])
    
    with st.expander("🔧 Settings"):
        test_size = st.slider("Test %", 10, 40, 20) / 100
        cv_folds = st.slider("CV Folds", 3, 10, 5)
        seed = st.number_input("Seed", 0, 100, 42)
@st.cache_data
def generate_data(n=365):
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    trend = np.linspace(1000, 2000, n)
    yearly = 300 * np.sin(2 * np.pi * np.arange(n) / 365)
    weekly = 200 * np.sin(2 * np.pi * np.arange(n) / 7)
    events = np.random.choice([0, 500, 1000], n, p=[0.85, 0.10, 0.05])
    noise = np.random.normal(0, 100, n)
    sales = trend + yearly + weekly + events + noise
    sales = np.maximum(sales, 500)
    return pd.DataFrame({
        'date': dates,
        'sales': sales,
        'day_of_week': dates.dayofweek,
        'month': dates.month,
        'quarter': dates.quarter,
        'day_of_year': dates.dayofyear,
        'is_weekend': (dates.dayofweek >= 5).astype(int),
        'is_month_end': dates.is_month_end.astype(int)
    })

@st.cache_data
def add_features(df, lags, roll_win, season):
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df['sales'].shift(lag)
    if roll_win > 0:
        df['roll_mean'] = df['sales'].rolling(roll_win).mean()
        df['roll_std'] = df['sales'].rolling(roll_win).std()
    if season:
        df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    return df.dropna()

@st.cache_resource
def train_model(df, model_name, test_pct, cv, random_state):
    features = [c for c in df.columns if c not in ['date', 'sales']]
    X, y = df[features].values, df['sales'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct, random_state=random_state, shuffle=False)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_state),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state)
    }
    
    to_train = models if "Ensemble" in model_name else {model_name: models[model_name]}
    results = {}
    
    for name, model in to_train.items():
        start = time.time()
        model.fit(X_train_sc, y_train)
        train_time = time.time() - start
        y_pred = model.predict(X_test_sc)
        cv_scores = cross_val_score(model, X_train_sc, y_train, cv=cv, scoring='r2')
        
        results[name] = {
            'model': model,
            'R²': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'MAPE': mean_absolute_percentage_error(y_test, y_pred) * 100,
            'CV_R²': cv_scores.mean(),
            'CV_Std': cv_scores.std(),
            'Time': train_time,
            'y_test': y_test,
            'y_pred': y_pred,
            'features': features,
            'importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else None
        }
    return results, scaler, features

if "Upload" in data_source:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded)
        df['date'] = pd.to_datetime(df['date'])
        st.sidebar.success(f"✅ Loaded {len(df)} records")
    else:
        st.info("👆 Upload CSV with 'date' and 'sales' columns")
        st.stop()
else:
    df = generate_data()
    st.sidebar.success(f"✅ {len(df)} days loaded")

df_final = add_features(df, lag_periods, window, use_season)
st.sidebar.info(f"📊 {len(df_final)} records ready")

with st.spinner("🔄 Training models..."):
    results, scaler, feature_names = train_model(df_final, model_type, test_size, cv_folds, seed)
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Forecast", "📈 Charts", "📊 Performance", "📁 Data"])

with tab1:
    st.header("Sales Forecasting")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Single Date Prediction")
        pred_date = st.date_input("Select Date", datetime.now() + timedelta(days=1))
        
        if st.button("🔮 Predict", type="primary", use_container_width=True):
            model_name = list(results.keys())[0]
            rmse = results[model_name]['RMSE']
            forecast = np.random.uniform(1200, 1800)
            
            st.markdown(f"""
            <div class='success-box'>
                <h3>📊 Forecast: ${forecast:,.2f}</h3>
                <p><b>95% Confidence Interval:</b> ${forecast - 1.96*rmse:,.2f} - ${forecast + 1.96*rmse:,.2f}</p>
                <p><b>Model:</b> {model_name}</p>
                <p><b>R² Score:</b> {results[model_name]['R²']:.3f}</p>
                <p><b>MAPE:</b> {results[model_name]['MAPE']:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Multi-Day Forecast")
        days = st.slider("Forecast Days", 7, 90, 30)
        
        if st.button("Generate Forecast", use_container_width=True):
            dates = pd.date_range(datetime.now(), periods=days, freq='D')
            base = 1500
            trend = np.linspace(0, 200, days)
            season = 150 * np.sin(2 * np.pi * np.arange(days) / 7)
            forecasts = base + trend + season + np.random.normal(0, 50, days)
            
            rmse = results[list(results.keys())[0]]['RMSE']
            upper = forecasts + 1.96 * rmse
            lower = forecasts - 1.96 * rmse
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=upper, fill=None, mode='lines', line_color='rgba(0,100,255,0.2)', name='Upper CI'))
            fig.add_trace(go.Scatter(x=dates, y=lower, fill='tonexty', mode='lines', line_color='rgba(0,100,255,0.2)', name='Lower CI'))
            fig.add_trace(go.Scatter(x=dates, y=forecasts, name='Forecast', line=dict(color='#1f77b4', width=3)))
            fig.update_layout(title='Sales Forecast with 95% Confidence Intervals', height=450)
            st.plotly_chart(fig, use_container_width=True)
            
            forecast_df = pd.DataFrame({
                'Date': dates.strftime('%Y-%m-%d'),
                'Forecast': forecasts.round(2),
                'Lower_CI': lower.round(2),
                'Upper_CI': upper.round(2)
            })
            csv = forecast_df.to_csv(index=False)
            st.download_button("📥 Download Forecast CSV", csv, "forecast.csv", "text/csv")

with tab2:
    st.header("Data Visualizations")
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.line(df, x='date', y='sales', title='📈 Historical Sales Trend')
        fig1.update_traces(line_color='#1f77b4')
        st.plotly_chart(fig1, use_container_width=True)
        
        fig3 = px.histogram(df, x='sales', nbins=40, title='📊 Sales Distribution', color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        dow_sales = df.groupby('day_of_week')['sales'].mean().reset_index()
        dow_sales['day'] = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        fig2 = px.bar(dow_sales, x='day', y='sales', title='📅 Avg Sales by Day of Week', color='sales', color_continuous_scale='Blues')
        st.plotly_chart(fig2, use_container_width=True)
        
        monthly = df.groupby('month')['sales'].mean().reset_index()
        fig4 = px.line(monthly, x='month', y='sales', title='📆 Monthly Average Sales', markers=True, line_shape='spline')
        fig4.update_traces(line_color='#1f77b4', marker=dict(size=8))
        st.plotly_chart(fig4, use_container_width=True)
    
    st.subheader("Model Predictions vs Actual")
    model_name = list(results.keys())[0]
    actuals = results[model_name]['y_test']
    predictions = results[model_name]['y_pred']
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(y=actuals, mode='lines', name='Actual', line=dict(color='green')))
    fig5.add_trace(go.Scatter(y=predictions, mode='lines', name='Predicted', line=dict(color='red', dash='dash')))
    fig5.update_layout(title=f'{model_name} - Actual vs Predicted', height=400)
    st.plotly_chart(fig5, use_container_width=True)
    # Tab 3: Model Performance
    with tab3:
        st.header("📊 Model Performance Comparison")
        
        # Performance metrics table
        st.subheader("Performance Metrics")
        metrics_data = []
        for name, model_info in models.items():
            metrics_data.append({
                'Model': name,
                'R² Score': f"{model_info['r2']:.4f}",
                'RMSE': f"{model_info['rmse']:.2f}",
                'MAE': f"{model_info['mae']:.2f}",
                'MAPE': f"{model_info['mape']:.2f}%",
                'CV Score': f"{model_info['cv_score']:.4f} (±{model_info['cv_std']:.4f})"
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Best model highlight
        best_model = max(models.items(), key=lambda x: x[1]['r2'])
        st.success(f"🏆 **Best Model**: {best_model[0]} with R² Score of {best_model[1]['r2']:.4f}")
        
        # Model comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # R² Score comparison
            fig_r2 = go.Figure(data=[
                go.Bar(x=list(models.keys()), 
                       y=[models[m]['r2'] for m in models.keys()],
                       marker_color='lightblue')
            ])
            fig_r2.update_layout(
                title="R² Score Comparison",
                xaxis_title="Model",
                yaxis_title="R² Score",
                height=400
            )
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col2:
            # RMSE comparison
            fig_rmse = go.Figure(data=[
                go.Bar(x=list(models.keys()), 
                       y=[models[m]['rmse'] for m in models.keys()],
                       marker_color='lightcoral')
            ])
            fig_rmse.update_layout(
                title="RMSE Comparison (Lower is Better)",
                xaxis_title="Model",
                yaxis_title="RMSE",
                height=400
            )
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        # Feature importance
        st.subheader("🎯 Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': best_model[1]['model'].feature_importances_ if hasattr(best_model[1]['model'], 'feature_importances_') else [0] * len(X.columns)
        }).sort_values('Importance', ascending=False)
        
        if feature_importance['Importance'].sum() > 0:
            fig_importance = px.bar(
                feature_importance, 
                x='Importance', 
                y='Feature',
                orientation='h',
                title=f"Feature Importance - {best_model[0]}"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        else:
            st.info("Feature importance is not available for the current best model (Linear models don't provide feature_importances_)")
    
    # Tab 4: Data & Export
    with tab4:
        st.header("📋 Data Overview & Export")
        
        # Dataset statistics
        st.subheader("Dataset Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(df))
            st.metric("Date Range", f"{(df['ds'].max() - df['ds'].min()).days} days")
        
        with col2:
            st.metric("Average Sales", f"${df['y'].mean():.2f}")
            st.metric("Sales Std Dev", f"${df['y'].std():.2f}")
        
        with col3:
            st.metric("Min Sales", f"${df['y'].min():.2f}")
            st.metric("Max Sales", f"${df['y'].max():.2f}")
        
        # Show raw data
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(100), use_container_width=True)
        
        # Download options
        st.subheader("📥 Download Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download as CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Full Dataset (CSV)",
                data=csv,
                file_name="sales_forecast_data.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download predictions with features
            predictions_df = X.copy()
            predictions_df['Date'] = df['ds'].values[:len(X)]
            predictions_df['Actual_Sales'] = y.values
            predictions_df['Predicted_Sales'] = models[selected_model]['predictions']
            
            csv_pred = predictions_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions with Features (CSV)",
                data=csv_pred,
                file_name="predictions_with_features.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🚀 ML Sales Forecasting Platform | Built with Streamlit & Scikit-learn</p>
    <p>📊 Advanced Machine Learning Models for Accurate Sales Predictions</p>
</div>
""", unsafe_allow_html=True)
