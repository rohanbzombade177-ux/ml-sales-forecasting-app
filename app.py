import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="ML Sales Forecasting", page_icon="📈", layout="wide")

# Title
st.title("📈 ML Sales Forecasting Application")
st.markdown("### Complete Standalone Machine Learning Web App")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Model Configuration")

# Generate synthetic sales data
@st.cache_data
def generate_data(n_samples=365):
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    
    # Create features
    day_of_week = dates.dayofweek
    month = dates.month
    day = dates.day
    
    # Generate sales with patterns
    trend = np.linspace(1000, 1500, n_samples)
    seasonal = 200 * np.sin(2 * np.pi * np.arange(n_samples) / 365)
    weekly = 100 * np.sin(2 * np.pi * np.arange(n_samples) / 7)
    noise = np.random.normal(0, 50, n_samples)
    
    sales = trend + seasonal + weekly + noise
    sales = np.maximum(sales, 500)  # Ensure positive
    
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'day_of_week': day_of_week,
        'month': month,
        'day': day,
        'day_of_year': dates.dayofyear
    })
    return df

# Train models
@st.cache_resource
def train_models(df):
    X = df[['day_of_week', 'month', 'day', 'day_of_year']].values
    y = df['sales'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train multiple models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    scores = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        trained_models[name] = model
        scores[name] = {
            'R²': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred)
        }
    
    return trained_models, scores, X_train, X_test, y_train, y_test

# Load data
df = generate_data()

# Train models
models, scores, X_train, X_test, y_train, y_test = train_models(df)

# Model selection
selected_model = st.sidebar.selectbox("Select Model", list(models.keys()))

# Show model performance
st.sidebar.markdown("### 📊 Model Performance")
for metric, value in scores[selected_model].items():
    st.sidebar.metric(metric, f"{value:.2f}")

# Main content in tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predictions", "📈 Visualizations", "📊 Model Comparison", "📁 Data"])

with tab1:
    st.header("Make Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Manual Input")
        
        input_date = st.date_input("Select Date", value=datetime.now())
        
        dow = input_date.weekday()
        month_val = input_date.month
        day_val = input_date.day
        doy = input_date.timetuple().tm_yday
        
        st.write(f"Day of Week: {input_date.strftime('%A')}")
        st.write(f"Month: {input_date.strftime('%B')}")
        
        if st.button("🔮 Predict Sales", type="primary"):
            features = np.array([[dow, month_val, day_val, doy]])
            prediction = models[selected_model].predict(features)[0]
            
            st.success(f"### Predicted Sales: ${prediction:,.2f}")
            st.info(f"Using {selected_model}")
            
            # Confidence interval
            rmse = scores[selected_model]['RMSE']
            lower = prediction - 1.96 * rmse
            upper = prediction + 1.96 * rmse
            
            st.write(f"95% Confidence Interval: ${lower:,.2f} - ${upper:,.2f}")
    
    with col2:
        st.subheader("Batch Predictions")
        
        n_days = st.slider("Forecast next N days", 1, 30, 7)
        
        if st.button("Generate Forecast"):
            future_dates = pd.date_range(start=datetime.now(), periods=n_days, freq='D')
            
            forecasts = []
            for date in future_dates:
                dow = date.dayofweek
                month_val = date.month
                day_val = date.day
                doy = date.timetuple().tm_yday
                
                features = np.array([[dow, month_val, day_val, doy]])
                pred = models[selected_model].predict(features)[0]
                forecasts.append(pred)
            
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Sales': forecasts
            })
            
            st.dataframe(forecast_df.style.format({'Predicted Sales': '${:,.2f}'}), use_container_width=True)
            
            # Plot forecast
            fig = px.line(forecast_df, x='Date', y='Predicted Sales',
                         title=f'{n_days}-Day Sales Forecast',
                         markers=True)
            fig.update_layout(hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Data Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.line(df, x='date', y='sales', title='Historical Sales Trend')
        st.plotly_chart(fig1, use_container_width=True)
        
        fig3 = px.histogram(df, x='sales', nbins=50, title='Sales Distribution')
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        dow_sales = df.groupby('day_of_week')['sales'].mean().reset_index()
        dow_sales['day_name'] = dow_sales['day_of_week'].map({
            0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'
        })
        fig2 = px.bar(dow_sales, x='day_name', y='sales',
                     title='Average Sales by Day of Week')
        st.plotly_chart(fig2, use_container_width=True)
        
        monthly_sales = df.groupby('month')['sales'].mean().reset_index()
        fig4 = px.line(monthly_sales, x='month', y='sales',
                      title='Average Sales by Month', markers=True)
        st.plotly_chart(fig4, use_container_width=True)

with tab3:
    st.header("Model Comparison")
    
    comparison_data = []
    for model_name, metrics in scores.items():
        comparison_data.append({
            'Model': model_name,
            'R² Score': metrics['R²'],
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Best R² Score", 
                 f"{comparison_df['R² Score'].max():.4f}",
                 f"{comparison_df.loc[comparison_df['R² Score'].idxmax(), 'Model']}")
    
    with col2:
        st.metric("Lowest RMSE",
                 f"{comparison_df['RMSE'].min():.2f}",
                 f"{comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Model']}")
    
    with col3:
        st.metric("Lowest MAE",
                 f"{comparison_df['MAE'].min():.2f}",
                 f"{comparison_df.loc[comparison_df['MAE'].idxmin(), 'Model']}")
    
    st.dataframe(comparison_df, use_container_width=True)
    
    fig = go.Figure()
    for metric in ['R² Score', 'RMSE', 'MAE']:
        fig.add_trace(go.Bar(name=metric, x=comparison_df['Model'], y=comparison_df[metric]))
    
    fig.update_layout(barmode='group', title='Model Performance Comparison')
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Dataset")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Date Range", f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    col3.metric("Avg Sales", f"${df['sales'].mean():,.2f}")
    
    st.dataframe(df.style.format({'sales': '${:,.2f}'}), use_container_width=True)
    
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name='sales_data.csv',
        mime='text/csv'
    )

st.markdown("---")
st.markdown("**Built with ❤️ using Streamlit | Powered by scikit-learn**")
