# Additional Flight Data Analysis for Airlines
# Building on your existing association rule mining

import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime, timedelta
# from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# Load your data
df = pd.read_csv('./data/cleaned_flight_data.csv')
air = df[df['airport'] == 'فرودگاه مهرآباد'].copy()

# ==========================================
# 1. OPERATIONAL PERFORMANCE ANALYTICS
# ==========================================

def operational_performance_analysis(df):
    """Comprehensive operational performance metrics"""
    
    # On-time Performance (OTP) by airline
    otp_by_airline = df.groupby('airline').agg({
        'Early_OnTime_Late_Indicator': lambda x: (x == 'On Time').mean() * 100,
        'delay_minutes': ['mean', 'median', 'std'],
        'flight_number': 'count'
    }).round(2)
    
    otp_by_airline.columns = ['OTP_Percentage', 'Avg_Delay_Minutes', 'Median_Delay', 'Delay_StdDev', 'Total_Flights']
    
    # Schedule reliability by route
    route_performance = df.groupby('destination_or_origin').agg({
        'Early_OnTime_Late_Indicator': lambda x: (x == 'On Time').mean() * 100,
        'delay_minutes': 'mean',
        'flight_number': 'count'
    }).round(2)
    
    route_performance.columns = ['Route_OTP', 'Avg_Route_Delay', 'Flight_Count']
    
    # Aircraft utilization and performance
    aircraft_performance = df.groupby('aircraft').agg({
        'Early_OnTime_Late_Indicator': lambda x: (x == 'On Time').mean() * 100,
        'delay_minutes': 'mean',
        'flight_number': 'count'
    }).round(2)
    
    aircraft_performance.columns = ['Aircraft_OTP', 'Avg_Aircraft_Delay', 'Flights_Count']
    
    return otp_by_airline, route_performance, aircraft_performance

# ==========================================
# 2. DELAY PATTERN ANALYSIS
# ==========================================

def delay_pattern_analysis(df):
    """Analyze delay patterns by time, season, and operational factors"""
    
    # Delay distribution by time of day
    time_delays = df.groupby('scheduled_time_of_day').agg({
        'delay_minutes': ['mean', 'median', 'count'],
        'Early_OnTime_Late_Indicator': lambda x: (x == 'Late').mean() * 100
    }).round(2)
    
    # Seasonal delay patterns
    seasonal_delays = df.groupby('scheduled_season').agg({
        'delay_minutes': ['mean', 'median'],
        'Early_OnTime_Late_Indicator': lambda x: (x == 'Late').mean() * 100
    }).round(2)
    
    # Weekend vs weekday analysis
    weekend_analysis = df.groupby('is_weekend').agg({
        'delay_minutes': ['mean', 'median'],
        'Early_OnTime_Late_Indicator': lambda x: (x == 'Late').mean() * 100
    }).round(2)
    
    # Holiday impact analysis
    holiday_impact = df.groupby(['is_Nourooz_4', 'is_Nourooz_13', 'Normal_holiday']).agg({
        'delay_minutes': 'mean',
        'Early_OnTime_Late_Indicator': lambda x: (x == 'Late').mean() * 100
    }).round(2)
    
    return time_delays, seasonal_delays, weekend_analysis, holiday_impact

# ==========================================
# 3. PREDICTIVE DELAY MODELING
# ==========================================

def delay_risk_scoring(df):
    """Create delay risk scores for different operational scenarios"""
    
    # Calculate delay probability by combination of factors
    risk_factors = df.groupby(['airline', 'scheduled_time_of_day', 'scheduled_day_of_week']).agg({
        'Early_OnTime_Late_Indicator': lambda x: (x == 'Late').mean() * 100,
        'delay_minutes': 'mean',
        'flight_number': 'count'
    }).round(2)
    
    risk_factors.columns = ['Late_Probability', 'Avg_Delay_When_Late', 'Sample_Size']
    
    # Filter for statistically significant samples
    risk_factors = risk_factors[risk_factors['Sample_Size'] >= 10]
    
    # Create risk score (probability * severity)
    risk_factors['Risk_Score'] = (risk_factors['Late_Probability'] * 
                                 risk_factors['Avg_Delay_When_Late']) / 100
    
    return risk_factors.sort_values('Risk_Score', ascending=False)

# ==========================================
# 4. RESOURCE OPTIMIZATION ANALYSIS
# ==========================================

def resource_optimization_analysis(df):
    """Analyze resource utilization and optimization opportunities"""
    
    # Gate/Counter utilization analysis
    counter_analysis = df.groupby('counter').agg({
        'delay_minutes': 'mean',
        'Early_OnTime_Late_Indicator': lambda x: (x == 'Late').mean() * 100,
        'flight_number': 'count'
    }).round(2)
    
    # Aircraft rotation efficiency
    aircraft_rotation = df.groupby(['register', 'scheduled_day_of_week']).agg({
        'flight_number': 'count',
        'delay_minutes': 'mean'
    }).round(2)
    
    # Peak hour analysis
    df['scheduled_hour'] = pd.to_datetime(df['scheduled_time']).dt.hour
    peak_hours = df.groupby('scheduled_hour').agg({
        'flight_number': 'count',
        'delay_minutes': 'mean',
        'Early_OnTime_Late_Indicator': lambda x: (x == 'Late').mean() * 100
    }).round(2)
    
    return counter_analysis, aircraft_rotation, peak_hours

# ==========================================
# 5. COST IMPACT ANALYSIS
# ==========================================

def cost_impact_analysis(df, avg_cost_per_minute=1000):
    """Calculate cost impact of delays (assuming cost per minute)"""
    
    # Total delay cost by airline
    airline_costs = df.groupby('airline').agg({
        'delay_minutes': ['sum', 'mean', 'count']
    }).round(2)
    
    airline_costs.columns = ['Total_Delay_Minutes', 'Avg_Delay_Minutes', 'Flight_Count']
    airline_costs['Estimated_Cost'] = airline_costs['Total_Delay_Minutes'] * avg_cost_per_minute
    
    # Monthly cost trends
    df['month'] = pd.to_datetime(df['scheduled_time']).dt.to_period('M')
    monthly_costs = df.groupby('month').agg({
        'delay_minutes': 'sum',
        'flight_number': 'count'
    }).round(2)
    
    monthly_costs['Estimated_Monthly_Cost'] = monthly_costs['delay_minutes'] * avg_cost_per_minute
    
    return airline_costs, monthly_costs

# ==========================================
# 6. COMPETITIVE BENCHMARKING
# ==========================================

def competitive_analysis(df):
    """Compare airline performance against industry benchmarks"""
    
    # Industry benchmarks
    industry_metrics = df.agg({
        'delay_minutes': 'mean',
        'Early_OnTime_Late_Indicator': lambda x: (x == 'On Time').mean() * 100
    }).round(2)
    
    # Airline performance vs industry
    airline_vs_industry = df.groupby('airline').agg({
        'delay_minutes': 'mean',
        'Early_OnTime_Late_Indicator': lambda x: (x == 'On Time').mean() * 100,
        'flight_number': 'count'
    }).round(2)
    
    airline_vs_industry.columns = ['Avg_Delay', 'OTP_Rate', 'Flight_Count']
    
    # Calculate performance gaps
    airline_vs_industry['Delay_Gap_vs_Industry'] = (airline_vs_industry['Avg_Delay'] - 
                                                   industry_metrics['delay_minutes'])
    airline_vs_industry['OTP_Gap_vs_Industry'] = (airline_vs_industry['OTP_Rate'] - 
                                                 industry_metrics['Early_OnTime_Late_Indicator'])
    
    return airline_vs_industry, industry_metrics

# ==========================================
# 7. ROUTE NETWORK ANALYSIS
# ==========================================

def route_network_analysis(df):
    """Analyze route network performance and optimization"""
    
    # Route profitability proxy (based on punctuality and frequency)
    route_metrics = df.groupby('destination_or_origin').agg({
        'flight_number': 'count',
        'delay_minutes': ['mean', 'std'],
        'Early_OnTime_Late_Indicator': lambda x: (x == 'On Time').mean() * 100
    }).round(2)
    
    route_metrics.columns = ['Frequency', 'Avg_Delay', 'Delay_Variability', 'OTP_Rate']
    
    # Route performance score
    route_metrics['Performance_Score'] = (route_metrics['OTP_Rate'] * 
                                        route_metrics['Frequency']) / 100
    
    # Most problematic routes
    problematic_routes = route_metrics[route_metrics['OTP_Rate'] < 70].sort_values('Avg_Delay', ascending=False)
    
    return route_metrics, problematic_routes

# ==========================================
# 8. PASSENGER IMPACT ANALYSIS
# ==========================================

def passenger_impact_analysis(df):
    """Analyze passenger experience and satisfaction metrics"""
    
    # Severe delay analysis (>30 minutes)
    severe_delays = df[df['delay_minutes'] > 30].groupby('airline').agg({
        'flight_number': 'count',
        'delay_minutes': 'mean'
    }).round(2)
    
    severe_delays.columns = ['Severe_Delay_Count', 'Avg_Severe_Delay']
    
    # Passenger disruption by time of day
    disruption_by_time = df.groupby('scheduled_time_of_day').agg({
        'delay_minutes': lambda x: (x > 30).sum(),  # Severe delays
        'flight_number': 'count'
    }).round(2)
    
    disruption_by_time.columns = ['Severe_Delays', 'Total_Flights']
    disruption_by_time['Severe_Delay_Rate'] = (disruption_by_time['Severe_Delays'] / 
                                              disruption_by_time['Total_Flights'] * 100)
    
    return severe_delays, disruption_by_time

# ==========================================
# RUN ALL ANALYSES
# ==========================================

# Execute all analyses
print("=== OPERATIONAL PERFORMANCE ANALYTICS ===")
otp_airline, route_perf, aircraft_perf = operational_performance_analysis(air)
print("Top 5 Airlines by OTP:")
print(otp_airline.sort_values('OTP_Percentage', ascending=False).head())

print("\n=== DELAY PATTERN ANALYSIS ===")
time_delays, seasonal_delays, weekend_analysis, holiday_impact = delay_pattern_analysis(air)
print("Delay patterns by time of day:")
print(time_delays)

print("\n=== DELAY RISK SCORING ===")
risk_scores = delay_risk_scoring(air)
print("Top 10 highest risk scenarios:")
print(risk_scores.head(10))

print("\n=== RESOURCE OPTIMIZATION ===")
counter_analysis, aircraft_rotation, peak_hours = resource_optimization_analysis(air)
print("Peak hour analysis:")
print(peak_hours.sort_values('flight_number', ascending=False).head())

print("\n=== COST IMPACT ANALYSIS ===")
airline_costs, monthly_costs = cost_impact_analysis(air)
print("Airlines with highest delay costs:")
print(airline_costs.sort_values('Estimated_Cost', ascending=False).head())

print("\n=== COMPETITIVE BENCHMARKING ===")
airline_vs_industry, industry_metrics = competitive_analysis(air)
print("Airline performance vs industry:")
print(airline_vs_industry.sort_values('OTP_Gap_vs_Industry', ascending=False).head())

print("\n=== ROUTE NETWORK ANALYSIS ===")
route_metrics, problematic_routes = route_network_analysis(air)
print("Most problematic routes:")
print(problematic_routes.head())

print("\n=== PASSENGER IMPACT ANALYSIS ===")
severe_delays, disruption_by_time = passenger_impact_analysis(air)
print("Severe delay analysis by airline:")
print(severe_delays.sort_values('Severe_Delay_Count', ascending=False).head())

# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================

def create_airline_dashboard(df):
    """Create visualizations for airline performance dashboard"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. OTP by Airline
    otp_data = df.groupby('airline')['Early_OnTime_Late_Indicator'].apply(
        lambda x: (x == 'On Time').mean() * 100
    ).sort_values(ascending=False)
    
    otp_data.plot(kind='bar', ax=axes[0,0], color='skyblue')
    axes[0,0].set_title('On-Time Performance by Airline')
    axes[0,0].set_ylabel('OTP Percentage')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Delay by Time of Day
    time_delay = df.groupby('scheduled_time_of_day')['delay_minutes'].mean()
    time_delay.plot(kind='bar', ax=axes[0,1], color='lightcoral')
    axes[0,1].set_title('Average Delay by Time of Day')
    axes[0,1].set_ylabel('Average Delay (minutes)')
    
    # 3. Seasonal Patterns
    seasonal_data = df.groupby('scheduled_season')['delay_minutes'].mean()
    seasonal_data.plot(kind='bar', ax=axes[1,0], color='lightgreen')
    axes[1,0].set_title('Average Delay by Season')
    axes[1,0].set_ylabel('Average Delay (minutes)')
    
    # 4. Aircraft Performance
    aircraft_otp = df.groupby('aircraft')['Early_OnTime_Late_Indicator'].apply(
        lambda x: (x == 'On Time').mean() * 100
    ).sort_values(ascending=False).head(10)
    
    aircraft_otp.plot(kind='bar', ax=axes[1,1], color='gold')
    axes[1,1].set_title('Top 10 Aircraft by OTP')
    axes[1,1].set_ylabel('OTP Percentage')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# Create dashboard
create_airline_dashboard(air)