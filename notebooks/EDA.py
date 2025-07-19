# %% [markdown]
# ## Association Mining with FP-Growth
# 
# This notebook demonstrates how to use the FP-Growth algorithm to find frequent itemsets and association rules in the flight delay dataset.

# %%
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import warnings
warnings.filterwarnings("ignore")

# %%
df = pd.read_csv('../data/cleaned_flight_data.csv')
df.head()

# %%
air = df[df['airport'] == 'فرودگاه مهرآباد'].copy()
print(f"percentage: {(air.shape[0])/df.shape[0]}")

# %%
df.columns

# %%
# Select features for association mining
# df_assoc = df[['airline', 'destination_or_origin', 'aircraft', 'airport', 'scheduled_day_of_week', 'scheduled_season', 'scheduled_time_of_day', 'Early_OnTime_Late_Indicator']].copy()
df_assoc = air[['airline', 'destination_or_origin', 'aircraft', 'scheduled_day_of_week', 'scheduled_season', 'scheduled_time_of_day', 'Early_OnTime_Late_Indicator']].copy()
# Drop rows with missing values
df_assoc.dropna(inplace=True)

df.head()

# %%
# Convert the dataframe into a list of transactions
transactions = df_assoc.to_numpy().tolist()

# %%
# Use TransactionEncoder to transform the data into a one-hot encoded format
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

# %%
# Run the FP-Growth algorithm
frequent_itemsets = fpgrowth(df_onehot, min_support=0.01, use_colnames=True)

# %%
# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Display the rules
rules.sort_values(by='lift', ascending=False).head()

# %%
late_rules = rules[rules['consequents'].apply(lambda x: 'Late' in x)].copy()
late_rules.shape

# %%
late_rules.sort_values(by='confidence', ascending=False).head(5)

# %%
late_rules_freeze = rules[rules['consequents'] == frozenset({'Late'})].copy()
late_rules_freeze.shape

# %%
late_rules_freeze = late_rules_freeze[late_rules_freeze['antecedents'].apply(lambda x: 'Spring' not in x)].copy()
late_rules_freeze.shape

# %%
late_rules_freeze.sort_values(by='confidence', ascending= False).head(5)

# %%
late_rules_freeze['antecedents_len'] = late_rules_freeze['antecedents'].apply(lambda x: len(x))

# %%
late_rules_freeze.sort_values(by=['antecedents_len'], ascending =[True]).head(5)

# %% [markdown]
# ## **By day**

# %%
days_of_the_week = set(df['scheduled_day_of_week'].unique())

def only_things(antecedents, _set):
    return antecedents.issubset(_set)

# %%
late_rules_by_day = late_rules_freeze[late_rules_freeze['antecedents'].apply(only_things, args = (days_of_the_week, ))].copy()
late_rules_by_day.shape
late_rules_by_day

# %% [markdown]
# ## **By destination**

# %%
destinations = set(df['destination_or_origin'].unique())

# %%
late_rules_by_dest = late_rules_freeze[late_rules_freeze['antecedents'].apply(only_things, args = (destinations, ))].copy()
late_rules_by_dest.shape

# %%
late_rules_by_dest.sort_values(by='confidence', ascending = False).head(5)

# %% [markdown]
# ## **by Airlines only**

# %%
airlines = set(df['airline'].unique())

# %%
late_rules_by_airline = late_rules_freeze[late_rules_freeze['antecedents'].apply(only_things, args = (airlines, ))].copy()
late_rules_by_airline.shape

# %%
late_rules_by_airline

# %% [markdown]
# ### **by Aircraft**

# %%
aircrafts = set(df['aircraft'].unique())

# %%
late_rules_by_aircraft = late_rules_freeze[late_rules_freeze['antecedents'].apply(only_things, args = (aircrafts, ))].copy()
late_rules_by_aircraft.shape

# %%
late_rules_by_aircraft

# %% [markdown]
# ## **by Time**

# %%
times = set(df['scheduled_time_of_day'].unique())

# %%
late_rules_by_time = late_rules_freeze[late_rules_freeze['antecedents'].apply(only_things, args = (times, ))]
late_rules_by_time.shape

# %%
late_rules_by_time

# %% [markdown]
# Here are the key additional analyses you can provide to airlines:
# 1. Operational Performance Analytics
# 
# On-Time Performance (OTP) metrics by airline, route, and aircraft
# Schedule reliability indicators
# Aircraft utilization efficiency metrics
# 
# 2. Delay Pattern Analysis
# 
# Time-based patterns: delays by hour, day of week, season
# Holiday impact analysis: performance during special periods
# Weekend vs weekday comparison
# 
# 3. Predictive Delay Risk Scoring
# 
# Risk assessment for different operational scenarios
# Probability-based scoring for delay likelihood
# Early warning systems for high-risk flights
# 
# 4. Resource Optimization
# 
# Gate/counter utilization analysis
# Aircraft rotation efficiency
# Peak hour capacity planning
# 
# 5. Cost Impact Analysis
# 
# Financial impact of delays (estimated costs)
# Monthly cost trends
# ROI analysis for operational improvements
# 
# 6. Competitive Benchmarking
# 
# Industry comparison metrics
# Performance gaps identification
# Market positioning analysis
# 
# 7. Route Network Analysis
# 
# Route profitability assessment
# Network optimization recommendations
# Problematic routes identification
# 
# 8. Passenger Impact Analysis
# 
# Severe delay analysis (>30 minutes)
# Passenger disruption patterns
# Service quality metrics
# 
# Key Benefits for Airlines:
# Strategic Planning: Use seasonal and time-based patterns for capacity planning
# Operational Efficiency: Identify bottlenecks and optimization opportunities
# Cost Management: Quantify delay costs and prioritize improvements
# Competitive Advantage: Benchmark against industry standards
# Customer Experience: Reduce passenger disruptions and improve satisfaction
# Risk Management: Proactive identification of high-risk scenarios
# These analyses complement your association rule mining by providing quantitative metrics, predictive insights, and actionable recommendations that airlines can use for operational improvements and strategic decision-making.RetryClaude can make mistakes. Please double-check responses. Sonnet 4


# %% [markdown]
# ## On-Time Performance Analysis
# %%
# Overall airline performance metrics
performance_metrics = air.groupby('airline').agg({
    'delay_minutes': ['mean', 'median', 'std', 'count'],
    'Early_OnTime_Late_Indicator': lambda x: (x == 'OnTime').sum() / len(x) * 100
}).round(2)

performance_metrics.columns = ['avg_delay', 'median_delay', 'delay_std', 'flight_count', 'ontime_percentage']
performance_metrics.sort_values('ontime_percentage', ascending=False)

# %%
# Performance by destination
dest_performance = air.groupby('destination_or_origin').agg({
    'delay_minutes': 'mean',
    'Early_OnTime_Late_Indicator': lambda x: (x == 'Late').sum() / len(x) * 100
}).round(2)
dest_performance.columns = ['avg_delay', 'late_percentage']
dest_performance.sort_values('late_percentage', ascending=False).head(10)

# %%
# Performance by aircraft type
aircraft_performance = air.groupby('aircraft').agg({
    'delay_minutes': ['mean', 'count'],
    'Early_OnTime_Late_Indicator': lambda x: (x == 'OnTime').sum() / len(x) * 100
}).round(2)
aircraft_performance.columns = ['avg_delay', 'flight_count', 'ontime_percentage']
aircraft_performance[aircraft_performance['flight_count'] >= 10]  # Filter for statistical significance
# %% [markdown]
# ## Temporal Pattern Analysis

# %%
# Peak delay hours analysis
hourly_delays = air.groupby('Scheduled_Hour_of_Day').agg({
    'delay_minutes': 'mean',
    'Early_OnTime_Late_Indicator': lambda x: (x == 'Late').sum() / len(x) * 100
}).round(2)
hourly_delays.columns = ['avg_delay', 'late_percentage']
hourly_delays.sort_values('late_percentage', ascending=False)

# %%
# Weekly pattern analysis
weekly_pattern = air.groupby('scheduled_day_of_week').agg({
    'delay_minutes': 'mean',
    'Early_OnTime_Late_Indicator': lambda x: (x == 'Late').sum() / len(x) * 100,
    'flight_number': 'count'
}).round(2)
weekly_pattern.columns = ['avg_delay', 'late_percentage', 'flight_volume']
weekly_pattern

# %%
# Seasonal impact analysis
seasonal_analysis = air.groupby('scheduled_season').agg({
    'delay_minutes': ['mean', 'median'],
    'Early_OnTime_Late_Indicator': lambda x: (x == 'Late').sum() / len(x) * 100
}).round(2)
seasonal_analysis.columns = ['avg_delay', 'median_delay', 'late_percentage']
seasonal_analysis

# %% [markdown]
# ## Operational Efficiency Analysis

# %%
# Counter utilization and efficiency
counter_analysis = air.groupby('counter').agg({
    'delay_minutes': 'mean',
    'flight_number': 'count',
    'Early_OnTime_Late_Indicator': lambda x: (x == 'OnTime').sum() / len(x) * 100
}).round(2)
counter_analysis.columns = ['avg_delay', 'flight_count', 'ontime_percentage']
counter_analysis.sort_values('flight_count', ascending=False).head(10)

# %%
# Holiday impact analysis
holiday_impact = air.groupby(['Normal_holiday', 'is_weekend']).agg({
    'delay_minutes': 'mean',
    'Early_OnTime_Late_Indicator': lambda x: (x == 'Late').sum() / len(x) * 100
}).round(2)
holiday_impact.columns = ['avg_delay', 'late_percentage']
holiday_impact

# %%
# Flight frequency vs delay correlation
airline_freq_delay = air.groupby('airline').agg({
    'flight_number': 'count',
    'delay_minutes': 'mean'
}).round(2)
airline_freq_delay.columns = ['flight_frequency', 'avg_delay']
airline_freq_delay['efficiency_score'] = airline_freq_delay['flight_frequency'] / (airline_freq_delay['avg_delay'] + 1)
airline_freq_delay.sort_values('efficiency_score', ascending=False)


# %% [markdown]
# ## Risk Assessment and Predictive Analysis

# %%
# High-risk combinations (airline + destination + time)
risk_combinations = air.groupby(['airline', 'destination_or_origin', 'scheduled_time_of_day']).agg({
    'delay_minutes': ['mean', 'count'],
    'Early_OnTime_Late_Indicator': lambda x: (x == 'Late').sum() / len(x) * 100
}).round(2)
risk_combinations.columns = ['avg_delay', 'flight_count', 'late_percentage']
high_risk = risk_combinations[(risk_combinations['flight_count'] >= 5) & (risk_combinations['late_percentage'] > 50)]
high_risk.sort_values('late_percentage', ascending=False)

# %%
# Delay propagation analysis (same day consecutive flights)
air_sorted = air.sort_values(['airline', 'scheduled_datetime'])
air_sorted['prev_delay'] = air_sorted.groupby('airline')['delay_minutes'].shift(1)
delay_propagation = air_sorted.groupby(pd.cut(air_sorted['prev_delay'], bins=[-1, 0, 30, 60, float('inf')], labels=['OnTime', 'Short', 'Medium', 'Long']))['delay_minutes'].mean()
delay_propagation

# %%
# Weather/seasonal delay patterns
weather_proxy = air.groupby(['scheduled_season', 'scheduled_time_of_day']).agg({
    'delay_minutes': 'mean',
    'Early_OnTime_Late_Indicator': lambda x: (x == 'Late').sum() / len(x) * 100
}).round(2)
weather_proxy.columns = ['avg_delay', 'late_percentage']
weather_proxy.sort_values('late_percentage', ascending=False)


# %% [markdown]
# ## Network and Route Analysis

# %%
# Route efficiency analysis
route_efficiency = air.groupby('destination_or_origin').agg({
    'delay_minutes': ['mean', 'std', 'count'],
    'airline': 'nunique',
    'Early_OnTime_Late_Indicator': lambda x: (x == 'OnTime').sum() / len(x) * 100
}).round(2)
route_efficiency.columns = ['avg_delay', 'delay_variability', 'flight_frequency', 'competing_airlines', 'ontime_percentage']
route_efficiency['route_score'] = route_efficiency['ontime_percentage'] / (route_efficiency['delay_variability'] + 1)
route_efficiency.sort_values('route_score', ascending=False)

# %%
# Airline market share by destination
market_share = air.groupby(['destination_or_origin', 'airline']).size().unstack(fill_value=0)
market_share_pct = market_share.div(market_share.sum(axis=1), axis=0) * 100
market_share_pct.round(2)

# %%
# Aircraft utilization patterns
aircraft_utilization = air.groupby(['aircraft', 'airline']).agg({
    'flight_number': 'count',
    'delay_minutes': 'mean',
    'destination_or_origin': 'nunique'
}).round(2)
aircraft_utilization.columns = ['total_flights', 'avg_delay', 'routes_served']
aircraft_utilization['utilization_efficiency'] = aircraft_utilization['total_flights'] / (aircraft_utilization['avg_delay'] + 1)
aircraft_utilization.sort_values('utilization_efficiency', ascending=False)

# %% [markdown]
# ## Competitive Intelligence

# %%
# Head-to-head airline comparison on same routes
route_competition = air.groupby(['destination_or_origin', 'airline']).agg({
    'delay_minutes': 'mean',
    'Early_OnTime_Late_Indicator': lambda x: (x == 'OnTime').sum() / len(x) * 100,
    'flight_number': 'count'
}).round(2)
route_competition.columns = ['avg_delay', 'ontime_percentage', 'flight_count']

# Get routes with multiple airlines
competitive_routes = route_competition.groupby('destination_or_origin').filter(lambda x: len(x) > 1)
competitive_routes.sort_values(['destination_or_origin', 'ontime_percentage'], ascending=[True, False])

# %%
# Airline performance benchmarking
airline_benchmark = air.groupby('airline').agg({
    'delay_minutes': ['mean', 'median', 'std'],
    'Early_OnTime_Late_Indicator': lambda x: (x == 'OnTime').sum() / len(x) * 100,
    'destination_or_origin': 'nunique',
    'flight_number': 'count'
}).round(2)
airline_benchmark.columns = ['avg_delay', 'median_delay', 'delay_consistency', 'ontime_percentage', 'routes_served', 'total_flights']
airline_benchmark['overall_score'] = (airline_benchmark['ontime_percentage'] * 0.4 + 
                                    (100 - airline_benchmark['avg_delay']) * 0.3 + 
                                    (100 - airline_benchmark['delay_consistency']) * 0.3)
airline_benchmark.sort_values('overall_score', ascending=False)

# %% [markdown]
# ## Advanced Cost Impact Analysis

# %%
# Estimate financial impact of delays (using industry averages)
COST_PER_MINUTE = 50  # USD per minute of delay (industry average)
PASSENGER_COMPENSATION_THRESHOLD = 120  # minutes

air['estimated_cost'] = air['delay_minutes'] * COST_PER_MINUTE
air['compensation_required'] = air['delay_minutes'] > PASSENGER_COMPENSATION_THRESHOLD

cost_analysis = air.groupby('airline').agg({
    'estimated_cost': ['sum', 'mean'],
    'compensation_required': 'sum',
    'delay_minutes': 'count'
}).round(2)
cost_analysis.columns = ['total_cost_estimate', 'avg_cost_per_flight', 'compensation_cases', 'total_flights']
cost_analysis['cost_per_flight'] = cost_analysis['total_cost_estimate'] / cost_analysis['total_flights']
cost_analysis.sort_values('total_cost_estimate', ascending=False)

# %%
# Monthly cost trends
air['month'] = air['scheduled_datetime'].dt.month
monthly_costs = air.groupby(['month', 'airline']).agg({
    'estimated_cost': 'sum',
    'delay_minutes': 'mean'
}).round(2)
monthly_costs.columns = ['monthly_cost', 'avg_delay']
monthly_costs.reset_index().pivot(index='month', columns='airline', values='monthly_cost')

# %% [markdown]
# ## Severe Delay Analysis (Passenger Impact)

# %%
# Severe delay categorization
def categorize_delay(minutes):
    if minutes <= 15:
        return 'Acceptable'
    elif minutes <= 30:
        return 'Minor'
    elif minutes <= 60:
        return 'Moderate'
    elif minutes <= 120:
        return 'Severe'
    else:
        return 'Critical'

air['delay_category'] = air['delay_minutes'].apply(categorize_delay)

# Passenger impact analysis
passenger_impact = air.groupby(['airline', 'delay_category']).size().unstack(fill_value=0)
passenger_impact_pct = passenger_impact.div(passenger_impact.sum(axis=1), axis=0) * 100
passenger_impact_pct.round(2)

# %%
# Critical delay incidents (>2 hours)
critical_delays = air[air['delay_minutes'] > 120].copy()
critical_analysis = critical_delays.groupby(['airline', 'destination_or_origin']).agg({
    'delay_minutes': ['count', 'mean'],
    'flight_number': 'count'
}).round(2)
critical_analysis.columns = ['critical_incidents', 'avg_critical_delay', 'affected_flights']
critical_analysis.sort_values('critical_incidents', ascending=False)

# %% [markdown]
# ## Operational Bottleneck Analysis

# %%
# Counter/Gate bottleneck analysis
counter_bottleneck = air.groupby('counter').agg({
    'delay_minutes': ['mean', 'std', 'count'],
    'Early_OnTime_Late_Indicator': lambda x: (x == 'Late').sum() / len(x) * 100
}).round(2)
counter_bottleneck.columns = ['avg_delay', 'delay_variability', 'flight_volume', 'late_percentage']
counter_bottleneck['bottleneck_score'] = (counter_bottleneck['late_percentage'] * 
                                         counter_bottleneck['delay_variability'] / 100)
counter_bottleneck.sort_values('bottleneck_score', ascending=False).head(10)

# %%
# Peak hour capacity analysis
peak_analysis = air.groupby('Scheduled_Hour_of_Day').agg({
    'flight_number': 'count',
    'delay_minutes': 'mean',
    'Early_OnTime_Late_Indicator': lambda x: (x == 'Late').sum() / len(x) * 100
}).round(2)
peak_analysis.columns = ['flight_volume', 'avg_delay', 'late_percentage']
peak_analysis['capacity_stress'] = peak_analysis['flight_volume'] * peak_analysis['late_percentage'] / 100
peak_analysis.sort_values('capacity_stress', ascending=False)

# %% [markdown]
# ## Predictive Delay Risk Scoring

# %%
# Risk scoring based on multiple factors
def calculate_risk_score(row):
    score = 0
    
    # Time-based risk
    if row['Scheduled_Hour_of_Day'] in [6, 7, 8, 18, 19, 20]:  # Peak hours
        score += 30
    
    # Day-based risk
    if row['scheduled_day_of_week'] in ['Friday', 'Saturday']:
        score += 20
    
    # Holiday risk
    if row['Normal_holiday'] == 1:
        score += 25
    
    # Seasonal risk
    if row['scheduled_season'] == 'Winter':
        score += 15
    
    return score

air['risk_score'] = air.apply(calculate_risk_score, axis=1)

# Risk validation
risk_validation = air.groupby(pd.cut(air['risk_score'], bins=[0, 25, 50, 75, 100], labels=['Low', 'Medium', 'High', 'Very High'])).agg({
    'Early_OnTime_Late_Indicator': lambda x: (x == 'Late').sum() / len(x) * 100,
    'delay_minutes': 'mean'
}).round(2)
risk_validation.columns = ['actual_late_percentage', 'actual_avg_delay']
risk_validation

# %%
# Airline-specific risk profiles
airline_risk = air.groupby('airline').agg({
    'risk_score': 'mean',
    'delay_minutes': 'mean',
    'Early_OnTime_Late_Indicator': lambda x: (x == 'Late').sum() / len(x) * 100
}).round(2)
airline_risk.columns = ['avg_risk_score', 'avg_delay', 'late_percentage']
airline_risk['risk_accuracy'] = abs(airline_risk['avg_risk_score'] - airline_risk['late_percentage'])
airline_risk.sort_values('avg_risk_score', ascending=False)

# %% [markdown]
# ## Schedule Optimization Analysis

# %%
# Optimal scheduling windows
schedule_optimization = air.groupby(['scheduled_day_of_week', 'Scheduled_Hour_of_Day']).agg({
    'delay_minutes': 'mean',
    'Early_OnTime_Late_Indicator': lambda x: (x == 'OnTime').sum() / len(x) * 100,
    'flight_number': 'count'
}).round(2)
schedule_optimization.columns = ['avg_delay', 'ontime_percentage', 'flight_count']

# Find optimal time slots (low delay, high on-time performance)
optimal_slots = schedule_optimization[
    (schedule_optimization['avg_delay'] < schedule_optimization['avg_delay'].median()) &
    (schedule_optimization['ontime_percentage'] > schedule_optimization['ontime_percentage'].median())
]
optimal_slots.sort_values('ontime_percentage', ascending=False)

# %%
# Buffer time analysis
air['buffer_effectiveness'] = air['delay_minutes'] < 15  # Within acceptable range
buffer_analysis = air.groupby(['airline', 'scheduled_time_of_day']).agg({
    'buffer_effectiveness': lambda x: x.sum() / len(x) * 100,
    'delay_minutes': ['mean', 'std']
}).round(2)
buffer_analysis.columns = ['buffer_success_rate', 'avg_delay', 'delay_variability']
buffer_analysis.sort_values('buffer_success_rate', ascending=False)

# %% [markdown]
# ## Quality Control and Anomaly Detection

# %%
# Anomaly detection for unusual delays
from scipy import stats

# Calculate z-scores for delay times
air['delay_zscore'] = stats.zscore(air['delay_minutes'])
anomalies = air[abs(air['delay_zscore']) > 2]  # More than 2 standard deviations

anomaly_analysis = anomalies.groupby(['airline', 'destination_or_origin']).agg({
    'delay_minutes': ['count', 'mean'],
    'flight_number': 'count'
}).round(2)
anomaly_analysis.columns = ['anomaly_count', 'avg_anomaly_delay', 'total_flights']
anomaly_analysis['anomaly_rate'] = anomaly_analysis['anomaly_count'] / anomaly_analysis['total_flights'] * 100
anomaly_analysis.sort_values('anomaly_rate', ascending=False)

# %%
# Consistency scoring
consistency_analysis = air.groupby('airline').agg({
    'delay_minutes': ['std', 'mean'],
    'Early_OnTime_Late_Indicator': lambda x: (x == 'OnTime').sum() / len(x) * 100
}).round(2)
consistency_analysis.columns = ['delay_std', 'avg_delay', 'ontime_percentage']
consistency_analysis['consistency_score'] = (100 - consistency_analysis['delay_std']) * 0.6 + consistency_analysis['ontime_percentage'] * 0.4
consistency_analysis.sort_values('consistency_score', ascending=False)

# %% [markdown]
# ## Fleet Management Insights

# %%
# Aircraft efficiency analysis
aircraft_efficiency = air.groupby(['aircraft', 'airline']).agg({
    'delay_minutes': ['mean', 'std', 'count'],
    'Early_OnTime_Late_Indicator': lambda x: (x == 'OnTime').sum() / len(x) * 100,
    'destination_or_origin': 'nunique'
}).round(2)
aircraft_efficiency.columns = ['avg_delay', 'delay_consistency', 'flight_count', 'ontime_percentage', 'routes_served']
aircraft_efficiency['efficiency_score'] = (aircraft_efficiency['ontime_percentage'] / 
                                         (aircraft_efficiency['avg_delay'] + 1))
aircraft_efficiency[aircraft_efficiency['flight_count'] >= 10].sort_values('efficiency_score', ascending=False)

# %%
# Aircraft age proxy analysis (using registration patterns)
air['aircraft_age_proxy'] = air['register'].str.extract('(\d+)').astype(float, errors='ignore')
age_performance = air.groupby('aircraft_age_proxy').agg({
    'delay_minutes': 'mean',
    'Early_OnTime_Late_Indicator': lambda x: (x == 'OnTime').sum() / len(x) * 100
}).round(2)
age_performance.columns = ['avg_delay', 'ontime_percentage']
age_performance.sort_values('avg_delay')

# %% [markdown]
# ## Customer Experience Metrics

# %%
# Service quality scoring
def calculate_service_score(row):
    score = 100
    
    # Delay penalty
    if row['delay_minutes'] > 60:
        score -= 50
    elif row['delay_minutes'] > 30:
        score -= 30
    elif row['delay_minutes'] > 15:
        score -= 15
    
    # Day matching bonus
    if row['Actual_Day_Matches_Scheduled_Day'] == 1:
        score += 5
    
    return max(0, score)

air['service_score'] = air.apply(calculate_service_score, axis=1)

service_quality = air.groupby('airline').agg({
    'service_score': 'mean',
    'delay_minutes': 'mean',
    'Early_OnTime_Late_Indicator': lambda x: (x == 'OnTime').sum() / len(x) * 100
}).round(2)
service_quality.columns = ['avg_service_score', 'avg_delay', 'ontime_percentage']
service_quality.sort_values('avg_service_score', ascending=False)

# %%
# Passenger satisfaction proxy
satisfaction_factors = air.groupby('airline').agg({
    'Actual_Day_Matches_Scheduled_Day': 'mean',
    'delay_minutes': lambda x: (x <= 15).sum() / len(x) * 100,  # Acceptable delays
    'Early_OnTime_Late_Indicator': lambda x: (x == 'OnTime').sum() / len(x) * 100
}).round(2)
satisfaction_factors.columns = ['schedule_reliability', 'acceptable_delay_rate', 'ontime_rate']
satisfaction_factors['satisfaction_index'] = (satisfaction_factors['schedule_reliability'] * 0.3 + 
                                            satisfaction_factors['acceptable_delay_rate'] * 0.4 + 
                                            satisfaction_factors['ontime_rate'] * 0.3)
satisfaction_factors.sort_values('satisfaction_index', ascending=False)

# %% [markdown]
# ## Operational Recommendations Engine

# %%
# Generate actionable recommendations
def generate_recommendations(airline_data):
    recommendations = []
    
    # High delay routes
    high_delay_routes = airline_data.groupby('destination_or_origin')['delay_minutes'].mean()
    if high_delay_routes.max() > 45:
        worst_route = high_delay_routes.idxmax()
        recommendations.append(f"Review operations for route to {worst_route} (avg delay: {high_delay_routes.max():.1f} min)")
    
    # Peak hour performance
    peak_hours = airline_data.groupby('Scheduled_Hour_of_Day')['delay_minutes'].mean()
    if peak_hours.max() > 30:
        worst_hour = peak_hours.idxmax()
        recommendations.append(f"Optimize scheduling for {worst_hour}:00 hour (avg delay: {peak_hours.max():.1f} min)")
    
    # Aircraft performance
    aircraft_perf = airline_data.groupby('aircraft')['delay_minutes'].mean()
    if aircraft_perf.max() > 35:
        worst_aircraft = aircraft_perf.idxmax()
        recommendations.append(f"Review {worst_aircraft} aircraft maintenance/scheduling")
    
    return recommendations

# Apply recommendations for each airline
for airline in air['airline'].unique():
    airline_data = air[air['airline'] == airline]
    recommendations = generate_recommendations(airline_data)
    print(f"\n{airline} Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

# %% [markdown]
# ## Performance Dashboard Summary

# %%
# Executive summary dashboard
dashboard_summary = air.groupby('airline').agg({
    'delay_minutes': ['mean', 'std'],
    'Early_OnTime_Late_Indicator': lambda x: (x == 'OnTime').sum() / len(x) * 100,
    'flight_number': 'count',
    'destination_or_origin': 'nunique',
    'estimated_cost': 'sum'
}).round(2)

dashboard_summary.columns = ['avg_delay', 'delay_consistency', 'ontime_percentage', 'total_flights', 'routes_served', 'total_cost']
dashboard_summary['performance_grade'] = pd.cut(dashboard_summary['ontime_percentage'], 
                                               bins=[0, 60, 70, 80, 90, 100], 
                                               labels=['F', 'D', 'C', 'B', 'A'])
dashboard_summary.sort_values('ontime_percentage', ascending=False)

# %%
# Key Performance Indicators (KPIs)
kpi_summary = {
    'Total Flights Analyzed': len(air),
    'Overall On-Time Performance': f"{(air['Early_OnTime_Late_Indicator'] == 'OnTime').sum() / len(air) * 100:.1f}%",
    'Average Delay': f"{air['delay_minutes'].mean():.1f} minutes",
    'Total Estimated Cost': f"${air['estimated_cost'].sum():,.0f}",
    'Flights Requiring Compensation': air['compensation_required'].sum(),
    'Most Problematic Hour': f"{air.groupby('Scheduled_Hour_of_Day')['delay_minutes'].mean().idxmax()}:00",
    'Best Performing Airline': dashboard_summary.index[0],
    'Worst Performing Route': air.groupby('destination_or_origin')['delay_minutes'].mean().idxmax()
}

print("=== FLIGHT OPERATIONS KPI DASHBOARD ===")
for kpi, value in kpi_summary.items():
    print(f"{kpi}: {value}")