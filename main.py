import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import urllib.parse
import urllib.request
from io import BytesIO
import math
import numpy as np
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
import functools
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(layout="wide", page_title="Agricultural Calculations")

# Initialize session state for all inputs to track resets
if 'reset_clicked' not in st.session_state:
    st.session_state['reset_clicked'] = False

# Initialize all other session state variables if they don't exist
session_defaults = {
    'city': "",
    'state': "",
    'start_date': None,
    'end_date': None,
    't_base': 50.0,
    't_lower': 50.0,
    't_upper': 86.0,
    'harvest_date': None,
    'starting_moisture': 0.80,
    'swath_density': 450.0,
    'application_rate': 0.0,
    'crop_type': 'alfalfa',
    'wind_speed': 5.0,
    'show_uncertainty': True,
    'swath_config': 'normal'
}

for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# Inject custom CSS to hide the top bar
st.markdown(
    """
    <style>
    /* Hide the entire header bar */
    [data-testid="stHeader"] {
        display: none;
    }
    /* Hide the Streamlit watermark/footer */
    [data-testid="stDecoration"] {
        display: none;
    }
    /* Adjust padding if needed */
    [data-testid="stAppViewContainer"] {
        padding-top: 0px !important;
    }
    /* Custom styling for info boxes */
    .info-box {
        
        border-left: 5px solid #2196F3;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .success-box {
        
        border-left: 5px solid #28a745;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def reset_inputs():
    """Reset all input values in session state"""
    for key, default_value in session_defaults.items():
        st.session_state[key] = default_value
    st.session_state['reset_clicked'] = True

# =================================================================
# 1) Enhanced Input Validation Functions
# =================================================================
def validate_gdd_inputs(t_base, t_lower, t_upper, start_date, end_date):
    """Comprehensive input validation for GDD calculations"""
    errors = []
    warnings = []
    
    # Temperature validation
    if not (0 <= t_base <= 100):
        errors.append("Base temperature should be between 0-100°F")
    
    if not (0 <= t_lower <= 100):
        errors.append("Lower threshold should be between 0-100°F")
        
    if not (50 <= t_upper <= 120):
        errors.append("Upper threshold should be between 50-120°F")
    
    if t_lower > t_upper:
        errors.append("Lower threshold must be ≤ Upper threshold")
    
    if t_base < t_lower:
        warnings.append("Base temperature is below lower threshold - this may affect calculations")
    
    # Date validation
    if start_date and end_date:
        if (end_date - start_date).days > 365:
            warnings.append("Date range exceeds 1 year - consider shorter periods for accuracy")
        
        if start_date > datetime.now().date():
            errors.append("Start date cannot be in the future")
        
        if (end_date - start_date).days < 1:
            errors.append("End date must be after start date")
    
    return errors, warnings

def validate_drying_inputs(harvest_date, starting_moisture, swath_density, application_rate):
    """Comprehensive input validation for drying calculations"""
    errors = []
    warnings = []
    
    if harvest_date:
        if harvest_date > datetime.now().date() + timedelta(days=30):
            warnings.append("Harvest date is far in the future - forecast accuracy may be limited")
    
    if not (0.1 <= starting_moisture <= 0.95):
        errors.append("Starting moisture should be between 10-95%")
    
    if not (100 <= swath_density <= 2000):
        errors.append("Swath density should be between 100-2000 g/m²")
    
    if not (0 <= application_rate <= 0.1):
        errors.append("Application rate should be between 0-0.1 g/g")
    
    return errors, warnings

def validate_and_clean_weather_data(df, data_type="gdd"):
    """Validate and clean weather data"""
    if df.empty:
        raise ValueError("Weather data is empty")
    
    # Check for required columns based on data type
    if data_type == "gdd":
        required_columns = ['datetime', 'tempmin', 'tempmax']
    else:  # drying
        required_columns = ['datetime', 'temp', 'dew']
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove impossible temperature values for GDD data
    if data_type == "gdd":
        initial_rows = len(df)
        df = df[(df['tempmin'] >= -50) & (df['tempmin'] <= 120)]
        df = df[(df['tempmax'] >= -50) & (df['tempmax'] <= 120)]
        df = df[df['tempmax'] >= df['tempmin']]  # Max should be >= Min
        
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            logger.warning(f"Removed {removed_rows} rows with invalid temperature data")
        
        # Interpolate small gaps (up to 2 days)
        df['tempmin'] = df['tempmin'].interpolate(method='linear', limit=2)
        df['tempmax'] = df['tempmax'].interpolate(method='linear', limit=2)
    
    return df

# =================================================================
# 2) Enhanced Weather Data Fetching with Caching
# =================================================================
@functools.lru_cache(maxsize=50)
def fetch_weather_data_cached(api_key, city, state, start_date, end_date, unit_group, elements):
    """Cached weather data fetching to avoid repeated API calls"""
    return fetch_weather_data_from_api(api_key, city, state, start_date, end_date, unit_group, elements)

def fetch_weather_data_from_api(api_key, city, state, start_date, end_date,
                                unit_group="us", elements="datetime,tempmin,tempmax"):
    """
    Enhanced weather data fetching with better error handling
    """
    location_string = f"{city}, {state}"
    encoded_location = urllib.parse.quote(location_string)

    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    url = (
        f"{base_url}/{encoded_location}/{start_date}/{end_date}"
        f"?unitGroup={unit_group}&include=days&elements={elements}"
        f"&key={api_key}&contentType=csv"
    )

    try:
        response = urllib.request.urlopen(url, timeout=30)
        df_api = pd.read_csv(BytesIO(response.read()))
        return df_api
    except urllib.error.HTTPError as e:
        error_msg = f"HTTP Error {e.code}: {e.read().decode()}"
        logger.error(error_msg)
        st.error(f"Weather API Error: {error_msg}")
    except urllib.error.URLError as e:
        error_msg = f"URL Error: {e.reason}"
        logger.error(error_msg)
        st.error(f"Network Error: {error_msg}")
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        st.error(f"Data Processing Error: {error_msg}")

    return pd.DataFrame()

def fetch_drying_weather_data_from_api(api_key, city, state, start_date, end_date, unit_group="us"):
    """Enhanced drying weather data fetching with validation"""
    location_string = f"{city}, {state}"
    encoded_location = urllib.parse.quote(location_string)

    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"

    # Daily data
    daily_url = (
        f"{base_url}/{encoded_location}/{start_date}/{end_date}"
        f"?unitGroup={unit_group}&include=days&elements=datetime,temp,dew,soilmoisturevol01,windspeed"
        f"&key={api_key}&contentType=csv"
    )

    try:
        response = urllib.request.urlopen(daily_url, timeout=30)
        df_days = pd.read_csv(BytesIO(response.read()))
    except Exception as e:
        logger.error(f"Error fetching daily data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

    # Hourly data
    hourly_url = (
        f"{base_url}/{encoded_location}/{start_date}/{end_date}"
        f"?unitGroup={unit_group}&include=hours&elements=datetime,solarradiation"
        f"&key={api_key}&contentType=csv"
    )

    try:
        response = urllib.request.urlopen(hourly_url, timeout=30)
        df_hours = pd.read_csv(BytesIO(response.read()))
    except Exception as e:
        logger.error(f"Error fetching hourly data: {str(e)}")
        return df_days, pd.DataFrame()

    return df_days, df_hours

# =================================================================
# 3) Enhanced GDD Calculation Functions
# =================================================================
def calc_average_gdd(t_min, t_max, T_base):
    """Standard average method for GDD calculation"""
    return max(((t_max + t_min) / 2) - T_base, 0)

def calc_single_sine_gdd(t_min, t_max, T_base):
    """Single sine wave method for GDD calculation"""
    T_mean = (t_max + t_min) / 2
    A = (t_max - t_min) / 2
    
    if t_max <= T_base:
        return 0.0
    if t_min >= T_base:
        return (T_mean - T_base)
    
    alpha = (T_base - T_mean) / A
    if alpha > 1:  # Handle edge case
        return 0.0
    
    theta = math.acos(alpha)
    dd = ((T_mean - T_base)*(math.pi - 2*theta) + A*math.sin(2*theta)) / math.pi
    return max(dd, 0)

def calc_single_triangle_gdd(t_min, t_max, T_base):
    """Single triangle method for GDD calculation"""
    if t_max <= T_base:
        return 0.0
    if t_min >= T_base:
        return ((t_max + t_min)/2 - T_base)
    
    proportion_of_day = (t_max - T_base) / (t_max - t_min)
    avg_above = ((t_max + T_base) / 2) - T_base
    dd = proportion_of_day * avg_above
    return max(dd, 0)

def calc_double_sine_gdd(t_min_today, t_max_today, t_min_tomorrow, T_base):
    """Corrected double sine wave method"""
    # First half: sunrise to max temperature
    seg1 = calc_single_sine_gdd(t_min_today, t_max_today, T_base) * 0.5
    
    # Second half: max temperature to next day's minimum (cooling curve)
    if t_max_today > t_min_tomorrow:
        seg2 = calc_single_sine_gdd(t_min_tomorrow, t_max_today, T_base) * 0.5
    else:
        seg2 = 0  # No cooling if tomorrow's min is higher than today's max
    
    return seg1 + seg2

def calc_double_triangle_gdd(t_min_today, t_max_today, t_min_tomorrow, T_base):
    """Corrected double triangle method"""
    seg1 = calc_single_triangle_gdd(t_min_today, t_max_today, T_base) * 0.5
    seg2 = calc_single_triangle_gdd(t_min_tomorrow, t_max_today, T_base) * 0.5
    return seg1 + seg2

def calculate_daily_gdd_vectorized(df, method="average", T_base=50.0, T_lower=50.0, T_upper=86.0):
    """Vectorized GDD calculation for better performance"""
    df = df.copy()
    
    # Apply temperature limits vectorized
    df['tmax_adj'] = np.minimum(df['tmax'], T_upper)
    df['tmin_adj'] = np.maximum(df['tmin'], T_lower)
    
    if method == "average":
        df['daily_gdd'] = np.maximum(((df['tmax_adj'] + df['tmin_adj']) / 2) - T_base, 0)
    
    elif method in ["sine", "triangle", "double_sine", "double_triangle"]:
        # For non-vectorizable methods, use apply with optimized function
        df['daily_gdd'] = df.apply(
            lambda row: calculate_daily_gdd_single(
                row, df, method, T_base, T_lower, T_upper
            ), axis=1
        )
    
    return df

def calculate_daily_gdd_single(row, df, method, T_base, T_lower, T_upper):
    """Single row GDD calculation with all methods"""
    idx = row.name
    t_max = min(row['tmax'], T_upper)
    t_min = max(row['tmin'], T_lower)
    
    if method == "average":
        return calc_average_gdd(t_min, t_max, T_base)
    elif method == "sine":
        return calc_single_sine_gdd(t_min, t_max, T_base)
    elif method == "triangle":
        return calc_single_triangle_gdd(t_min, t_max, T_base)
    elif method == "double_sine":
        if idx < len(df) - 1:
            t_min_next = max(df.loc[idx+1, 'tmin'], T_lower)
            return calc_double_sine_gdd(t_min, t_max, t_min_next, T_base)
        else:
            return calc_single_sine_gdd(t_min, t_max, T_base)
    elif method == "double_triangle":
        if idx < len(df) - 1:
            t_min_next = max(df.loc[idx+1, 'tmin'], T_lower)
            return calc_double_triangle_gdd(t_min, t_max, t_min_next, T_base)
        else:
            return calc_single_triangle_gdd(t_min, t_max, T_base)
    else:
        raise ValueError(f"Unknown method: {method}")

def calculate_gdds_with_uncertainty(df, start_date, end_date, method="average", 
                                   T_base=50.0, T_lower=50.0, T_upper=86.0):
    """Calculate GDD with uncertainty bounds"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    
    # Base calculation
    df_base = calculate_daily_gdd_vectorized(df, method, T_base, T_lower, T_upper)
    
    # Temperature uncertainty (±2°F typical for weather data)
    temp_uncertainty = 2.0
    
    # Upper bound (temperatures +2°F)
    df_upper = df.copy()
    df_upper['tmax'] += temp_uncertainty
    df_upper['tmin'] += temp_uncertainty
    df_upper = calculate_daily_gdd_vectorized(df_upper, method, T_base, T_lower, T_upper)
    
    # Lower bound (temperatures -2°F)
    df_lower = df.copy()
    df_lower['tmax'] -= temp_uncertainty
    df_lower['tmin'] -= temp_uncertainty
    df_lower = calculate_daily_gdd_vectorized(df_lower, method, T_base, T_lower, T_upper)
    
    # Apply date filters
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    for df_temp in [df_base, df_upper, df_lower]:
        df_temp.loc[df_temp['date'] < start_date, 'daily_gdd'] = 0
        df_temp.loc[df_temp['date'] > end_date, 'daily_gdd'] = 0
    
    # Calculate cumulative GDD
    df_base['cumulative_gdd'] = df_base['daily_gdd'].cumsum()
    df_base['gdd_upper'] = df_upper['daily_gdd'].cumsum()
    df_base['gdd_lower'] = df_lower['daily_gdd'].cumsum()
    
    return df_base

# =================================================================
# 4) Enhanced Drying Calculation Functions
# =================================================================
def merge_dfs_enhanced(daily_df, hourly_df):
    """Enhanced dataframe merging with better error handling"""
    try:
        d_df = daily_df.copy()
        h_df = hourly_df.copy()

        # Convert datetime columns
        h_df['datetime'] = pd.to_datetime(h_df['datetime'])
        d_df['datetime'] = pd.to_datetime(d_df['datetime'])

        # Extract date from hourly timestamps
        h_df['date'] = h_df['datetime'].dt.date

        # Group by date and get max solar radiation
        daily_peaks = h_df.groupby('date')['solarradiation'].max().reset_index()
        daily_peaks.rename(columns={'solarradiation': 'peak_solarradiation'}, inplace=True)

        # Merge with daily dataframe
        d_df['date'] = d_df['datetime'].dt.date
        merged_df = pd.merge(d_df, daily_peaks, on='date', how='left')

        # Fill missing solar radiation values with interpolation
        merged_df['peak_solarradiation'] = merged_df['peak_solarradiation'].interpolate(method='linear')
        
        # If still NaN, use a reasonable default (300 W/m² for partly cloudy)
        merged_df['peak_solarradiation'] = merged_df['peak_solarradiation'].fillna(300)

        merged_df.drop(columns='date', inplace=True)
        return merged_df
    
    except Exception as e:
        logger.error(f"Error merging dataframes: {str(e)}")
        return pd.DataFrame()

def calculate_vapor_pressure_deficit_enhanced(df):
    """Enhanced VPD calculation with error handling"""
    df = df.copy()
    
    try:
        # Convert temperature and dew point from Fahrenheit to Celsius
        df['temp_C'] = (df['temp'] - 32) / 1.8
        df['dew_C'] = (df['dew'] - 32) / 1.8

        # Calculate saturation and actual vapor pressure using Tetens formula
        df['saturation_vapor_pressure'] = 0.6108 * np.exp((17.27 * df['temp_C']) / (df['temp_C'] + 237.3))
        df['actual_vapor_pressure'] = 0.6108 * np.exp((17.27 * df['dew_C']) / (df['dew_C'] + 237.3))

        # Calculate VPD
        df['vapor_pressure_deficit'] = df['saturation_vapor_pressure'] - df['actual_vapor_pressure']
        
        # Ensure VPD is not negative
        df['vapor_pressure_deficit'] = np.maximum(df['vapor_pressure_deficit'], 0)

        return df
    
    except Exception as e:
        logger.error(f"Error calculating VPD: {str(e)}")
        # Return original dataframe with default VPD
        df['vapor_pressure_deficit'] = 1.0  # Default VPD value
        return df

def get_equilibrium_moisture(crop_type, vpd, temperature):
    """
    Calculate equilibrium moisture content based on crop type and weather conditions
    """
    # Base equilibrium moisture by crop type
    base_equilibrium = {
        "alfalfa": 0.08,
        "clover": 0.07,
        "grass": 0.06,
        "timothy": 0.06,
        "fescue": 0.06
    }
    
    base = base_equilibrium.get(crop_type, 0.07)
    
    # Adjust based on environmental conditions
    # Higher VPD (drier air) = lower equilibrium moisture
    vpd_adjustment = max(-0.01, -0.005 * (vpd - 1.0))
    
    # Higher temperature = lower equilibrium moisture
    temp_adjustment = max(-0.01, -0.0002 * (temperature - 70))
    
    return max(base + vpd_adjustment + temp_adjustment, 0.04)  # Minimum 4%

def calculate_enhanced_drying_rate_constant(SI, VPD, DAY, SM, SD, AR=0, wind_speed=5.0, crop_type="alfalfa", swath_config="normal"):
    """
    Enhanced drying rate calculation with swath configuration and improved crop coefficients
    
    Parameters:
    SI = solar insolation, W/m^2
    VPD = vapor pressure deficit, kPa
    DAY = 1 for first day, 0 otherwise
    SM = soil moisture content, % dry basis
    SD = swath density, g/m^2
    AR = application rate of chemical solution, g_solution/g_dry-matter
    wind_speed = wind speed, m/s
    crop_type = type of crop
    swath_config = swath configuration: "thin", "normal", "thick"
    """
    
    # Updated crop-specific coefficients based on real-world drying behavior
    # Legumes (alfalfa, clover) have waxy cuticles that slow drying
    # Grasses typically dry faster due to thinner leaves and stems
    crop_coefficients = {
        "alfalfa": 0.85,    # Slower due to thick stems and waxy leaves
        "clover": 1.1,      # Best drying (as you noted)
        "grass": 1.3,       # Fast drying, thin leaves
        "timothy": 1.25,    # Fast drying grass
        "fescue": 1.2       # Fast drying grass
    }
    crop_factor = crop_coefficients.get(crop_type, 1.0)
    
    # Swath configuration factor - critical for first 24 hours
    # Based on Wisconsin Extension research on swath thickness impact
    swath_factors = {
        "thin": 1.4,     # Wide, thin swath - maximum surface area exposure
        "normal": 1.0,   # Standard swath
        "thick": 0.6     # Thick swath - reduced air circulation
    }
    swath_factor = swath_factors.get(swath_config, 1.0)
    
    # Enhanced first-day drying multiplier
    # First 24 hours are critical - most free water is lost during this period
    if DAY == 1:
        first_day_multiplier = 2.5 * swath_factor  # Swath config is most important on day 1
    else:
        first_day_multiplier = 1.0
    
    # Wind speed factor (more pronounced effect)
    wind_factor = 1 + (0.08 * wind_speed)  # 8% increase per m/s
    
    # Enhanced drying rate formula with swath consideration
    numerator = (SI * (1.0 + 9.03*AR) * wind_factor * crop_factor * first_day_multiplier) + (43.8 * VPD * swath_factor)
    
    # Adjusted denominator to account for swath density impact
    # Thick swaths create their own humid microclimate
    swath_resistance = SD * (1.82 - 0.83 * DAY) * ((1.68 + 24.8 * AR))
    if swath_config == "thick":
        swath_resistance *= 1.3  # Additional resistance for thick swaths
    elif swath_config == "thin":
        swath_resistance *= 0.8  # Reduced resistance for thin swaths
    
    denominator = (61.4 * SM) + swath_resistance + 1500
    
    drying_rate = numerator / denominator
    
    # Weather condition multipliers
    if SI > 400 and VPD > 1.5:  # Excellent drying weather
        drying_rate *= 1.5
    elif SI > 300 and VPD > 1.0:  # Good drying weather
        drying_rate *= 1.2
    
    # Return rate with realistic bounds
    # First day can have very high rates (up to 120% per day in ideal conditions)
    # Subsequent days are more moderate
    max_rate = 1.2 if DAY == 1 else 0.6
    return max(min(drying_rate, max_rate), 0.001)

def predict_moisture_content_enhanced(df, startdate, swath_density=450, starting_moisture=0.80, 
                                    application_rate=0, crop_type="alfalfa", wind_speed=5.0, 
                                    swath_config="normal"):
    """
    Enhanced moisture content prediction with swath configuration
    """
    try:
        # Ensure datetime and sort
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df[df['datetime'] >= pd.to_datetime(startdate)].copy()
        df.sort_values('datetime', inplace=True)

        if df.empty:
            raise ValueError("No data available for the specified date range")

        # Calculate VPD if not already present
        if 'vapor_pressure_deficit' not in df.columns:
            df = calculate_vapor_pressure_deficit_enhanced(df)

        # Initialize tracking lists
        moisture_contents = [starting_moisture]
        drying_rates = []
        current_moisture = starting_moisture

        for idx, row in df.iterrows():
            day_number = len(moisture_contents) - 1
            DAY = 1 if day_number == 0 else 0

            # Get parameters with defaults
            SI = max(row.get('peak_solarradiation', 300), 0)
            VPD = max(row.get('vapor_pressure_deficit', 1.0), 0)
            SM = row.get('soilmoisturevol01', 0.1) * 100 if not pd.isna(row.get('soilmoisturevol01', 0.1)) else 10
            wind_actual = row.get('windspeed', wind_speed)
            
            # Calculate drying rate with swath configuration
            k = calculate_enhanced_drying_rate_constant(
                SI, VPD, DAY, SM, swath_density, application_rate, 
                wind_actual, crop_type, swath_config
            )

            # Apply exponential decay model
            # For very high first-day rates, use a modified decay to prevent unrealistic drops
            if DAY == 1 and k > 0.8:
                # Use a more gradual model for extremely high rates
                daily_loss_fraction = 1 - math.exp(-k * 0.7)  # Moderate the extreme rates slightly
            else:
                daily_loss_fraction = 1 - math.exp(-k)
            
            # Calculate new moisture content
            current_moisture = current_moisture * (1 - daily_loss_fraction)
            
            # Set equilibrium moisture based on crop type and conditions
            equilibrium_moisture = get_equilibrium_moisture(crop_type, VPD, row.get('temp', 70))
            current_moisture = max(current_moisture, equilibrium_moisture)
            
            moisture_contents.append(current_moisture)
            drying_rates.append(k)

            # Early termination if moisture stabilizes
            if current_moisture <= equilibrium_moisture * 1.01:  # Within 1% of equilibrium
                break

        # Create result DataFrame
        result_df = df.iloc[:len(moisture_contents)-1].copy()
        result_df['drying_rates'] = drying_rates[:len(result_df)]
        result_df['predicted_moisture'] = moisture_contents[:-1]
        result_df['predicted_moisture_pct'] = result_df['predicted_moisture'] * 100

        return result_df.dropna(subset=['predicted_moisture'])
    
    except Exception as e:
        logger.error(f"Error in moisture prediction: {str(e)}")
        return pd.DataFrame()

def calculate_crop_specific_timelines(crop_type, weather_conditions="average"):
    """
    Return expected drying timelines for different crops under various conditions
    Based on Wisconsin Extension and other agricultural research
    """
    timelines = {
        "clover": {
            "excellent": 1,  # Perfect weather
            "good": 2,       # Good drying weather
            "average": 2,    # Average conditions
            "poor": 4        # High humidity, low wind
        },
        "grass": {
            "excellent": 2,
            "good": 3,
            "average": 3,
            "poor": 5
        },
        "timothy": {
            "excellent": 2,
            "good": 3,
            "average": 3,
            "poor": 5
        },
        "fescue": {
            "excellent": 2,
            "good": 3,
            "average": 4,
            "poor": 6
        },
        "alfalfa": {
            "excellent": 3,
            "good": 4,
            "average": 5,
            "poor": 7
        }
    }
    
    return timelines.get(crop_type, timelines["alfalfa"])

# =================================================================
# 5) Enhanced Plotting Functions
# =================================================================
def create_enhanced_gdd_plot(df, location_str, method, show_uncertainty=True):
    """Create enhanced GDD plot with uncertainty bands"""
    fig = go.Figure()
    
    # Main GDD line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['cumulative_gdd'],
        mode='lines',
        name='Cumulative GDD',
        line=dict(color='blue', width=3)
    ))
    
    if show_uncertainty and 'gdd_upper' in df.columns:
        # Upper bound
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['gdd_upper'],
            mode='lines',
            name='Upper Bound (+2°F)',
            line=dict(color='lightblue', width=1, dash='dash')
        ))
        
        # Lower bound
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['gdd_lower'],
            mode='lines',
            name='Lower Bound (-2°F)',
            line=dict(color='lightblue', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(173, 216, 230, 0.3)'
        ))
    
    fig.update_layout(
        title=f"Cumulative Growing Degree Days<br><sub>Location: {location_str} | Method: {method}</sub>",
        xaxis_title="Date",
        yaxis_title="Cumulative GDD (°F-days)",
        hovermode='x unified',
        height=600,
        showlegend=True
    )
    
    return fig

def create_enhanced_drying_plot(df, location_str, harvest_date):
    """Create enhanced drying plot with multiple indicators"""
    fig = go.Figure()
    
    # Main moisture line
    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['predicted_moisture_pct'],
        mode='lines+markers',
        name='Predicted Moisture',
        line=dict(color='blue', width=3),
        marker=dict(size=6)
    ))
    
    # Target moisture line (15% for baling)
    fig.add_hline(
        y=15,
        line_dash="dash",
        line_color="green",
        annotation_text="Optimal for Baling (15%)",
        annotation_position="top right"
    )
    
    # Critical moisture line (8% - too dry)
    fig.add_hline(
        y=8,
        line_dash="dash",
        line_color="red",
        annotation_text="Too Dry (8%)",
        annotation_position="bottom right"
    )
    
    fig.update_layout(
        title=f"Crop Moisture Content Prediction<br><sub>Location: {location_str} | Harvest: {harvest_date}</sub>",
        xaxis_title="Date",
        yaxis_title="Moisture Content (%)",
        hovermode='x unified',
        height=600,
        yaxis=dict(range=[5, max(df['predicted_moisture_pct'].max() + 5, 85)])
    )
    
    return fig

# =================================================================
# 6) Main Application
# =================================================================
def main():
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    
    if not API_KEY:
        st.error("⚠️ Weather API key not found. Please set API_KEY in your environment variables.")
        st.stop()

    # Enhanced title with styling
    st.markdown("""
        <h1 style='text-align: center; color: #2E7D32; margin-bottom: 30px;'>
            🌾 Enhanced Agricultural Forecasting System
        </h1>
        
    """, unsafe_allow_html=True)

    # Create sidebar for inputs
    with st.sidebar:
        st.markdown("### 📊 Input Parameters")

        # Algorithm Selection
        algorithm_options = ["GDD Calculation", "Drying Calculation"]
        algorithm = st.selectbox(
            "Select Calculation Algorithm",
            options=algorithm_options,
            key="algorithm",
            help="Choose between Growing Degree Days calculation or Crop Drying prediction"
        )

        # GDD Calculation inputs
        if algorithm == "GDD Calculation":
            st.markdown("### 🌡️ GDD Parameters")

            # Location inputs with validation
            city = st.text_input("City", value=st.session_state['city'], 
                               placeholder="e.g., Ithaca")
            state = st.text_input("State", value=st.session_state['state'], 
                                placeholder="e.g., NY")

            # Date inputs
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", 
                                         value=st.session_state['start_date'] if st.session_state['start_date'] else None,
                                         help="Select the start date for GDD calculation")
            with col2:
                end_date = st.date_input("End Date", 
                                       value=st.session_state['end_date'] if st.session_state['end_date'] else None,
                                       help="Select the end date for GDD calculation")

            # Method selection
            method_options = ["average", "sine", "triangle", "double_sine", "double_triangle"]
            method = st.selectbox(
                "Calculation Method",
                options=method_options,
                help="Choose the GDD calculation method. Average is simplest, sine/triangle are more accurate."
            )

            # Temperature parameters
            st.markdown("#### Temperature Thresholds")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                t_base = st.number_input("Base Temp (°F)", 
                                       value=st.session_state['t_base'], 
                                       min_value=0.0, max_value=100.0, step=0.5,
                                       help="Base temperature for crop development")
            with col2:
                t_lower = st.number_input("Lower Limit (°F)", 
                                        value=st.session_state['t_lower'], 
                                        min_value=0.0, max_value=100.0, step=0.5,
                                        help="Lower temperature threshold")
            with col3:
                t_upper = st.number_input("Upper Limit (°F)", 
                                        value=st.session_state['t_upper'], 
                                        min_value=50.0, max_value=120.0, step=0.5,
                                        help="Upper temperature threshold")

            # Advanced options
            with st.expander("🔧 Advanced Options"):
                show_uncertainty = st.checkbox("Show Uncertainty Bands", 
                                             value=st.session_state['show_uncertainty'],
                                             help="Display confidence intervals based on temperature uncertainty")

            # Save input values to session state
            st.session_state['city'] = city
            st.session_state['state'] = state
            st.session_state['start_date'] = start_date
            st.session_state['end_date'] = end_date
            st.session_state['t_base'] = t_base
            st.session_state['t_lower'] = t_lower
            st.session_state['t_upper'] = t_upper
            st.session_state['show_uncertainty'] = show_uncertainty

            # Validation
            errors, warnings = validate_gdd_inputs(t_base, t_lower, t_upper, start_date, end_date)
            
            # Display validation messages
            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
            
            if warnings:
                for warning in warnings:
                    st.warning(f"⚠️ {warning}")

            # Calculate button
            calculate_button = st.button("🚀 Calculate GDD", 
                                       disabled=bool(errors or not city or not state or not start_date or not end_date),
                                       use_container_width=True)

        # Drying Calculation inputs
        elif algorithm == "Drying Calculation":
            st.markdown("### 🌾 Drying Parameters")

            # Location inputs
            city = st.text_input("City", value=st.session_state['city'], 
                               placeholder="e.g., Ithaca")
            state = st.text_input("State", value=st.session_state['state'], 
                                placeholder="e.g., NY")
            
            harvest_date = st.date_input("Harvest Date", 
                                       value=st.session_state['harvest_date'],
                                       help="Date when crop was/will be harvested")

            # Crop parameters
            st.markdown("#### Crop Characteristics")
            
            col1, col2 = st.columns(2)
            with col1:
                crop_type = st.selectbox("Crop Type", 
                                       options=["clover", "grass", "timothy", "fescue", "alfalfa"],
                                       index=0 if st.session_state.get('crop_type', 'alfalfa') == 'clover' else 
                                             ["clover", "grass", "timothy", "fescue", "alfalfa"].index(st.session_state.get('crop_type', 'alfalfa')),
                                       help="Type of crop being dried (ordered by typical drying speed)")
            
            with col2:
                starting_moisture = st.slider("Starting Moisture (%)", 
                                            min_value=10, max_value=95, 
                                            value=int(st.session_state.get('starting_moisture', 0.80) * 100),
                                            help="Initial moisture content at harvest") / 100

            # Swath configuration - NEW PARAMETER
            st.markdown("#### Swath Configuration")
            swath_config = st.selectbox(
                "Swath Thickness",
                options=["thin", "normal", "thick"],
                index=1,  # Default to "normal"
                help="Swath configuration affects drying rate, especially in first 24 hours. Thin swaths dry fastest."
            )
            
            # Visual indicator for swath impact
            swath_impact = {
                "thin": "🟢 Fastest drying - maximum air exposure",
                "normal": "🟡 Standard drying rate",
                "thick": "🔴 Slower drying - limited air circulation"
            }
            # st.info(swath_impact[swath_config])

            col1, col2 = st.columns(2)
            with col1:
                swath_density = st.number_input("Swath Density (g/m²)", 
                                              value=st.session_state.get('swath_density', 450.0), 
                                              min_value=100.0, max_value=2000.0, step=25.0,
                                              help="Density of crop material in swath")
            
            with col2:
                application_rate = st.number_input("Application Rate (g/g)", 
                                                 value=st.session_state.get('application_rate', 0.0), 
                                                 min_value=0.0, max_value=0.1, step=0.005,
                                                 help="Chemical application rate (conditioner)")

            # Environmental parameters
            with st.expander("🌬️ Environmental Parameters"):
                wind_speed = st.slider("Average Wind Speed (m/s)", 
                                     min_value=0.0, max_value=15.0, 
                                     value=st.session_state.get('wind_speed', 5.0), step=0.5,
                                     help="Expected average wind speed during drying period")

            # Expected timeline display
            # with st.expander("📅 Expected Drying Timeline Reference"):
            #     timelines = calculate_crop_specific_timelines(crop_type)
            #     st.markdown(f"""
            #     **Typical drying times for {crop_type.title()} to reach 15% moisture:**
            #     - Excellent conditions (high sun, low humidity, good wind): **{timelines['excellent']} day(s)**
            #     - Good conditions: **{timelines['good']} day(s)**
            #     - Average conditions: **{timelines['average']} day(s)**
            #     - Poor conditions (cloudy, humid, low wind): **{timelines['poor']} day(s)**
                
            #     *These are general guidelines. Actual drying depends on specific weather conditions.*
            #     """)

            # Save input values to session state
            st.session_state['city'] = city
            st.session_state['state'] = state
            st.session_state['harvest_date'] = harvest_date
            st.session_state['crop_type'] = crop_type
            st.session_state['starting_moisture'] = starting_moisture
            st.session_state['swath_density'] = swath_density
            st.session_state['application_rate'] = application_rate
            st.session_state['wind_speed'] = wind_speed
            st.session_state['swath_config'] = swath_config

            # Validation
            errors, warnings = validate_drying_inputs(harvest_date, starting_moisture, swath_density, application_rate)
            
            # Display validation messages
            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
            
            if warnings:
                for warning in warnings:
                    st.warning(f"⚠️ {warning}")

            # Calculate button
            drying_button = st.button("🚀 Predict Drying", 
                                    disabled=bool(errors or not city or not state or not harvest_date),
                                    use_container_width=True)

        # Reset button at the bottom of sidebar
        st.markdown("---")
        if st.button("🔄 Reset All Inputs", use_container_width=True):
            reset_inputs()
            st.rerun()

    # Main content area - for results only
    if algorithm == "GDD Calculation" and 'calculate_button' in locals() and calculate_button:
        with st.spinner("🌡️ Fetching weather data and calculating GDD..."):
            # Fetch data
            df_api = fetch_weather_data_from_api(
                api_key=API_KEY,
                city=city,
                state=state,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                unit_group="us",
                elements="datetime,tempmin,tempmax"
            )

            if not df_api.empty:
                # Display raw API data
                # with st.expander("🔍 View Raw Weather API Data"):
                #     st.dataframe(df_api)

                # Validate and clean the raw data
                df_validated = validate_and_clean_weather_data(df_api, "gdd")

                # Rename columns
                df_renamed = df_validated.copy()
                df_renamed.rename(columns={
                    "datetime": "date",
                    "tempmax": "tmax",
                    "tempmin": "tmin"
                }, inplace=True)

                # Calculate GDD with or without uncertainty
                if show_uncertainty:
                    df_calc = calculate_gdds_with_uncertainty(
                        df=df_renamed,
                        start_date=start_date,
                        end_date=end_date,
                        method=method,
                        T_base=t_base,
                        T_lower=t_lower,
                        T_upper=t_upper
                    )
                else:
                    df_calc = calculate_daily_gdd_vectorized(
                        df=df_renamed,
                        method=method,
                        T_base=t_base,
                        T_lower=t_lower,
                        T_upper=t_upper
                    )
                    # Apply date filters and calculate cumulative
                    start_date_pd = pd.to_datetime(start_date)
                    end_date_pd = pd.to_datetime(end_date)
                    df_calc['date'] = pd.to_datetime(df_calc['date'])
                    df_calc.loc[df_calc['date'] < start_date_pd, 'daily_gdd'] = 0
                    df_calc.loc[df_calc['date'] > end_date_pd, 'daily_gdd'] = 0
                    df_calc['cumulative_gdd'] = df_calc['daily_gdd'].cumsum()

                # Display results
                location_str = f"{city}, {state}"
                st.markdown("## 📈 GDD Calculation Results")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_gdd = df_calc['cumulative_gdd'].iloc[-1]
                    st.metric("Total GDD", f"{total_gdd:.1f}°F-days")
                
                with col2:
                    avg_daily_gdd = df_calc[df_calc['daily_gdd'] > 0]['daily_gdd'].mean()
                    st.metric("Avg Daily GDD", f"{avg_daily_gdd:.1f}°F-days")
                
                with col3:
                    active_days = len(df_calc[df_calc['daily_gdd'] > 0])
                    st.metric("Active Days", f"{active_days}")
                
                with col4:
                    max_daily_gdd = df_calc['daily_gdd'].max()
                    st.metric("Max Daily GDD", f"{max_daily_gdd:.1f}°F-days")

                # Create and display plot
                fig = create_enhanced_gdd_plot(df_calc, location_str, method, show_uncertainty)
                st.plotly_chart(fig, use_container_width=True)

                # Data table
                with st.expander("📊 View Detailed Data"):
                    display_df = df_calc[['date', 'tmin', 'tmax', 'daily_gdd', 'cumulative_gdd']].copy()
                    display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
                    display_df.columns = ['Date', 'Min Temp (°F)', 'Max Temp (°F)', 'Daily GDD', 'Cumulative GDD']
                    st.dataframe(display_df, use_container_width=True)

            else:
                st.error("❌ Failed to fetch weather data. Please check your inputs and try again.")

    elif algorithm == "Drying Calculation" and 'drying_button' in locals() and drying_button:
        with st.spinner("🌾 Fetching weather data and predicting drying..."):
            # Calculate forecast period (2 months from harvest date)
            harvest_date_str = harvest_date.strftime("%Y-%m-%d")
            end_date_dt = harvest_date + relativedelta(months=2)
            end_date_str = end_date_dt.strftime("%Y-%m-%d")

            # Fetch data
            daily_df, hourly_df = fetch_drying_weather_data_from_api(
                api_key=API_KEY,
                city=city,
                state=state,
                start_date=harvest_date_str,
                end_date=end_date_str,
                unit_group="us"
            )

            if not daily_df.empty and not hourly_df.empty:
                # Display raw API data
                # with st.expander("🔍 View Raw Weather API Data"):
                #     st.markdown("#### Daily Data")
                #     st.dataframe(daily_df)
                #     st.markdown("#### Hourly Data")
                #     st.dataframe(hourly_df)

                # Merge dataframes
                merged_df = merge_dfs_enhanced(daily_df, hourly_df)

                # Calculate drying prediction with updated function call
                drying_df = predict_moisture_content_enhanced(
                    df=merged_df,
                    startdate=harvest_date_str,
                    swath_density=swath_density,
                    starting_moisture=starting_moisture,
                    application_rate=application_rate,
                    crop_type=crop_type,
                    wind_speed=wind_speed,
                    swath_config=st.session_state.get('swath_config', 'normal')
                )

                if not drying_df.empty:
                    # Display results
                    location_str = f"{city}, {state}"
                    st.markdown("## 📈 Crop Drying Prediction Results")

                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        current_moisture = drying_df['predicted_moisture_pct'].iloc[-1]
                        st.metric("Final Moisture", f"{current_moisture:.1f}%")
                    
                    with col2:
                        days_elapsed = len(drying_df)
                        st.metric("Days Predicted", f"{days_elapsed}")
                    
                    # Calculate days to reach 15% moisture
                    target_moisture = 15
                    baling_ready_rows = drying_df[drying_df['predicted_moisture_pct'] <= target_moisture]
                    
                    with col3:
                        if not baling_ready_rows.empty:
                            baling_date = pd.to_datetime(baling_ready_rows.iloc[0]['datetime'])
                            days_to_baling = (baling_date - pd.to_datetime(harvest_date_str)).days
                            st.metric("Days to Baling", f"{days_to_baling}")
                        else:
                            st.metric("Days to Baling", "Not reached")
                    
                    with col4:
                        avg_drying_rate = drying_df['drying_rates'].mean() * 100
                        st.metric("Avg Drying Rate", f"{avg_drying_rate:.2f}%/day")

                    # Create and display plot
                    fig = create_enhanced_drying_plot(drying_df, location_str, harvest_date_str)
                    st.plotly_chart(fig, use_container_width=True)

                    # Baling recommendation
                    if not baling_ready_rows.empty:
                        baling_date = pd.to_datetime(baling_ready_rows.iloc[0]['datetime'])
                        days_to_baling = (baling_date - pd.to_datetime(harvest_date_str)).days
                        
                        st.markdown(f"""
                        <div class="success-box">
                            <h4>✅ Baling Recommendation</h4>
                            <p>Based on weather forecasts, your {crop_type} crop will reach optimal baling moisture (15%) in approximately <strong>{days_to_baling} days</strong> from harvest, around <strong>{baling_date.strftime('%B %d, %Y')}</strong>.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="warning-box">
                            <h4>⚠️ Extended Drying Required</h4>
                            <p>The crop may not reach optimal baling moisture (15%) within the forecast period. Consider extending the drying period or checking for adverse weather conditions.</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Data table
                    with st.expander("📊 View Detailed Drying Data"):
                        display_df = drying_df[['datetime', 'predicted_moisture_pct', 'drying_rates', 'peak_solarradiation', 'vapor_pressure_deficit']].copy()
                        display_df['datetime'] = pd.to_datetime(display_df['datetime']).dt.strftime('%Y-%m-%d')
                        display_df['drying_rates'] = display_df['drying_rates'] * 100  # Convert to percentage
                        display_df.columns = ['Date', 'Moisture (%)', 'Drying Rate (%/day)', 'Solar Radiation (W/m²)', 'VPD (kPa)']
                        st.dataframe(display_df, use_container_width=True)

                else:
                    st.error("❌ Failed to process drying data. Please check your inputs and try again.")

            else:
                st.error("❌ Failed to fetch weather data. Please check your inputs and try again.")

    # Information section
    if not (algorithm == "GDD Calculation" and 'calculate_button' in locals() and calculate_button) and \
       not (algorithm == "Drying Calculation" and 'drying_button' in locals() and drying_button):
        
        st.markdown("## 📖 About This Tool")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🌡️ Growing Degree Days (GDD)
            - **Purpose**: Predict crop development timing
            - **Methods**: Average, Sine, Triangle, Double methods
            - **Applications**: Planting decisions, harvest timing
            - **Accuracy**: ±2°F uncertainty bands available
            """)
        
        with col2:
            st.markdown("""
            ### 🌾 Crop Drying Prediction
            - **Purpose**: Optimize hay/forage drying
            - **Factors**: Solar radiation, humidity, wind, crop type, swath config
            - **Target**: 15% moisture for safe baling
            - **Forecast**: Up to 2 months ahead
            """)

        # st.markdown("## 🔬 Enhanced Drying Model Features")
        
        col1, col2, col3 = st.columns(3)
        
        # with col1:
        #     st.markdown("""
        #     ### 🌾 Crop-Specific Coefficients
        #     - **Clover**: Fastest drying (1-2 days)
        #     - **Grass crops**: Fast drying (2-3 days)
        #     - **Alfalfa**: Slower due to thick stems (3-5 days)
        #     """)
        
        # with col2:
        #     st.markdown("""
        #     ### 📏 Swath Configuration Impact
        #     - **Thin swaths**: 40% faster drying
        #     - **Normal swaths**: Standard rate
        #     - **Thick swaths**: 40% slower drying
        #     """)
        
        # with col3:
        #     st.markdown("""
        #     ### ⏰ First-Day Optimization
        #     - **2.5x multiplier** for first 24 hours
        #     - **Maximum moisture loss** period
        #     - **Weather-dependent** adjustments
        #     """)

if __name__ == "__main__":
    main()