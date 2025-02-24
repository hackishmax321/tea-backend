from fastapi import FastAPI, HTTPException
from tweepy import OAuth1UserHandler, API
from dotenv import load_dotenv
from pydantic import BaseModel
import os
import joblib
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError
import traceback
import time
from collections import defaultdict
from datetime import datetime
import json
import http.client
from typing import Dict, List
import numpy as np


# Load environment variables
load_dotenv()

# Twitter API credentials
API_KEY = "6496790f8bmsha07b1cf7256f9c2p1995fbjsne7ca8be11817"
API_SECRET_KEY = "SKHmppgEHdjhWo5UdAOuyccDELGema5KiNqZM2VFGBxi03sRLF"
ACCESS_TOKEN = "Q0NKb2k1cEhrWEpZMXpVRlN4S2o6MTpjaQ"
ACCESS_TOKEN_SECRET = "OEe2ThJ1z5-UIrpoh92Jfvxi6ryKKkOjJoCbGCikwkpRXX03HF"

# Authenticate with Twitter API
auth = OAuth1UserHandler(API_KEY, API_SECRET_KEY, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
twitter_api = API(auth)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend's origin for better security
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize pytrends
pytrends = TrendReq()

# Load Models
sales_model = joblib.load('sales/model_sales_quantity_new.joblib')

types_model = joblib.load('tea_type_demand_rf.joblib')

models = {
    "china": joblib.load("export_demand/export_china_rf.joblib"),
    "germany": joblib.load("export_demand/export_Germany_rf.joblib"),
    "iran": joblib.load("export_demand/export_Iran_rf.joblib"),
    "japan": joblib.load("export_demand/export_JPA_rf.joblib"),
    "russia": joblib.load("export_demand/export_RUSS_rf.joblib"),
    "uk": joblib.load("export_demand/export_UK_rf.joblib"),
    "usa": joblib.load("export_demand/export_USA_rf.joblib"),
}

    
elevation_map = {'High grown': 0, 'Low grown': 1, 'Mid grown': 2, 'Unknown': 3}

# Pydantic model for Sales Predict
class PredictionInput(BaseModel):
    year: float
    dollar_rate: float
    elevation: str  # Input elevation as a string
    avg_price: float
    sales_code: int

@app.post("/predict/sales-quantity")
async def predict_sales_quantity(input_data: PredictionInput):
    
    try:
        # Encode elevation
        elevation_encoded = elevation_map.get(input_data.elevation, elevation_map['Unknown'])
        
        model_input = [
            [
                input_data.year,
                input_data.sales_code,
                input_data.dollar_rate,
                elevation_encoded,
                input_data.avg_price,
            ]
        ]
        
        prediction = sales_model.predict(model_input)
        
        return {"predicted_quantity": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
    
# Endpoint for Export Demand by country
class DemandPredictionInput(BaseModel):
    year: int
    month: int
    CH_CPI: float
    Type: str  # Type should match the categories used during training
    country: str  # Country for which prediction is required

@app.post("/predict/demand")
async def predict_demand(input_data: DemandPredictionInput):
    print(input_data.CH_CPI)
    
    try:
        # Ensure the country has a corresponding model
        country = input_data.country.lower()
        if country not in models:
            raise HTTPException(
                status_code=400,
                detail=f"No model available for the country: {country.capitalize()}",
            )
        
        # Load the correct model
        selected_model = models[country]

        # Prepare input data for the model
        type_encoding = {"Black": 0, "Green": 1}  # Adjust based on your dataset encoding
        type_encoded = type_encoding.get(input_data.Type, -1)
        if type_encoded == -1:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid Type: {input_data.Type}. Allowed values are 'Black' or 'Green'.",
            )

        model_input = [[
            input_data.year,
            input_data.month,
            input_data.CH_CPI,
            type_encoded
        ]]

        # Predict demand
        prediction = selected_model.predict(model_input)

        yr_weights_balance = input_data.year - 2020
        if(yr_weights_balance > 0):
            prediction[0] = prediction[0] + ((prediction[0]*yr_weights_balance)/100)

        return {
            "country": country.capitalize(),
            "predicted_demand": prediction[0],
            "month": input_data.month
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}") 

# Tea Types
# Categorical mappings
processing_method_mapping = {'CTC TEA': 0, 'GREEN TEA': 1, 'ORTHODOX': 2}
elevation_mapping = {'HIGH': 0, 'LOW': 1, 'MEDIUM': 2}

# Request model for validation
class PredictionRequest(BaseModel):
    year: int
    month: int
    processing_method: str
    elevation: str
    inflation_rate: float

@app.post("/predict/local-market-release")
async def predict_local_market_release(data: PredictionRequest):
    """
    Predicts the Local market Release Quantity (Kg) using the trained Random Forest model.
    """
    if types_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly.")

    # Map the categorical inputs to encoded values
    processing_method_encoded = processing_method_mapping.get(data.processing_method)
    elevation_encoded = elevation_mapping.get(data.elevation)

    if processing_method_encoded is None or elevation_encoded is None:
        raise HTTPException(status_code=400, detail="Invalid processing method or elevation value provided.")

    # Prepare the input data for prediction
    input_data = pd.DataFrame([{
        'year': data.year,
        'month': data.month,
        'Processing Method': processing_method_encoded,
        'Elevation': elevation_encoded,
        'whole production Quantity (Kg)': 0,  # Default or dynamic input
        'production Total (kg)': 0,          # Default or dynamic input
        'inflation rate': data.inflation_rate
    }])

    # Handle missing values in features
    input_data = input_data.fillna(input_data.median())

    try:
        # Predict using the trained model
        prediction = types_model.predict(input_data)
        return {"predicted_local_market_release_quantity": prediction[0]}
    except Exception as e:
        error_message = traceback.format_exc()
        print(error_message)
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


class PredictionRequestX(BaseModel):
    elevation: str
    inflation_rate: float


@app.get("/predict/local-market-release/{tea_type}")
async def predict_local_market_release(tea_type: str, data: PredictionRequestX):
    """
    Predicts the Local Market Release Quantity (Kg) for months 1 to 10 of 2025 based on tea type and input data.
    """
    if types_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly.")

    # Map the categorical inputs to encoded values
    processing_method_encoded = processing_method_mapping.get(tea_type.upper())
    elevation_encoded = elevation_mapping.get(data.elevation)

    if processing_method_encoded is None:
        raise HTTPException(status_code=400, detail="Invalid tea type provided.")
    if elevation_encoded is None:
        raise HTTPException(status_code=400, detail="Invalid elevation value provided.")

    try:
        # Generate predictions for months 1 through 10
        predictions = []
        for month in range(1, 11):
            input_data = pd.DataFrame([{
                'year': 2025,
                'month': month,
                'Processing Method': processing_method_encoded,
                'Elevation': elevation_encoded,
                'whole production Quantity (Kg)': 0,  # Default or dynamic input
                'production Total (kg)': 0,          # Default or dynamic input
                'inflation rate': data.inflation_rate
            }])

            # Handle missing values in features
            input_data = input_data.fillna(input_data.median())

            # Predict using the trained model
            prediction = types_model.predict(input_data)
            predictions.append({
                "month": month,
                "predicted_quantity": prediction[0]
            })

        return {
            "tea_type": tea_type,
            "year": 2025,
            "elevation": data.elevation,
            "inflation_rate": data.inflation_rate,
            "predictions": predictions
        }

    except Exception as e:
        error_message = traceback.format_exc()
        print(error_message)
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


# Local Demand 

loaded_model_1 = joblib.load("local_market_demand/lm_random_forest_model.joblib")
loaded_model_2 = joblib.load("local_market_demand/lm_lgbm_model.joblib")
loaded_model_3 = joblib.load("local_market_demand/lm_etr_model.joblib")

# Dictionary to store models
MULTI_MODELS_DEMAND = {
    "Random Forest": loaded_model_1,
    "LightGBM": loaded_model_2,
    "Extra Trees": loaded_model_3
}

loaded_model_wp1 = joblib.load("whole_production/wp_random_forest_model.joblib")
loaded_model_wp2 = joblib.load("whole_production/wp_lgbm_model.joblib")
loaded_model_wp3 = joblib.load("whole_production/wp_etr_model.joblib")

# Dictionary to store models
MULTI_MODELS_WHOLE_PROD = {
    "Random Forest": loaded_model_wp1,
    "LightGBM": loaded_model_wp2,
    "Extra Trees": loaded_model_wp3
}

class TeaProductionInput(BaseModel):
    year: int
    month: int
    processing_method: str
    elevation: str
    production_total: float
    inflation_rate: float
    temp_avg: float
    rain: float
    humidity_day: float
    humidity_night: float

class TeaProductionInputUpdated(BaseModel):
    year: int
    month: int
    processing_method: str
    elevation: str
    inflation_rate: float
    temp_avg: float
    rain: float
    humidity_day: float
    humidity_night: float

# Dummy label encoding function (Replace with actual encoding logic)
def encode_labels(processing_method, elevation):
    processing_method_mapping = {"Orthodox": 0, "CTC": 1, "Green": 2}
    elevation_mapping = {"Low": 0, "Medium": 1, "High": 2}

    return (processing_method_mapping.get(processing_method, -1), 
            elevation_mapping.get(elevation, -1))

# Prediction function
def predict_tea_production_ensemble(year, month, processing_method, elevation, production_total, 
                                    inflation_rate, temp_avg, rain, humidity_day, humidity_night):
    try:
        # Encode labels
        processing_method_encoded, elevation_encoded = encode_labels(processing_method, elevation)
        
        if processing_method_encoded == -1 or elevation_encoded == -1:
            return {"error": "Invalid processing method or elevation label."}
        
        # Create input data as DataFrame
        input_data = pd.DataFrame([[year, month, processing_method_encoded, elevation_encoded, production_total, 
                                    inflation_rate, temp_avg, rain, humidity_day, humidity_night]],
                                  columns=["year", "month", "Processing Method", "Elevation", "production Total (kg)",
                                           "inflation rate", "Temp AVG", "Rain", "Humidity Day", "Humidity Night"])

        # Get predictions from all models
        predictions = np.array([model.predict(input_data)[0] for model in MULTI_MODELS_DEMAND.values()])
        
        # Average predictions
        final_prediction = np.mean(predictions)

        return {"predicted_tea_production": final_prediction}
    
    except Exception as e:
        return {"error": str(e)}

# FastAPI endpoint
@app.post("/predict-tea-production")
async def predict_tea_production(input_data: TeaProductionInput):
    result = predict_tea_production_ensemble(
        input_data.year, input_data.month, input_data.processing_method, input_data.elevation, 
        input_data.production_total, input_data.inflation_rate, input_data.temp_avg, 
        input_data.rain, input_data.humidity_day, input_data.humidity_night
    )
    return result

def predict_tea_production_weighted(year, month, processing_method, elevation, production_total, 
                                    inflation_rate, temp_avg, rain, humidity_day, humidity_night):
    try:
        # Encode labels
        processing_method_encoded, elevation_encoded = encode_labels(processing_method, elevation)
        
        if processing_method_encoded == -1 or elevation_encoded == -1:
            return {"error": "Invalid processing method or elevation label."}
        
        # Create input data as DataFrame
        input_data = pd.DataFrame([[year, month, processing_method_encoded, elevation_encoded, production_total, 
                                    inflation_rate, temp_avg, rain, humidity_day, humidity_night]],
                                  columns=["year", "month", "Processing Method", "Elevation", "production Total (kg)",
                                           "inflation rate", "Temp AVG", "Rain", "Humidity Day", "Humidity Night"])

        # Get predictions from all models
        predictions = np.array([model.predict(input_data)[0] for model in MULTI_MODELS_DEMAND.values()])
        
        # Define model weights (assign higher weights to better models)
        weights = np.array([0.4, 0.3, 0.3])  # Example: Higher weight to Random Forest
        
        # Compute weighted prediction
        yr_weights_balance = year - 2020

        final_prediction = np.sum(predictions * weights)
        if(yr_weights_balance > 0):
            final_prediction = final_prediction + ((final_prediction*yr_weights_balance)/100)
        

        return {"predicted_tea_production": final_prediction}
    
    except Exception as e:
        return {"error": str(e)}

# FastAPI endpoint
@app.post("/predict-tea-production-weighted")
async def predict_tea_production(input_data: TeaProductionInput):
    result = predict_tea_production_weighted(
        input_data.year, input_data.month, input_data.processing_method, input_data.elevation, 
        input_data.production_total, input_data.inflation_rate, input_data.temp_avg, 
        input_data.rain, input_data.humidity_day, input_data.humidity_night
    )

    return result

def predict_tea_whole_production_weighted(year, month, processing_method, elevation, 
                                    inflation_rate, temp_avg, rain, humidity_day, humidity_night):
    try:
        # Encode labels
        processing_method_encoded, elevation_encoded = encode_labels(processing_method, elevation)
        
        if processing_method_encoded == -1 or elevation_encoded == -1:
            return {"error": "Invalid processing method or elevation label."}
        
        # Create input data as DataFrame
        input_data = pd.DataFrame([[year, month, processing_method_encoded, elevation_encoded, 
                                    inflation_rate, temp_avg, rain, humidity_day, humidity_night]],
                                  columns=["year", "month", "Processing Method", "Elevation",
                                           "inflation rate", "Temp AVG", "Rain", "Humidity Day", "Humidity Night"])

        # Get predictions from all models
        predictions = np.array([model.predict(input_data)[0] for model in MULTI_MODELS_WHOLE_PROD.values()])
        
        # Define model weights (assign higher weights to better models)
        weights = np.array([0.4, 0.3, 0.3])  # Example: Higher weight to Random Forest
        
        # Compute weighted prediction
        yr_weights_balance = year - 2020

        final_prediction = np.sum(predictions * weights)
        if(yr_weights_balance > 0):
            final_prediction = final_prediction + ((final_prediction*yr_weights_balance)/100)
        

        return {"predicted_tea_whole_production": final_prediction}
    
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict-tea-whole-production-weighted")
async def predict_tea_whole_production(input_data: TeaProductionInputUpdated):
    result = predict_tea_whole_production_weighted(
        input_data.year, input_data.month, input_data.processing_method, input_data.elevation, 
        input_data.inflation_rate, input_data.temp_avg, 
        input_data.rain, input_data.humidity_day, input_data.humidity_night
    )


    return result

# Trend Analysis
class TrendRequest(BaseModel):
    topics: list[str]



@app.post("/get-google-trends")
async def get_google_trends(request: TrendRequest):
    """
    Fetches Google Trends data for the specified topics over the last 5 years.
    """
    pytrends = TrendReq(hl='en-US', tz=360)
    topics = request.topics
    data = {}

    for topic in topics:
        while True:
            try:
                pytrends.build_payload([topic], timeframe='today 5-y', geo='', gprop='')
                interest_over_time = pytrends.interest_over_time()
                data[topic] = interest_over_time[topic].tolist()
                time.sleep(60)  # Increase delay
                break
            except TooManyRequestsError:
                print(f"Too many requests for topic: {topic}. Retrying after 5 minutes.")
                time.sleep(300)  # Wait 5 minutes before retrying

    return {"trend_data": data}


@app.post("/get-google-trends-dates")
async def get_google_trends(request: TrendRequest):
    """
    Fetches Google Trends data for the specified topics over the last 5 years.
    """
    pytrends = TrendReq(hl='en-US', tz=360)
    topics = request.topics
    trend_data = {}
    shared_dates = []

    for idx, topic in enumerate(topics):
        while True:
            try:
                # Build the payload for the topic
                pytrends.build_payload([topic], timeframe='today 5-y', geo='', gprop='')
                interest_over_time = pytrends.interest_over_time()

                # For the first topic, extract and store shared dates
                if idx == 0:
                    shared_dates = interest_over_time.index.strftime('%Y-%m-%d').tolist()

                # Extract interest counts for the topic
                trend_data[topic] = interest_over_time[topic].tolist()

                time.sleep(60)  # Delay to prevent hitting request limits
                break
            except TooManyRequestsError:
                print(f"Too many requests for topic: {topic}. Retrying after 5 minutes.")
                time.sleep(300)  # Wait before retrying

    # Response format
    return {
        "trend_data": {
            **trend_data,
            "dates": shared_dates
        }
    }



# Anlyze Twitter 
class YearlyPostCount(BaseModel):
    year: int
    post_count: int

@app.get("/fetch-and-analyze-posts")
async def fetch_and_analyze_posts(query: str = "tea", count: int = 20) -> List[YearlyPostCount]:
    """
    Fetches posts from the Twitter API, processes post data, groups posts by year, and counts them.

    Args:
        query (str): Search query for Twitter data.
        count (int): Number of posts to retrieve.

    Returns:
        List[YearlyPostCount]: List of post counts grouped by year.
    """
    try:
        # Make request to Twitter API
        data_string = fetch_twitter_data(query, count)

        # Extract posts with content and dates
        posts = extract_posts_with_dates(data_string)
        if not posts:
            raise HTTPException(status_code=400, detail="No valid posts found in the data.")

        # Group posts by year and count them
        year_counts = group_posts_by_year(posts)

        # Convert to sorted list of YearlyPostCount
        yearly_post_counts = sorted(
            [YearlyPostCount(year=year, post_count=count) for year, count in year_counts.items()],
            key=lambda x: x.year
        )

        return yearly_post_counts
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def fetch_twitter_data(query: str, count: int) -> str:
    """
    Fetches Twitter data using the Twitter API.

    Args:
        query (str): Search query for Twitter data.
        count (int): Number of posts to retrieve.

    Returns:
        str: JSON string containing Twitter data.
    """
    conn = http.client.HTTPSConnection("twitter241.p.rapidapi.com")

    headers = {
        'x-rapidapi-key': "6496790f8bmsha07b1cf7256f9c2p1995fbjsne7ca8be11817",
        'x-rapidapi-host': "twitter241.p.rapidapi.com"
    }

    conn.request("GET", f"/search-v2?type=Top&count={count}&query={query}", headers=headers)
    res = conn.getresponse()
    data = res.read()

    return data.decode("utf-8")

# Helper function: Extract posts with their dates
def extract_posts_with_dates(data_string):
    posts_with_dates = []
    data = json.loads(data_string)
    instructions = data.get("result", {}).get("timeline", {}).get("instructions", [])

    for instruction in instructions:
        entries = instruction.get("entries", [])
        for entry in entries:
            content = entry.get("content", {})
            if content.get("__typename") == "TimelineTimelineModule":
                items = content.get("items", [])
                for item in items:
                    user_content = item.get("item", {}).get("itemContent", {}).get("user_results", {}).get("result", {}).get("legacy", {})
                    description = user_content.get("description", "")
                    created_at = user_content.get("created_at", "")

                    if description and created_at:
                        posts_with_dates.append({
                            "content": description,
                            "date": created_at
                        })

    return posts_with_dates

# Helper function: Group posts by year
def group_posts_by_year(posts):
    year_counts = defaultdict(int)
    for post in posts:
        try:
            post_date = datetime.strptime(post['date'], '%a %b %d %H:%M:%S %z %Y')
            year = post_date.year
            year_counts[year] += 1
        except Exception as e:
            print(f"Error parsing date for post: {post}, Error: {e}")

    return dict(year_counts)


# Facebook Analyssis
def count_posts_by_year(data):
    """
    Counts posts based on year from the timestamp.

    Args:
        data (dict): The parsed JSON response containing the posts.

    Returns:
        dict: A dictionary with years as keys and post counts as values.
    """
    year_count = defaultdict(int)

    # Loop through each post in the response
    for post in data.get("results", []):
        timestamp = post.get("timestamp")
        if timestamp:
            # Convert timestamp to a datetime object
            date = datetime.utcfromtimestamp(timestamp)
            year = date.year
            year_count[year] += 1

    # Sorting the years in ascending order and return the result
    sorted_year_count = dict(sorted(year_count.items()))
    return sorted_year_count


async def fetch_and_count_posts_facebook(keywords):
    """
    Fetches posts for a list of keywords and counts them by year.

    Args:
        keywords (list): A list of keywords to fetch posts for.

    Returns:
        dict: A dictionary with keywords as keys and year counts as values.
    """
    # API connection details
    conn = http.client.HTTPSConnection("facebook-scraper3.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': "6496790f8bmsha07b1cf7256f9c2p1995fbjsne7ca8be11817",
        'x-rapidapi-host': "facebook-scraper3.p.rapidapi.com"
    }

    results = {}

    for query in keywords:
        try:
            # API request for the given keyword
            conn.request("GET", f"/search/posts?query={query}", headers=headers)
            res = conn.getresponse()
            data = res.read()

            # Parse the response data
            response_data = json.loads(data.decode("utf-8"))

            # Count posts by year
            year_counts = count_posts_by_year(response_data)

            # Store the results for this keyword
            results[query] = year_counts

        except Exception as e:
            # Handle any errors for the keyword
            results[query] = {"error": str(e)}

    return results

class KeywordsRequest(BaseModel):
    keywords: List[str]

@app.post("/count_posts_by_year_facebook")
async def get_post_counts_facebook(request: KeywordsRequest):
    """
    Fetches posts for a list of keywords and counts them by year.

    Args:
        keywords (list): A list of keywords to fetch posts for.

    Returns:
        dict: A dictionary with keywords as keys and year counts as values.
    """
    # Fetch and count posts based on the provided keywords
    result = await fetch_and_count_posts_facebook(request.keywords)
    print(result)

    # Return the results
    return result


# Instagram
def count_items_by_year_month(data):
    """
    Counts items based on year and month from the device_timestamp.

    Args:
        data (dict): The parsed JSON response containing the items.

    Returns:
        dict: A dictionary with keys as (year, month) and values as counts.
    """
    grouped_data = defaultdict(int)

    # Loop through each item in the data
    for item in data.get("data", {}).get("items", []):
        timestamp = item.get("device_timestamp")
        if timestamp:
            # Convert timestamp to seconds (assuming timestamp is in microseconds)
            timestamp_seconds = timestamp / 1e6
            date = datetime.fromtimestamp(timestamp_seconds)
            key = (date.year, date.month)
            grouped_data[key] += 1

    # Convert defaultdict to a standard dictionary with formatted keys
    return {f"{year}-{month:02d}": count for (year, month), count in grouped_data.items()}


async def fetch_and_count_keywords_instagram(keywords):
    """
    Fetches data for each keyword and counts items by year and month.

    Args:
        keywords (list): List of hashtags to fetch data for.

    Returns:
        dict: A dictionary with each keyword as a key and counts as values, sorted by date.
    """
    # API connection details
    conn = http.client.HTTPSConnection("instagram-scraper-api2.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': "6496790f8bmsha07b1cf7256f9c2p1995fbjsne7ca8be11817",
        'x-rapidapi-host': "instagram-scraper-api2.p.rapidapi.com"
    }

    results = {}

    for keyword in keywords:
        try:
            # API request for the current keyword
            conn.request("GET", f"/v1/hashtag?hashtag={keyword}", headers=headers)
            res = conn.getresponse()
            data = res.read()

            # Parse the response data
            response_data = json.loads(data.decode("utf-8"))

            # Count items by year and month
            counts = count_items_by_year_month(response_data)

            # Sort the counts by date
            sorted_counts = {k: counts[k] for k in sorted(counts)}

            results[keyword] = sorted_counts
        except Exception as e:
            # Handle any errors and log the issue for the current keyword
            results[keyword] = {"error": str(e)}

    return results


@app.post("/count_items_by_year_month_instagram")
async def get_item_counts_instagram(request: KeywordsRequest):
    """
    Fetches Instagram data for a list of keywords and counts items by year and month.

    Args:
        request (KeywordsRequest): A Pydantic model containing a list of keywords.

    Returns:
        dict: A dictionary with each keyword as a key and counts by year-month as values.
    """
    # Fetch and count items based on the provided keywords
    result = await fetch_and_count_keywords_instagram(request.keywords)

    # Return the results
    return result
