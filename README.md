# 🏡 AI-Powered Real Estate Price Prediction Platform

An end-to-end Machine Learning powered real estate valuation platform that predicts property prices across major Indian cities using advanced data analytics, geospatial intelligence, and feature engineering.

The platform enables users to estimate property values instantly by providing property specifications such as city, location, area, bedrooms, and amenities through an intuitive web interface powered by FastAPI.

## 🚀 Live Demo

🔗 **[Try the Live Demo](https://ai-powered-real-estate-recommendation-r0xe.onrender.com/)**

> **Live Application:** https://ai-powered-real-estate-recommendation-r0xe.onrender.com/

### 🎯 Try It Yourself

1. Select a **City**
2. Select a **Location**
3. Enter the **Property Area**
4. Select the number of **Bedrooms**
5. Choose available **Amenities**
6. Click **Predict Price**
7. Get an **AI-powered estimated property value instantly**

---

## 🚀 Project Overview

Real estate valuation is influenced by multiple factors including:

* Property Area
* Number of Bedrooms
* City
* Location
* Geographic Coordinates
* Amenities
* Property Type (Resale/New)

Traditional property valuation often requires extensive market research and expert consultation.

This project leverages Machine Learning and Geospatial Feature Engineering to provide accurate and instant property price estimations.

---

## 🎯 Business Problem

Property buyers, investors, and real estate professionals often face challenges in:

* Determining fair property prices
* Comparing locations
* Evaluating investment opportunities
* Understanding location-driven price variations

This platform addresses these challenges by providing AI-driven property valuations based on historical real estate data and location intelligence.

---

# ✨ Key Features

### 🤖 AI-Based Property Valuation

Predicts property prices using trained Machine Learning models.

### 📍 Location Intelligence

Uses Latitude and Longitude mapping for every location.

### 🏙 Multi-City Support

Supports property valuation across major Indian cities.

### 🏠 Property Amenities Analysis

Considers:

* Gymnasium
* Swimming Pool
* Club House
* Power Backup
* Lift
* AC
* Car Parking

### ⚡ Fast API Inference

Provides real-time predictions through FastAPI REST APIs.

### 🎨 Modern User Interface

Interactive frontend with:

* Dynamic City Selection
* Dynamic Location Selection
* Real-Time Predictions
* Responsive Design

---

# 🧠 Machine Learning Workflow

## 1. Data Collection

Collected historical real estate datasets containing:

* Property Details
* Location Information
* Amenities
* Price Information

---

## 2. Data Cleaning

Performed:

* Missing Value Handling
* Duplicate Removal
* Outlier Detection
* Data Standardization

Example:

```text
Delhi
New Delhi
DELHI
```

Standardized into a consistent format.

---

## 3. Feature Engineering

Created additional intelligent features:

### Area per Bedroom

```python
Area_per_Bedroom = Area / Bedrooms
```

### Geographical Interaction

```python
Lat_Long = Latitude * Longitude
```

### Luxury Score

```python
Luxury_Score =
Gym +
SwimmingPool +
ClubHouse +
PowerBackup +
LiftAvailable
```

These engineered features significantly improved model performance.

---

## 4. Geospatial Intelligence

One of the most critical challenges in real estate prediction is location representation.

Instead of using only categorical location names:

```text
Sheikh Sarai
Noida Sector 62
Whitefield
```

Locations were transformed into:

```text
Latitude
Longitude
```

coordinates, enabling the model to understand geographical relationships between properties.

---

## 5. Model Training

Multiple algorithms were evaluated:

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor

The final model was selected based on:

* R² Score
* MAE
* RMSE
* Cross Validation Performance

---

## 6. Model Deployment

The trained model was serialized using:

```python
joblib
```

and deployed using:

```python
FastAPI
```

for production-ready inference.

---

# 🏗 System Architecture

```text
┌──────────────────┐
│      User        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Frontend (HTML)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ FastAPI Backend  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Feature Engine   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ ML Pipeline      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Price Prediction │
└──────────────────┘
```

---

# 🛠 Technology Stack

## Backend

* Python
* FastAPI
* Pandas
* NumPy
* Joblib

## Machine Learning

* Scikit-Learn
* XGBoost

## Frontend

* HTML5
* CSS3
* JavaScript

## Deployment

* Uvicorn
* FastAPI

---

# 🔌 API Endpoints

## Get Available Cities

```http
GET /cities
```

Response:

```json
{
  "cities": ["Delhi", "Mumbai", "Bangalore"]
}
```

---

## Get Locations

```http
GET /locations/{city}
```

Response:

```json
{
  "locations": ["Sheikh Sarai", "Dwarka"]
}
```

---

## Predict Property Price

```http
POST /predict
```

Request:

```json
{
  "area": 1200,
  "bedrooms": 2,
  "city": "Delhi",
  "location": "Sheikh Sarai",
  "gymnasium": 1,
  "swimming_pool": 0,
  "clubhouse": 1,
  "power_backup": 1,
  "car_parking": 1,
  "lift_available": 1,
  "ac": 1,
  "resale": 0
}
```

Response:

```json
{
  "predicted_price": 12500000
}
```

---

# 🚧 Challenges Faced

## Data Quality Issues

Property datasets contained:

* Missing Values
* Duplicate Entries
* Inconsistent Location Names

Resolved through extensive preprocessing.

---

## Geographical Feature Representation

The biggest challenge was enabling the model to understand location influence on property prices.

Implemented latitude-longitude mapping and custom geospatial features to improve prediction accuracy.

---

## Model Generalization

Avoided overfitting using:

* Cross Validation
* Feature Engineering
* Hyperparameter Tuning

---

# 📈 Future Enhancements

* Deep Learning Models
* Property Recommendation Engine
* Interactive Maps Integration
* Market Trend Forecasting
* RAG-Based Real Estate Assistant
* Generative AI Property Insights

---

# 💡 Key Learnings

This project strengthened expertise in:

* Machine Learning Pipelines
* Feature Engineering
* Geospatial Data Processing
* FastAPI Development
* Model Deployment
* Real Estate Analytics
* End-to-End AI Product Development

---

# 👨‍💻 Author

**Sandeep Kumar**
📧 [Sandeepspk1797@gmail.com](mailto:Sandeepspk1797@gmail.com)

---

⭐ If you found this project useful, consider giving it a star.

🔗 **Live Demo:** https://ai-powered-real-estate-recommendation-r0xe.onrender.com/
