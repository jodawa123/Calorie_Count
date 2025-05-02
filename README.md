# Calories Tracker Application

A web application for tracking food intake and calories using machine learning for calorie prediction.

## Features

- ✅ User authentication (login/register)
- 🔍 Food search with calorie prediction
- 📝 Food logging and tracking
- 📊 Dashboard with daily calorie summary

## Prerequisites

- 🐳 Docker and Docker Compose
- 🐍 Python 3.8+ (for local development)

## Setup

```bash
# Clone the repository
git clone https://github.com/jodawa123/Calorie_Count.git
cd Calorie_Count

## Screenshots

### 1. Main Interface
<img src="https://raw.githubusercontent.com/jodawa123/Calorie_Count/master/images/overall.png" width="600">

### 2. Adding Food
<img src="https://raw.githubusercontent.com/jodawa123/Calorie_Count/master/images/testing.png" width="600">

### 3. Results
<img src="https://raw.githubusercontent.com/jodawa123/Calorie_Count/master/images/results.png" width="600">

### 4. Data View
<img src="https://raw.githubusercontent.com/jodawa123/Calorie_Count/master/images/new.png" width="600">

# Start the application (using Docker)
docker-compose up --build
Project Screenshots
Feature	Preview
Main Dashboard	Dashboard
Add New Food	Add Food
Prediction Results	Results
Data Visualization	Visualization
Development
bash
# Set up virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
flask run