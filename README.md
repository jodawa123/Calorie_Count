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