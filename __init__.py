from flask import Flask
from flask_login import LoginManager
from models.user import User
from database import init_db, get_db
import os

# Initialize extensions
login_manager = LoginManager()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    
    # Initialize extensions
    login_manager.init_app(app)
    login_manager.login_view = 'login'
    
    # Initialize database
    with app.app_context():
        init_db()
    
    # Initialize nutrition data (only once)
    if not hasattr(app, 'nutrition_data'):
        from models.food_data import FoodData
        data_path = os.path.join('data', 'cleaned_nutrition_dataset.csv')
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Nutrition dataset not found at {data_path}")
        
        app.nutrition_data = FoodData(data_path)
        app.logger.info("Nutrition data initialized successfully")
    
    return app

@login_manager.user_loader
def load_user(user_id):
    """Flask-Login user loader callback"""
    conn = get_db()
    try:
        user_data = conn.execute(
            'SELECT * FROM users WHERE id = ?', 
            (user_id,)
        ).fetchone()
        if user_data:
            return User(user_data['id'], user_data['username'], user_data['password'])
        return None
    finally:
        conn.close()