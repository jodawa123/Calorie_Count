import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, current_user, login_required, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
import sys
import logging
from functools import wraps  
from logging.handlers import RotatingFileHandler
from models.user import User
from models.food_data import FoodData  # Import the FoodData class for calorie prediction
from flask_cors import CORS

# Import create_app from the current directory
from __init__ import create_app

app = create_app()
CORS(app)  # Enable Cross-Origin Resource Sharing for API endpoints
food_data = FoodData()

# Initialize LoginManager for user authentication
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Set the default login view

# Database connection function
def get_db():
    """Get a connection to the SQLite database"""
    from Calories import get_db as _get_db
    return _get_db()

def allowed_file(filename):
    """Check if a file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Database connection decorator
def with_db_connection(func):
    """
    Decorator to handle database connections and errors
    
    This decorator:
    1. Creates a database connection
    2. Passes it to the wrapped function
    3. Handles any database errors
    4. Ensures the connection is closed
    """
    @wraps(func)  # Preserve the original function's metadata
    def wrapper(*args, **kwargs):
        conn = None
        try:
            conn = get_db()
            return func(conn=conn, *args, **kwargs)
        except sqlite3.Error as e:
            app.logger.error(f"Database error: {str(e)}")
            flash('Database error occurred', 'error')
            if request.accept_mimetypes.accept_json:
                return jsonify({'success': False, 'message': 'Database error'}), 500
            return redirect(url_for('index'))
        finally:
            if conn:
                conn.close()
    return wrapper

# Configure logging
if not app.debug:
    if not os.path.exists('logs'):
        os.mkdir('logs')
    file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('App startup')

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    """Load a user from the database by ID"""
    conn = get_db()
    user_data = conn.execute(
        "SELECT * FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    conn.close()
    if user_data:
        return User(user_data['id'], user_data['username'], user_data['password'])
    return None

# Routes
@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not username or not password:
            flash('Username and password are required', 'error')
            return redirect(url_for('login'))
        
        conn = get_db()
        try:
            user_data = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
            
            if user_data and check_password_hash(user_data['password'], password):
                user = User(user_data['id'], user_data['username'], user_data['password'])
                login_user(user)
                next_page = request.args.get('next')
                return redirect(next_page or url_for('dashboard'))
            else:
                flash('Invalid username or password', 'error')
        except sqlite3.Error as e:
            app.logger.error(f"Login error: {str(e)}")
            flash('An error occurred during login', 'error')
        finally:
            conn.close()
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Input validation
        if not username or not password:
            flash('Username and password are required', 'error')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('register'))
        
        if len(password) < 8:
            flash('Password must be at least 8 characters', 'error')
            return redirect(url_for('register'))
        
        if len(username) < 3:
            flash('Username must be at least 3 characters', 'error')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        conn = get_db()
        
        try:
            # Check if username already exists
            existing_user = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
            
            if existing_user:
                flash('Username already exists', 'error')
                return redirect(url_for('register'))
            
            # Insert new user
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, hashed_password)
            )
            conn.commit()
            
            # Get the newly created user
            user_data = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
            
            if not user_data:
                raise Exception("Failed to retrieve created user")
            
            # Log in the new user
            user = User(user_data['id'], user_data['username'], user_data['password'])
            login_user(user)
            
            flash('Registration successful!', 'success')
            return redirect(url_for('dashboard'))
            
        except sqlite3.IntegrityError:
            flash('Username already exists', 'error')
            app.logger.error(f"Registration failed - Username already exists: {username}")
        except sqlite3.Error as e:
            flash('Database error during registration', 'error')
            app.logger.error(f"Database error during registration: {str(e)}")
        except Exception as e:
            flash('An unexpected error occurred', 'error')
            app.logger.error(f"Unexpected error during registration: {str(e)}")
        finally:
            conn.close()
    
    return render_template('register.html')

@app.route('/dashboard')
@login_required
@with_db_connection
def dashboard(conn):
    """Display the user's dashboard with food logs and statistics"""
    today = datetime.now().strftime('%Y-%m-%d')
    week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    try:
        # Get today's food logs
        today_logs = conn.execute(
            """SELECT id, food_name, calories, protein, carbs, fat, portion, 
                      strftime('%H:%M', date_added) as time 
               FROM food_logs 
               WHERE user_id = ? AND date(date_added) = ? 
               ORDER BY date_added DESC""",
            (current_user.id, today)
        ).fetchall()
        
        # Get weekly calorie totals
        weekly_logs = conn.execute(
            """SELECT date(date_added) as log_date, 
                      SUM(calories * portion / 100) as total_calories 
               FROM food_logs 
               WHERE user_id = ? AND date_added >= ?
               GROUP BY date(date_added) 
               ORDER BY log_date""",
            (current_user.id, week_ago)
        ).fetchall()
        
        # Calculate today's totals
        totals = {
            'calories': sum(log['calories'] * log['portion'] / 100 for log in today_logs) if today_logs else 0,
            'protein': sum(log['protein'] * log['portion'] / 100 for log in today_logs) if today_logs else 0,
            'carbs': sum(log['carbs'] * log['portion'] / 100 for log in today_logs) if today_logs else 0,
            'fat': sum(log['fat'] * log['portion'] / 100 for log in today_logs) if today_logs else 0
        }
        
        return render_template(
            'dashboard.html',
            today_logs=today_logs,
            weekly_logs=weekly_logs,
            total_calories=round(totals['calories'], 1),
            total_protein=round(totals['protein'], 1),
            total_carbs=round(totals['carbs'], 1),
            total_fat=round(totals['fat'], 1)
        )
    except sqlite3.Error as e:
        app.logger.error(f"Error loading dashboard data: {str(e)}")
        flash('Error loading dashboard data', 'error')
        return render_template('dashboard.html', today_logs=[], weekly_logs=[])
    

@app.route('/api/search_food')
@login_required
def search_food():
    """API endpoint for searching food items"""
    query = request.args.get('query', '').strip()
    if not query:
        return jsonify({
            'status': 'error',
            'message': 'Search query is required'
        }), 400
        
    try:
        # Check if food_data is properly initialized
        if not food_data.df is not None:
            app.logger.error("Food dataset not loaded")
            return jsonify({
                'status': 'error',
                'message': 'Food database not initialized'
            }), 500
            
        if not food_data.model:
            app.logger.error("Prediction model not loaded")
            return jsonify({
                'status': 'error',
                'message': 'Prediction model not initialized'
            }), 500
            
        result = food_data.search(query)
        if result:
            return jsonify({
                'status': 'success',
                'data': {
                    'food_name': result['food_name'],
                    'category': result['category'],
                    'calories': result['calories'],
                    'nutritional_data': result['nutritional_data']
                }
            })
        return jsonify({
            'status': 'not_found',
            'message': f'No food items found matching "{query}"'
        })
    except Exception as e:
        app.logger.error(f"Search error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error searching for food: {str(e)}'
        }), 500
        
        
        
@app.route('/api/predict_calories', methods=['POST'])
@login_required
def predict_calories():
    """API endpoint for predicting calories from nutritional values"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get food name if provided, otherwise use "Custom food"
        food_name = data.get('food_name', 'Custom food')
        
        # Convert to the format expected by the FoodData model
        input_data = {
            'protein': float(data.get('Protein (g)', data.get('protein', 0))),
            'carbohydrates': float(data.get('Carbohydrates (g)', data.get('carbohydrates', 0))),
            'fat': float(data.get('Fat (g)', data.get('fat', 0))),
            'fiber': float(data.get('Fiber (g)', data.get('fiber', 0))),
            'sugars': float(data.get('Sugars (g)', data.get('sugars', 0))),
            'sodium': float(data.get('Sodium (mg)', data.get('sodium', 0))),
            'cholesterol': float(data.get('Cholesterol (mg)', data.get('cholesterol', 0)))
        }
        
        # Use the model for prediction if available
        if food_data.model:
            predicted_calories = food_data.predict(input_data)
        else:
            # Fallback to basic formula if model not available
            predicted_calories = (input_data['protein'] * 4) + \
                               (input_data['carbohydrates'] * 4) + \
                               (input_data['fat'] * 9)
        
        return jsonify({
            'predicted_calories': predicted_calories,
            'food_name': food_name
        })
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': f'Error predicting calories: {str(e)}'
        }), 500

@app.route('/api/add_food', methods=['POST'])
@login_required
@with_db_connection
def add_food(conn):
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400
        
        # Handle both database foods (with food_id) and custom foods (without food_id)
        if 'food_id' in data and data['food_id']:
            # Get food from database
            food = conn.execute(
                "SELECT * FROM foods WHERE id = ?", 
                (data['food_id'],)
            ).fetchone()
            
            if not food:
                return jsonify({
                    'success': False, 
                    'message': 'Food not found'
                }), 404
                
            food_name = food['name']
            calories = food['calories']
            protein = food['protein']
            carbs = food['carbs']
            fat = food['fat']
            
        else:
            # Custom food entry with direct nutrition data
            food_name = data.get('food_name', 'Custom food')
            calories = float(data.get('calories', 0))
            protein = float(data.get('protein', 0))
            carbs = float(data.get('carbs', 0))
            fat = float(data.get('fat', 0))
        
        try:
            portion = float(data['portion'])
            if portion <= 0:
                raise ValueError("Portion must be positive")
        except (ValueError, TypeError):
            return jsonify({
                'success': False, 
                'message': 'Invalid portion size'
            }), 400
        
        conn.execute(
            """INSERT INTO food_logs 
               (user_id, food_name, calories, protein, carbs, fat, portion)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (current_user.id, food_name, calories, 
             protein, carbs, fat, portion)
        )
        conn.commit()
        
        return jsonify({'success': True})
    except sqlite3.Error as e:
        app.logger.error(f"Add food error: {str(e)}")
        return jsonify({
            'success': False, 
            'message': 'Database error'
        }), 500
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            'success': False, 
            'message': 'An error occurred'
        }), 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)