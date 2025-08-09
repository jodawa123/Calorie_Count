import pandas as pd
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from clean_dataset import STANDARD_VALUES, CATEGORY_RULES

class FoodData:
    def __init__(self, data_file='data/cleaned_nutrition_dataset.csv'):
        """
        Initialize the FoodData class with default parameters and setup directories.
        
        Args:
            data_file (str): Path to the nutrition dataset CSV file
        """
        self.model = None
        self.preprocessor = None
        self.df = None
        
        # Core nutritional features that directly affect calories
        self.feature_columns = [
            'Protein (g)', 'Carbohydrates (g)', 'Fat (g)',
            'Fiber (g)'  # Added fiber as it affects net carbs
        ]
        self.target_column = 'Calories (kcal)'
        
        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        self.model_path = 'models/nutrition_model.joblib1'
        
        self.initialize_data(data_file)

    def initialize_data(self, data_file):
        """Initialize or rebuild the data and model"""
        try:
            print("\n🔧 Initializing food data...")
            if os.path.exists(data_file):
                print(f"📂 Found dataset at {data_file}")
                self.df = pd.read_csv(data_file)
                
                # Verify dataset structure
                required_columns = ['Food_Item', 'Category'] + self.feature_columns + [self.target_column]
                missing_columns = [col for col in required_columns if col not in self.df.columns]
                if missing_columns:
                    raise ValueError(f"Dataset missing required columns: {missing_columns}")
                
                # Clean the data
                self.df = self.clean_data(self.df)
                print(f"✅ Loaded and cleaned dataset with {len(self.df)} entries")
                
                try:
                    self.load_model()
                    print("✅ Loaded existing model")
                except Exception as e:
                    print(f"⚠️ Model loading failed: {str(e)}")
                    print("🔄 Training new model...")
                    self.process_and_train(data_file)
            else:
                print(f"❌ Dataset not found at {data_file}")
                raise FileNotFoundError(f"Required dataset file not found: {data_file}")
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                raise e
            print(f"❌ Initialization failed: {str(e)}")
            raise

    def clean_data(self, df):
        """
        Clean and preprocess the nutrition dataset with enhanced validation.
        
        Args:
            df (pd.DataFrame): Raw nutrition dataset
            
        Returns:
            pd.DataFrame: Cleaned and validated dataset
        """
        df = df.copy()
        
        # Convert columns to numeric
        numeric_cols = ['Calories (kcal)', 'Protein (g)', 'Carbohydrates (g)', 'Fat (g)', 'Fiber (g)']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with missing values
        df = df.dropna(subset=numeric_cols)
        
        # Calculate calories using 4-4-9 rule
        df['Calculated_Calories'] = (
            df['Protein (g)'] * 4 + 
            df['Carbohydrates (g)'] * 4 + 
            df['Fat (g)'] * 9
        )
        
        # Validate entries against category rules
        valid_entries = []
        replaced_entries = 0
        
        for idx, row in df.iterrows():
            food = row['Food_Item']
            category = row['Category']
            
            # Skip if category or food not in our standards
            if category not in STANDARD_VALUES or food not in STANDARD_VALUES[category]:
                continue
                
            # Get validation rules for this category
            rules = CATEGORY_RULES[category]
            standard = STANDARD_VALUES[category][food]
            
            # Check if values are within reasonable bounds
            is_valid = True
            for col, (min_val, max_val) in rules.items():
                value = row[col]
                if not (min_val <= value <= max_val):
                    is_valid = False
                    break
            
            # Check if calculated calories are within 20% of actual calories
            if is_valid:
                actual_calories = row['Calories (kcal)']
                calculated_calories = row['Calculated_Calories']
                if abs(actual_calories - calculated_calories) > (actual_calories * 0.2):
                    is_valid = False
            
            if is_valid:
                valid_entries.append(row)
            else:
                # Replace with standard values
                new_row = row.copy()
                for col in ['Calories (kcal)', 'Protein (g)', 'Carbohydrates (g)', 'Fat (g)', 'Fiber (g)']:
                    new_row[col] = standard[col]
                valid_entries.append(new_row)
                replaced_entries += 1
        
        # Create cleaned dataframe
        self.df = pd.DataFrame(valid_entries)
        print(f"✅ Cleaned dataset size: {len(self.df)} entries")
        print(f"🔄 Replaced {replaced_entries} invalid entries with standard values")
        
        # Calculate net carbohydrates (total carbs - fiber)
        self.df['Net_Carbs'] = self.df['Carbohydrates (g)'] - self.df['Fiber (g)']
        self.df['Net_Carbs'] = self.df['Net_Carbs'].clip(lower=0)  # Ensure non-negative
        
        # Calculate total calories from macronutrients (4-4-9 rule)
        self.df['Calculated_Calories'] = (
            self.df['Protein (g)'] * 4 + 
            self.df['Net_Carbs'] * 4 +  # Use net carbs instead of total carbs
            self.df['Fat (g)'] * 9
        )
        
        # Remove rows where actual calories are too different from calculated
        self.df['Calorie_Diff'] = abs(self.df['Calories (kcal)'] - self.df['Calculated_Calories'])
        self.df = self.df[self.df['Calorie_Diff'] < 50]  # Stricter tolerance of 50 kcal
        
        # Add macronutrient ratios as features
        self.df['Protein_to_Fat_Ratio'] = np.where(self.df['Fat (g)'] > 0, 
                                                  self.df['Protein (g)'] / self.df['Fat (g)'], 0)
        self.df['Carbs_to_Fat_Ratio'] = np.where(self.df['Fat (g)'] > 0, 
                                                self.df['Net_Carbs'] / self.df['Fat (g)'], 0)
        
        # Add these derived features to feature columns
        for feature in ['Net_Carbs', 'Protein_to_Fat_Ratio', 'Carbs_to_Fat_Ratio']:
            if feature not in self.feature_columns:
                self.feature_columns.append(feature)
        
        return self.df

    def build_preprocessor(self):
        """
        Build preprocessing pipeline with robust scaling.
        
        Returns:
            ColumnTransformer: Preprocessing pipeline
        """
        # Use RobustScaler which is less sensitive to outliers
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),  # Use median for robustness
            ('scaler', RobustScaler())  # More robust to outliers than StandardScaler
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.feature_columns)
            ])
        
        return preprocessor

    def train_model(self, X, y):
        """
        Train the model with enhanced parameters and cross-validation.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            
        Returns:
            RandomForestRegressor: Trained model
        """
        # Split data with stratification to ensure balanced distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        # Build and apply preprocessing
        self.preprocessor = self.build_preprocessor()
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Define parameter grid for GridSearchCV
        # This grid will test different combinations of Random Forest parameters
        # to find the optimal configuration for our calorie prediction task
        param_grid = {
            # Number of trees in the forest
            # More trees = better performance but slower training
            # We choose 200-300 as a balance between accuracy and speed
            'n_estimators': [200, 300],
            
            # Maximum depth of each tree
            # Controls how many decisions a tree can make
            # Deeper trees can capture more complex patterns but may overfit
            # We choose relatively shallow depths (4-8) to prevent overfitting
            'max_depth': [4, 6, 8],
            
            # Minimum number of samples required to split a node
            # Higher values prevent the tree from creating too many small groups
            # We choose 2-4 to allow reasonable splitting while preventing overfitting
            'min_samples_split': [2, 4],
            
            # Minimum number of samples required in a leaf node
            # Prevents the tree from creating leaves with very few samples
            # We choose 1-2 to allow some flexibility while maintaining stability
            'min_samples_leaf': [1, 2]
        }
        
        # Use GridSearchCV with k-fold cross-validation
        # This will try all combinations of parameters and select the best one
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            scoring='neg_mean_squared_error',  # We want to minimize prediction error
            cv=cv,  # Use 5-fold cross-validation
            n_jobs=-1,  # Use all available CPU cores
            verbose=1  # Show progress during training
        )
        
        print("\n🔍 Tuning model parameters...")
        grid_search.fit(X_train_processed, y_train)
        
        # Get best model
        model = grid_search.best_estimator_
        
        # Evaluate model performance
        train_pred = model.predict(X_train_processed)
        test_pred = model.predict(X_test_processed)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        print("\n📊 Model Evaluation:")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Train R²: {train_r2:.4f} = {train_r2*100:.2f}%")
        print(f"Test R²: {test_r2:.4f} = {test_r2*100:.2f}%")
        print(f"Train RMSE: {train_rmse:.2f} kcal")#  average model predictions are off on the train set by  
        print(f"Test RMSE: {test_rmse:.2f} kcal")# model predictions are off on the train set by
        print(f"Train MAE: {train_mae:.2f} kcal")
        print(f"Test MAE: {test_mae:.2f} kcal")
        
        return model

    def process_and_train(self, data_file):
        """Full data processing and training workflow"""
        try:
            print("\n🔧 Processing dataset...")
            self.df = self.clean_data(pd.read_csv(data_file))
            
            # Prepare features/target
            X = self.df[self.feature_columns]
            y = self.df[self.target_column]
            
            # Train and save
            print("\n🤖 Training model...")
            self.model = self.train_model(X, y)
            self.save_model()
            
        except Exception as e:
            print(f"❌ Processing failed: {str(e)}")
            raise

    def save_model(self):
        """Save model, preprocessor and selector"""
        joblib.dump({
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_columns': self.feature_columns
        }, self.model_path)
        print(f"💾 Model saved to {self.model_path}")

    def load_model(self):
        """Load saved model"""
        artifacts = joblib.load(self.model_path)
        self.model = artifacts['model']
        self.preprocessor = artifacts['preprocessor']
        self.feature_columns = artifacts['feature_columns']
        print(f"💽 Loaded model from {self.model_path}")

    def predict(self, food_data):
        """Predict calories for a food item based on nutritional values with derived features"""
        if not self.model or not self.preprocessor:
            print("⚠️ No model loaded for prediction")
            return None
            
        try:
            # Basic nutritional features
            input_data = {
                'Protein (g)': food_data.get('protein', 0),
                'Carbohydrates (g)': food_data.get('carbohydrates', 0),
                'Fat (g)': food_data.get('fat', 0),
            }
            
            # Calculate calories using the 4-4-9 rule as a baseline
            baseline_calories = (
                input_data['Protein (g)'] * 4 + 
                input_data['Carbohydrates (g)'] * 4 + 
                input_data['Fat (g)'] * 9
            )
            
            input_df = pd.DataFrame([input_data])
            
            # Ensure all feature columns are present
            for col in self.feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Preprocess the data
            processed_data = self.preprocessor.transform(input_df[self.feature_columns])
            
            # Get model prediction
            model_prediction = float(self.model.predict(processed_data)[0])
            
            # Ensure prediction is within reasonable bounds
            # Use baseline calories as a reference point
            min_calories = max(0, baseline_calories * 0.5)  # At least 50% of baseline
            max_calories = baseline_calories * 1.5  # At most 150% of baseline
            
            # Clip prediction to reasonable range
            predicted_calories = max(min_calories, min(model_prediction, max_calories))
            
            # For very small portions, ensure minimum calories
            if predicted_calories < 1 and (input_data['Protein (g)'] > 0 or 
                                         input_data['Carbohydrates (g)'] > 0 or 
                                         input_data['Fat (g)'] > 0):
                predicted_calories = 1
            
            return round(predicted_calories, 1)
            
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            return None

    def search(self, query, exact_match=False):
        """Search with proper calorie prediction"""
        if self.df is None or self.df.empty or not self.model:
            return None

        query = str(query).strip().lower()
        if not query:
            return None

        try:
            # Find matching food item(s)
            if exact_match:
                matches = self.df[self.df['Food_Item'].str.lower() == query]
            else:
                matches = self.df[
                    self.df['Food_Item'].str.lower().str.contains(query, na=False)
                ]

            if matches.empty:
                return None

            # Get first match and its nutritional data
            food_row = matches.iloc[0]
            
            # Extract basic nutritional data
            nutritional_data = {
                'protein': float(food_row.get('Protein (g)', 0)),
                'carbohydrates': float(food_row.get('Carbohydrates (g)', 0)),
                'fat': float(food_row.get('Fat (g)', 0)),
            }

            # Use the predict method which handles all preprocessing steps
            predicted_calories = self.predict(nutritional_data)

            return {
                'food_name': food_row['Food_Item'],
                'category': food_row.get('Category', ''),
                'calories': round(predicted_calories, 1),
                'nutritional_data': nutritional_data
            }

        except Exception as e:
            print(f"❌ Search error: {str(e)}")
            return None