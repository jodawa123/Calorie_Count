import pandas as pd
import numpy as np
from pathlib import Path

# Standard nutritional values for common foods (per 100g)
STANDARD_VALUES = {
    'Fruits': {
        'Apple': {'Calories (kcal)': 52, 'Protein (g)': 0.3, 'Carbohydrates (g)': 14, 'Fat (g)': 0.2, 'Fiber (g)': 2.4},
        'Banana': {'Calories (kcal)': 89, 'Protein (g)': 1.1, 'Carbohydrates (g)': 23, 'Fat (g)': 0.3, 'Fiber (g)': 2.6},
        'Orange': {'Calories (kcal)': 47, 'Protein (g)': 0.9, 'Carbohydrates (g)': 12, 'Fat (g)': 0.1, 'Fiber (g)': 2.4},
        'Grapes': {'Calories (kcal)': 69, 'Protein (g)': 0.7, 'Carbohydrates (g)': 18, 'Fat (g)': 0.2, 'Fiber (g)': 0.9},
        'Strawberry': {'Calories (kcal)': 32, 'Protein (g)': 0.7, 'Carbohydrates (g)': 8, 'Fat (g)': 0.3, 'Fiber (g)': 2.0}
    },
    'Vegetables': {
        'Carrot': {'Calories (kcal)': 41, 'Protein (g)': 0.9, 'Carbohydrates (g)': 10, 'Fat (g)': 0.2, 'Fiber (g)': 2.8},
        'Broccoli': {'Calories (kcal)': 34, 'Protein (g)': 2.8, 'Carbohydrates (g)': 7, 'Fat (g)': 0.4, 'Fiber (g)': 2.6},
        'Spinach': {'Calories (kcal)': 23, 'Protein (g)': 2.9, 'Carbohydrates (g)': 3.6, 'Fat (g)': 0.4, 'Fiber (g)': 2.2},
        'Potato': {'Calories (kcal)': 77, 'Protein (g)': 2, 'Carbohydrates (g)': 17, 'Fat (g)': 0.1, 'Fiber (g)': 2.2},
        'Tomato': {'Calories (kcal)': 18, 'Protein (g)': 0.9, 'Carbohydrates (g)': 3.9, 'Fat (g)': 0.2, 'Fiber (g)': 1.2}
    },
    'Meat': {
        'Chicken Breast': {'Calories (kcal)': 165, 'Protein (g)': 31, 'Carbohydrates (g)': 0, 'Fat (g)': 3.6, 'Fiber (g)': 0},
        'Beef Steak': {'Calories (kcal)': 271, 'Protein (g)': 26, 'Carbohydrates (g)': 0, 'Fat (g)': 19, 'Fiber (g)': 0},
        'Pork Chop': {'Calories (kcal)': 242, 'Protein (g)': 27, 'Carbohydrates (g)': 0, 'Fat (g)': 14, 'Fiber (g)': 0},
        'Salmon': {'Calories (kcal)': 208, 'Protein (g)': 22, 'Carbohydrates (g)': 0, 'Fat (g)': 13, 'Fiber (g)': 0},
        'Eggs': {'Calories (kcal)': 143, 'Protein (g)': 13, 'Carbohydrates (g)': 0.7, 'Fat (g)': 10, 'Fiber (g)': 0}
    },
    'Dairy': {
        'Milk': {'Calories (kcal)': 42, 'Protein (g)': 3.4, 'Carbohydrates (g)': 4.8, 'Fat (g)': 1, 'Fiber (g)': 0},
        'Cheese': {'Calories (kcal)': 402, 'Protein (g)': 25, 'Carbohydrates (g)': 0.4, 'Fat (g)': 33, 'Fiber (g)': 0},
        'Yogurt': {'Calories (kcal)': 59, 'Protein (g)': 3.5, 'Carbohydrates (g)': 4.7, 'Fat (g)': 3.3, 'Fiber (g)': 0},
        'Butter': {'Calories (kcal)': 717, 'Protein (g)': 0.1, 'Carbohydrates (g)': 0.1, 'Fat (g)': 81, 'Fiber (g)': 0},
        'Paneer': {'Calories (kcal)': 265, 'Protein (g)': 18, 'Carbohydrates (g)': 1.2, 'Fat (g)': 20, 'Fiber (g)': 0}
    },
    'Grains': {
        'Rice': {'Calories (kcal)': 130, 'Protein (g)': 2.7, 'Carbohydrates (g)': 28, 'Fat (g)': 0.3, 'Fiber (g)': 0.4},
        'Oats': {'Calories (kcal)': 389, 'Protein (g)': 17, 'Carbohydrates (g)': 66, 'Fat (g)': 7, 'Fiber (g)': 10},
        'Pasta': {'Calories (kcal)': 131, 'Protein (g)': 5, 'Carbohydrates (g)': 25, 'Fat (g)': 0.5, 'Fiber (g)': 1.2},
        'Bread': {'Calories (kcal)': 265, 'Protein (g)': 9, 'Carbohydrates (g)': 49, 'Fat (g)': 3.2, 'Fiber (g)': 2.7},
        'Quinoa': {'Calories (kcal)': 120, 'Protein (g)': 4.4, 'Carbohydrates (g)': 21, 'Fat (g)': 1.9, 'Fiber (g)': 2.8}
    },
    'Beverages': {
        'Water': {'Calories (kcal)': 0, 'Protein (g)': 0, 'Carbohydrates (g)': 0, 'Fat (g)': 0, 'Fiber (g)': 0},
        'Orange Juice': {'Calories (kcal)': 45, 'Protein (g)': 0.7, 'Carbohydrates (g)': 10, 'Fat (g)': 0.2, 'Fiber (g)': 0.2},
        'Coffee': {'Calories (kcal)': 0, 'Protein (g)': 0.1, 'Carbohydrates (g)': 0, 'Fat (g)': 0, 'Fiber (g)': 0},
        'Green Tea': {'Calories (kcal)': 0, 'Protein (g)': 0, 'Carbohydrates (g)': 0, 'Fat (g)': 0, 'Fiber (g)': 0},
        'Milkshake': {'Calories (kcal)': 112, 'Protein (g)': 3.4, 'Carbohydrates (g)': 18, 'Fat (g)': 3, 'Fiber (g)': 0}
    },
    'Snacks': {
        'Cookies': {'Calories (kcal)': 502, 'Protein (g)': 6, 'Carbohydrates (g)': 65, 'Fat (g)': 24, 'Fiber (g)': 2.3},
        'Chips': {'Calories (kcal)': 536, 'Protein (g)': 7, 'Carbohydrates (g)': 53, 'Fat (g)': 35, 'Fiber (g)': 4.4},
        'Chocolate': {'Calories (kcal)': 546, 'Protein (g)': 5.5, 'Carbohydrates (g)': 61, 'Fat (g)': 31, 'Fiber (g)': 3.4},
        'Nuts': {'Calories (kcal)': 607, 'Protein (g)': 20, 'Carbohydrates (g)': 21, 'Fat (g)': 54, 'Fiber (g)': 7},
        'Popcorn': {'Calories (kcal)': 375, 'Protein (g)': 12, 'Carbohydrates (g)': 74, 'Fat (g)': 4.3, 'Fiber (g)': 14}
    }
}

# Category-specific validation rules
CATEGORY_RULES = {
    'Fruits': {
        'Calories (kcal)': (20, 150),
        'Protein (g)': (0, 2),
        'Carbohydrates (g)': (5, 30),
        'Fat (g)': (0, 1),
        'Fiber (g)': (0, 5)
    },
    'Vegetables': {
        'Calories (kcal)': (10, 100),
        'Protein (g)': (0, 5),
        'Carbohydrates (g)': (2, 20),
        'Fat (g)': (0, 1),
        'Fiber (g)': (0, 5)
    },
    'Meat': {
        'Calories (kcal)': (100, 300),
        'Protein (g)': (15, 35),
        'Carbohydrates (g)': (0, 2),
        'Fat (g)': (2, 25),
        'Fiber (g)': (0, 0.5)
    },
    'Dairy': {
        'Calories (kcal)': (30, 750),
        'Protein (g)': (0, 30),
        'Carbohydrates (g)': (0, 5),
        'Fat (g)': (0, 85),
        'Fiber (g)': (0, 0.5)
    },
    'Grains': {
        'Calories (kcal)': (100, 400),
        'Protein (g)': (2, 20),
        'Carbohydrates (g)': (15, 80),
        'Fat (g)': (0, 10),
        'Fiber (g)': (0, 15)
    },
    'Beverages': {
        'Calories (kcal)': (0, 150),
        'Protein (g)': (0, 5),
        'Carbohydrates (g)': (0, 40),
        'Fat (g)': (0, 5),
        'Fiber (g)': (0, 2)
    },
    'Snacks': {
        'Calories (kcal)': (200, 700),
        'Protein (g)': (2, 25),
        'Carbohydrates (g)': (10, 80),
        'Fat (g)': (5, 60),
        'Fiber (g)': (0, 15)
    }
}

def clean_dataset(input_file='data/daily_food_nutrition_dataset.csv', output_file='data/cleaned_nutrition_dataset.csv'):
    """Clean and validate the nutrition dataset."""
    print("🔍 Loading dataset...")
    df = pd.read_csv(input_file)
    original_count = len(df)
    print(f"📊 Original dataset size: {original_count} entries")
    
    # Convert columns to numeric
    numeric_cols = ['Calories (kcal)', 'Protein (g)', 'Carbohydrates (g)', 'Fat (g)', 'Fiber (g)']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with missing values
    df = df.dropna(subset=numeric_cols)
    print(f"🧹 Removed {original_count - len(df)} entries with missing values")
    
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
    cleaned_df = pd.DataFrame(valid_entries)
    print(f"✅ Cleaned dataset size: {len(cleaned_df)} entries")
    print(f"🔄 Replaced {replaced_entries} invalid entries with standard values")
    
    # Save cleaned dataset
    cleaned_df.to_csv(output_file, index=False)
    print(f"💾 Saved cleaned dataset to {output_file}")
    
    # Print summary statistics
    print("\n📈 Summary Statistics:")
    for category in CATEGORY_RULES.keys():
        category_df = cleaned_df[cleaned_df['Category'] == category]
        if len(category_df) > 0:
            print(f"\n{category}:")
            print(f"  Entries: {len(category_df)}")
            print(f"  Avg Calories: {category_df['Calories (kcal)'].mean():.1f}")
            print(f"  Avg Protein: {category_df['Protein (g)'].mean():.1f}g")
            print(f"  Avg Carbs: {category_df['Carbohydrates (g)'].mean():.1f}g")
            print(f"  Avg Fat: {category_df['Fat (g)'].mean():.1f}g")

if __name__ == '__main__':
    clean_dataset() 