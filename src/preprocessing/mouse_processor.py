import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.preprocessing.data_manager import PCPartsDataManager

class MouseProcessor:
    def __init__(self, data, output_dir):
        self.data = data
        self.output_dir = output_dir
        self.df = None

    def process(self):
        self.clean_data()
        self.predict_missing_prices()
        self.standardize_data()
        self.classify_purpose()
        self.calculate_performance_score()
        self.save_data()
        self.visualize_data()

    def clean_data(self):
        self.df = pd.DataFrame(self.data)
        self.df.columns = self.df.columns.str.lower()

        # Remove rows with missing critical fields
        self.df = self.df.dropna(subset=['name', 'tracking_method'])
        self.df = self.df.drop_duplicates(subset=['name'])

        # Convert price and max_dpi to numeric (in case of 'null' or empty strings)
        self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce')
        self.df['max_dpi'] = pd.to_numeric(self.df['max_dpi'], errors='coerce')

        # Handle missing max_dpi with median
        self.df['max_dpi'] = self.df['max_dpi'].fillna(self.df['max_dpi'].median())

    def predict_missing_prices(self):
        df_complete = self.df[self.df['price'].notnull()]
        df_missing = self.df[self.df['price'].isnull()]

        if df_missing.empty or df_complete.empty:
            print("No missing prices or insufficient data for training.")
            self.df['price'] = self.df['price'].fillna(df_complete['price'].median())
            return

        features = ['max_dpi', 'tracking_method', 'connection_type', 'hand_orientation', 'color']
        for col in features:
            self.df[col] = self.df[col].astype(str).fillna('Unknown')

        # Prepare training data
        X_train = df_complete[features]
        y_train = df_complete['price']

        # Model pipeline
        preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['tracking_method', 'connection_type', 'hand_orientation', 'color'])
        ], remainder='passthrough')

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        model.fit(X_train, y_train)

        # Predict and fill missing prices
        X_missing = df_missing[features]
        predicted_prices = model.predict(X_missing)
        self.df.loc[self.df['price'].isnull(), 'price'] = predicted_prices

    def standardize_data(self):
        numeric_cols = ['price', 'max_dpi']
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
            self.df[col] = self.df[col].round(2)

        categorical_cols = ['tracking_method', 'connection_type', 'hand_orientation', 'color']
        for col in categorical_cols:
            self.df[col] = self.df[col].astype(str).fillna('Unknown')

    def classify_purpose(self):
        conditions = [
            (self.df['max_dpi'] >= 20000) & (self.df['price'] > 100),  # Premium Gaming
            (self.df['price'] <= 30) & (self.df['max_dpi'] <= 4000),  # Office/Budget
            (self.df['connection_type'].str.contains('Wireless')) & (self.df['max_dpi'] <= 12000)  # Portable
        ]
        choices = ['Premium Gaming', 'Office/Budget', 'Portable']
        self.df['purpose'] = np.select(conditions, choices, default='Mid-Range Gaming')

    def calculate_performance_score(self):
        dpi_score = (self.df['max_dpi'] / self.df['max_dpi'].max()).clip(0, 1)
        connection_score = self.df['connection_type'].apply(lambda x: 1 if 'Wireless' in x else 0.8 if 'Both' in x else 0.6 if 'Wired' in x else 0)

        raw_score = (dpi_score * 0.6 + connection_score * 0.4) * 100
        max_score = raw_score.max()
        normalized_score = (raw_score / max_score) * 100

        self.df['performance_score'] = normalized_score.round(2)

    def save_data(self):
        output_file = os.path.join(self.output_dir, 'mouse_cleaned.csv')
        self.df.to_csv(output_file, index=False)
        print(f"Saved cleaned data to {output_file}")

    def visualize_data(self):
        plt.figure(figsize=(12, 8))

        # Histogram giá
        plt.subplot(2, 2, 1)
        sns.histplot(self.df['price'], bins=30, kde=True)
        plt.title('Price Distribution (Mouse)')

        # Scatter giá vs hiệu suất
        plt.subplot(2, 2, 2)
        sns.scatterplot(x='price', y='performance_score', hue='purpose', data=self.df)
        plt.title('Price vs Performance (Mouse)')

        # Boxplot giá theo mục đích
        plt.subplot(2, 2, 3)
        sns.boxplot(x='purpose', y='price', data=self.df)
        plt.title('Price by Purpose (Mouse)')

        # Pie chart mục đích
        plt.subplot(2, 2, 4)
        self.df['purpose'].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title('Purpose Distribution (Mouse)')

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'mouse_visualization.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    data_manager = PCPartsDataManager(data_dir='dataset', output_dir='data')
    data_manager.load_data()
    data_manager.process_part('mouse', MouseProcessor)
    print("Available parts:", data_manager.list_available_parts())