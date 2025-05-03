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

class KeyboardProcessor:
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

        self.df = self.df.dropna(subset=['name', 'style'])
        self.df = self.df.drop_duplicates(subset=['name'])

        # Convert numeric
        self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce')

    def predict_missing_prices(self):
        df_complete = self.df[self.df['price'].notnull()]
        df_missing = self.df[self.df['price'].isnull()]

        if df_missing.empty or df_complete.empty:
            print("No missing prices or insufficient data for training.")
            self.df['price'] = self.df['price'].fillna(df_complete['price'].median())
            return

        features = ['style', 'switches', 'backlit', 'tenkeyless', 'connection_type', 'color']
        for col in features:
            self.df[col] = self.df[col].astype(str).fillna('Unknown')

        X_train = df_complete[features]
        y_train = df_complete['price']

        preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), features)
        ])

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        model.fit(X_train, y_train)

        X_missing = df_missing[features]
        predicted_prices = model.predict(X_missing)
        self.df.loc[self.df['price'].isnull(), 'price'] = predicted_prices

    def standardize_data(self):
        self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce').fillna(0).round(2)

        categorical_cols = ['style', 'switches', 'backlit', 'tenkeyless', 'connection_type', 'color']
        for col in categorical_cols:
            self.df[col] = self.df[col].astype(str).fillna('Unknown')

    def classify_purpose(self):
        conditions = [
            (self.df['style'].str.contains('Gaming', case=False) & (self.df['price'] > 150)),
            (self.df['style'].str.contains('Standard|Slim|Ergonomic', case=False) & (self.df['price'] <= 50)),
            (self.df['style'].str.contains('Mini', case=False) | (self.df['tenkeyless'].str.lower() == 'true'))
        ]
        choices = ['Premium Gaming', 'Office/Budget', 'Compact']
        self.df['purpose'] = np.select(conditions, choices, default='Mid-Range Gaming')

    def calculate_performance_score(self):
        switch_score = self.df['switches'].apply(lambda x: 1 if 'Cherry MX' in x or 'Razer' in x else 0.8 if 'Gateron' in x or 'Kailh' in x else 0.5 if x != 'Unknown' else 0)
        backlit_score = self.df['backlit'].apply(lambda x: 1 if 'RGB' in x else 0.7 if x != 'Unknown' and x.lower() != 'none' else 0)
        connection_score = self.df['connection_type'].apply(lambda x: 1 if 'Wireless' in x else 0.8 if 'Both' in x else 0.6 if 'Wired' in x else 0)

        raw_score = (switch_score * 0.4 + backlit_score * 0.3 + connection_score * 0.3) * 100
        max_score = raw_score.max()
        self.df['performance_score'] = ((raw_score / max_score) * 100).round(2)

    def save_data(self):
        output_file = os.path.join(self.output_dir, 'keyboard_cleaned.csv')
        self.df.to_csv(output_file, index=False)
        print(f"Saved cleaned data to {output_file}")

    def visualize_data(self):
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        sns.histplot(self.df['price'], bins=30, kde=True)
        plt.title('Price Distribution (Keyboard)')

        plt.subplot(2, 2, 2)
        sns.scatterplot(x='price', y='performance_score', hue='purpose', data=self.df)
        plt.title('Price vs Performance (Keyboard)')

        plt.subplot(2, 2, 3)
        sns.boxplot(x='purpose', y='price', data=self.df)
        plt.title('Price by Purpose (Keyboard)')

        plt.subplot(2, 2, 4)
        self.df['purpose'].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title('Purpose Distribution (Keyboard)')

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'keyboard_visualization.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    data_manager = PCPartsDataManager(data_dir='dataset', output_dir='data')
    data_manager.load_data()
    data_manager.process_part('keyboard', KeyboardProcessor)
    print("Available parts:", data_manager.list_available_parts())
