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

class MotherboardProcessor:
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

        self.df = self.df.dropna(subset=['name', 'socket', 'form_factor'])
        self.df = self.df.drop_duplicates(subset=['name'])

        self.df['color'] = self.df['color'].fillna('Unknown').astype(str)
        self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce')

    def predict_missing_prices(self):
        df_complete = self.df[self.df['price'].notnull()]
        df_missing = self.df[self.df['price'].isnull()]

        if df_missing.empty or df_complete.empty:
            print("No missing prices or insufficient data for training.")
            self.df['price'] = self.df['price'].fillna(df_complete['price'].median())
            return

        features = ['socket', 'form_factor', 'color', 'max_memory', 'memory_slots']
        for col in features:
            self.df[col] = self.df[col].astype(str).fillna('Unknown')

        # Convert to numeric where appropriate
        self.df['max_memory'] = pd.to_numeric(self.df['max_memory'], errors='coerce').fillna(0)
        self.df['memory_slots'] = pd.to_numeric(self.df['memory_slots'], errors='coerce').fillna(0)

        X_train = df_complete[features]
        y_train = df_complete['price']

        preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['socket', 'form_factor', 'color']),
        ], remainder='passthrough')

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        model.fit(X_train, y_train)

        X_missing = df_missing[features]
        predicted_prices = model.predict(X_missing)
        self.df.loc[self.df['price'].isnull(), 'price'] = predicted_prices

    def standardize_data(self):
        numeric_cols = ['price', 'max_memory', 'memory_slots']
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
            self.df[col] = self.df[col].round(2)

        self.df['socket'] = self.df['socket'].str.upper()
        self.df['form_factor'] = self.df['form_factor'].str.lower()

        # Thêm cột memory_type
        self.df['memory_type'] = self.df.apply(self.assign_memory_type, axis=1)

    def assign_memory_type(self, row):
        socket = row['socket'].upper()
        max_memory = row['max_memory']
        form_factor = row['form_factor'].lower()

        # DDR5: AM5, LGA1700 (Intel 12th-14th Gen, prefer DDR5)
        if socket in ['AM5'] or (socket == 'LGA1700' and max_memory >= 128):
            return 'DDR5'
        # DDR4: AM4, LGA1200, LGA1151, STRX4, some LGA1700
        elif socket in ['AM4', 'LGA1200', 'LGA1151', 'STRX4'] or (socket == 'LGA1700' and max_memory <= 64):
            return 'DDR4'
        # DDR3: Older sockets like LGA1150, LGA1155, AM3, or low max_memory
        elif socket in ['LGA1150', 'LGA1155', 'LGA1156', 'AM3', 'AM3+', 'FM1', 'FM2', 'FM2+', 'LGA2011', 'LGA2011-3'] or max_memory <= 32:
            return 'DDR3'
        # DDR2: Very old sockets or very low max_memory
        elif socket in ['LGA775', 'LGA1366', 'AM2', 'AM2+/AM2'] or max_memory <= 8:
            return 'DDR2'
        # Integrated or rare sockets
        elif 'INTEGRATED' in socket:
            return 'DDR3' if max_memory <= 16 else 'DDR4'
        # Fallback: Default based on max_memory
        else:
            return 'DDR4' if max_memory >= 64 else 'DDR3'

    def classify_purpose(self):
        conditions = [
            (self.df['max_memory'] >= 128) & (self.df['memory_slots'] >= 4) & 
            (self.df['socket'].isin(['AM5', 'LGA1700', 'LGA1200'])),
            (self.df['max_memory'] >= 64) & (self.df['form_factor'].isin(['micro atx', 'mini itx'])),
            (self.df['max_memory'] >= 128) & (self.df['memory_slots'] >= 4) & 
            (self.df['form_factor'].isin(['atx', 'eatx']))
        ]
        choices = ['Gaming', 'Office', 'Design']
        self.df['purpose'] = np.select(conditions, choices, default='General')

    def calculate_performance_score(self):
        memory_bonus = self.df['max_memory'] / 128
        slots_bonus = self.df['memory_slots'] / 4

        socket_scores = {
            'AM5': 1.0, 'LGA1700': 1.0, 'LGA1200': 0.9, 'AM4': 0.8,
            'STRX4': 0.7, 'LGA1151': 0.6, 'LGA1155': 0.4, 'LGA1156': 0.3,
            'AM3': 0.2, 'LGA775': 0.1
        }
        socket_bonus = self.df['socket'].map(socket_scores).fillna(0.5)

        raw_score = (0.4 * memory_bonus + 0.3 * slots_bonus + 0.3 * socket_bonus)
        max_score = raw_score.max()
        normalized_score = (raw_score / max_score) * 100

        self.df['performance_score'] = normalized_score.round(2)

    def save_data(self):
        output_file = os.path.join(self.output_dir, 'motherboard_cleaned.csv')
        self.df.to_csv(output_file, index=False)
        print(f"Saved cleaned data to {output_file}")

    def visualize_data(self):
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        sns.histplot(self.df['price'], bins=30, kde=True)
        plt.title('Price Distribution (Motherboard)')

        plt.subplot(2, 2, 2)
        sns.scatterplot(x='price', y='performance_score', hue='purpose', data=self.df)
        plt.title('Price vs Performance (Motherboard)')

        plt.subplot(2, 2, 3)
        sns.boxplot(x='purpose', y='price', data=self.df)
        plt.title('Price by Purpose (Motherboard)')

        plt.subplot(2, 2, 4)
        self.df['purpose'].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title('Purpose Distribution (Motherboard)')

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'motherboard_visualization.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    data_manager = PCPartsDataManager(data_dir='dataset', output_dir='data')
    data_manager.load_data()
    data_manager.process_part('motherboard', MotherboardProcessor)
    print("Available parts:", data_manager.list_available_parts())