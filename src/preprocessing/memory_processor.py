import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.preprocessing.data_manager import PCPartsDataManager

class MemoryProcessor:
    def __init__(self, data, output_dir):
        self.data = data
        self.output_dir = output_dir
        self.df = None

    def process(self):
        self.clean_data()
        self.standardize_data()
        self.classify_purpose()
        self.calculate_performance_score()
        self.save_data()
        self.visualize_data()

    def clean_data(self):
        self.df = pd.DataFrame(self.data)
        self.df.columns = self.df.columns.str.lower()
        
        # Remove rows with missing critical fields
        self.df = self.df.dropna(subset=['price', 'name', 'speed', 'modules'])
        self.df = self.df.drop_duplicates(subset=['name'])
        
        # Handle color
        self.df['color'] = self.df['color'].fillna('Unknown').astype(str)

    def standardize_data(self):
        # Extract speed and ddr_type from speed list
        self.df['speed_mhz'] = self.df['speed'].apply(lambda x: x[1])
        self.df['ddr_type'] = self.df['speed'].apply(lambda x: f'DDR{x[0]}')
        
        # Calculate total capacity from modules
        self.df['total_capacity'] = self.df['modules'].apply(lambda x: x[0] * x[1])
        
        # Convert numeric columns
        numeric_cols = ['price', 'speed_mhz', 'total_capacity', 'price_per_gb', 'first_word_latency', 'cas_latency']
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(-1)
        
        # Drop original speed and modules columns
        self.df = self.df.drop(columns=['speed', 'modules'])

    def classify_purpose(self):
        conditions = [
            (self.df['speed_mhz'] >= 6000) & (self.df['first_word_latency'] <= 10),  # Gaming
            (self.df['speed_mhz'] >= 3200) & (self.df['total_capacity'] >= 16) & (self.df['first_word_latency'] <= 12),  # Office
            (self.df['total_capacity'] >= 32) & (self.df['speed_mhz'] >= 5600)  # Design
        ]
        choices = ['Gaming', 'Office', 'Design']
        self.df['purpose'] = np.select(conditions, choices, default='General')

    def calculate_performance_score(self):
        # Performance score based on speed, latency, and capacity
        latency_factor = 1 / (self.df['first_word_latency'] + self.df['cas_latency'] + 1)  # Inverse latency for higher score
        capacity_bonus = self.df['total_capacity'] / 32  # Normalize by 32GB
        speed_bonus = self.df['speed_mhz'] / 6000  # Normalize by 6000MHz

        raw_score = (0.4 * speed_bonus + 0.3 * latency_factor + 0.3 * capacity_bonus)
        max_score = raw_score.max()
        normalized_score = (raw_score * 100) / max_score

        self.df['performance_score'] = normalized_score

    def save_data(self):
        output_file = os.path.join(self.output_dir, 'memory_cleaned.csv')
        self.df.to_csv(output_file, index=False)
        print(f"Saved cleaned data to {output_file}")

    def visualize_data(self):
        plt.figure(figsize=(12, 8))

        # Histogram price
        plt.subplot(2, 2, 1)
        sns.histplot(self.df['price'], bins=30, kde=True)
        plt.title('Price Distribution (Memory)')

        # Scatter price vs performance
        plt.subplot(2, 2, 2)
        sns.scatterplot(x='price', y='performance_score', hue='purpose', data=self.df)
        plt.title('Price vs Performance (Memory)')

        # Boxplot price by purpose
        plt.subplot(2, 2, 3)
        sns.boxplot(x='purpose', y='price', data=self.df)
        plt.title('Price by Purpose (Memory)')

        # Pie chart purpose distribution
        plt.subplot(2, 2, 4)
        self.df['purpose'].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title('Purpose Distribution (Memory)')

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'memory_visualization.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    data_manager = PCPartsDataManager(data_dir='dataset', output_dir='data')
    data_manager.load_data()
    data_manager.process_part('memory', MemoryProcessor)
    print("Available parts:", data_manager.list_available_parts())