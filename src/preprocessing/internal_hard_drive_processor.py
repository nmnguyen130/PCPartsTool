import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.preprocessing.data_manager import PCPartsDataManager

class InternalHardDriveProcessor:
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
        self.df = self.df.dropna(subset=['price', 'name', 'capacity'])
        self.df = self.df.drop_duplicates(subset=['name'])
        
        # Handle cache
        self.df['cache'] = self.df['cache'].fillna(0).astype(float)

    def standardize_data(self):
        numeric_cols = ['price', 'capacity', 'price_per_gb', 'cache']
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
        
        # Convert type to categorical
        self.df['type'] = self.df['type'].astype(str)
        self.df['interface'] = self.df['interface'].astype(str)
        self.df['form_factor'] = self.df['form_factor'].astype(str)

    def classify_purpose(self):
        conditions = [
            (self.df['type'] == 'SSD') & (self.df['interface'].str.contains('PCIe 4.0|PCIe 5.0')),  # Gaming
            (self.df['type'].str.contains('5400|7200')) & (self.df['capacity'] >= 4000),  # Storage
            (self.df['type'] == 'SSD') & (self.df['interface'].str.contains('SATA')) & (self.df['capacity'] <= 2000)  # Office
        ]
        choices = ['Gaming', 'Storage', 'Office']
        self.df['purpose'] = np.select(conditions, choices, default='General')

    def calculate_performance_score(self):
        # Interface speed multiplier
        interface_speed = self.df['interface'].apply(
            lambda x: 5 if 'PCIe 5.0' in x else 4 if 'PCIe 4.0' in x else 3 if 'PCIe 3.0' in x else 2 if 'SATA' in x else 1
        )
        
        # Type bonus (SSD vs HDD)
        type_bonus = self.df['type'].apply(lambda x: 2 if x == 'SSD' else 1)
        
        # Cache bonus
        cache_bonus = self.df['cache'].apply(lambda x: x / 1000 if x > 0 else 0)
        
        raw_score = (
            (self.df['capacity'] / 1000) * interface_speed * type_bonus + cache_bonus
        ) / (self.df['price_per_gb'] + 0.001)  # Avoid division by zero
        
        max_score = raw_score.max()
        normalized_score = (raw_score * 100) / max_score
        
        self.df['performance_score'] = normalized_score

    def save_data(self):
        output_file = os.path.join(self.output_dir, 'internal_hard_drive_cleaned.csv')
        self.df.to_csv(output_file, index=False)
        print(f"Saved cleaned data to {output_file}")

    def visualize_data(self):
        plt.figure(figsize=(12, 8))

        # Histogram giá
        plt.subplot(2, 2, 1)
        sns.histplot(self.df['price'], bins=30, kde=True)
        plt.title('Price Distribution (Internal Hard Drive)')

        # Scatter giá vs hiệu suất
        plt.subplot(2, 2, 2)
        sns.scatterplot(x='price', y='performance_score', hue='purpose', data=self.df)
        plt.title('Price vs Performance (Internal Hard Drive)')

        # Boxplot giá theo mục đích
        plt.subplot(2, 2, 3)
        sns.boxplot(x='purpose', y='price', data=self.df)
        plt.title('Price by Purpose (Internal Hard Drive)')

        # Pie chart mục đích
        plt.subplot(2, 2, 4)
        self.df['purpose'].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title('Purpose Distribution (Internal Hard Drive)')

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'internal_hard_drive_visualization.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    data_manager = PCPartsDataManager(data_dir='dataset', output_dir='data')
    data_manager.load_data()
    data_manager.process_part('internal-hard-drive', InternalHardDriveProcessor)
    print("Available parts:", data_manager.list_available_parts())