import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.preprocessing.data_manager import PCPartsDataManager

class VideoCardProcessor:
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
        self.df = self.df.dropna(subset=['price', 'name', 'chipset', 'memory'])
        self.df = self.df.drop_duplicates(subset=['name'])
        
        # Handle missing boost_clock
        self.df['boost_clock'] = self.df['boost_clock'].fillna(self.df['core_clock']).astype(float)

    def standardize_data(self):
        numeric_cols = ['price', 'memory', 'core_clock', 'boost_clock', 'length']
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
        
        # Convert chipset and color to categorical
        self.df['chipset'] = self.df['chipset'].astype(str)
        self.df['color'] = self.df['color'].astype(str)

    def classify_purpose(self):
        conditions = [
            (self.df['chipset'].str.contains('RTX 4090|RTX 4080|RX 7900 XTX')),  # High-End Gaming
            (self.df['memory'] >= 16) & (self.df['boost_clock'] >= 2000),  # Workstation
            (self.df['price'] <= 300) & (self.df['memory'] <= 8)  # Budget Gaming
        ]
        choices = ['High-End Gaming', 'Workstation', 'Budget Gaming']
        self.df['purpose'] = np.select(conditions, choices, default='Mid-Range Gaming')

    def calculate_performance_score(self):
        clock_score = self.df['boost_clock'] / 2500  # Chuẩn hóa tốc độ xung nhịp (max ~2500 MHz)
        memory_score = self.df['memory'] / 24  # Chuẩn hóa dung lượng bộ nhớ (max ~24GB)
        cuda_score = self.df.get('cuda_cores', 0) / 16000  # Chuẩn hóa số nhân CUDA (max ~16000, RTX 4090)

        # Trọng số: 50% clock, 30% memory, 20% cuda_cores
        raw_score = (clock_score * 0.5 + memory_score * 0.3 + cuda_score * 0.2) * 100
        
        # Chuẩn hóa điểm về thang 100
        max_score = raw_score.max()
        normalized_score = (raw_score / max_score) * 100
        
        self.df['performance_score'] = normalized_score

    def save_data(self):
        output_file = os.path.join(self.output_dir, 'video_card_cleaned.csv')
        self.df.to_csv(output_file, index=False)
        print(f"Saved cleaned data to {output_file}")

    def visualize_data(self):
        plt.figure(figsize=(12, 8))

        # Histogram giá
        plt.subplot(2, 2, 1)
        sns.histplot(self.df['price'], bins=30, kde=True)
        plt.title('Price Distribution (Video Card)')

        # Scatter giá vs hiệu suất
        plt.subplot(2, 2, 2)
        sns.scatterplot(x='price', y='performance_score', hue='purpose', data=self.df)
        plt.title('Price vs Performance (Video Card)')

        # Boxplot giá theo mục đích
        plt.subplot(2, 2, 3)
        sns.boxplot(x='purpose', y='price', data=self.df)
        plt.title('Price by Purpose (Video Card)')

        # Pie chart mục đích
        plt.subplot(2, 2, 4)
        self.df['purpose'].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title('Purpose Distribution (Video Card)')

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'video_card_visualization.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    data_manager = PCPartsDataManager(data_dir='dataset', output_dir='data')
    data_manager.load_data()
    data_manager.process_part('video-card', VideoCardProcessor)
    print("Available parts:", data_manager.list_available_parts())