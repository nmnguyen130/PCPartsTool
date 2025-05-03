import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.preprocessing.data_manager import PCPartsDataManager

class CPUProcessor:
    def __init__(self, data, output_dir):
        self.data = data
        self.output_dir = output_dir
        self.df = None

    def process(self):
        self.clean_data()
        self.standardize_data()
        self.infer_socket()
        self.classify_purpose()
        # self.list_unique_graphics()
        self.calculate_performance_score()
        self.save_data()
        self.visualize_data()

    def clean_data(self):
        self.df = pd.DataFrame(self.data)
        self.df.columns = self.df.columns.str.lower()
        
        # Remove rows with missing critical fields
        self.df = self.df.dropna(subset=['price', 'name', 'core_count'])
        self.df = self.df.drop_duplicates(subset=['name'])
        
        # Handle graphics
        self.df['graphics'] = self.df['graphics'].fillna('None').astype(str)

    def standardize_data(self):
        numeric_cols = ['price', 'core_count', 'core_clock', 'boost_clock', 'tdp']
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
        
        self.df['smt'] = self.df['smt'].fillna(False).astype(bool)

    def infer_socket(self):
        def get_socket(cpu_name):
            cpu_name = cpu_name.lower()
            
            # AMD CPUs
            if 'ryzen' in cpu_name:
                # Ryzen 3000/5000 series (AM4) và Ryzen 7000 series (AM5)
                if any(series in cpu_name for series in ['3', '5', '7', '9']):
                    return 'AM4'
                elif '7000' in cpu_name:
                    return 'AM5'
            
            elif 'threadripper' in cpu_name:
                # Threadripper 3960X/3970X/3990X (sTRX4) và các dòng khác (sTR4)
                if '3990x' in cpu_name or '3970x' in cpu_name or '3960x' in cpu_name:
                    return 'sTRX4'
                return 'sTR4'
            
            elif 'fx' in cpu_name:
                # FX-8320, FX-8350, FX-9590, FX-9370 → AM3+, các FX khác → AM3
                if any(model in cpu_name for model in ['8320', '8350', '9590', '9370']):
                    return 'AM3+'
                return 'AM3'
            
            elif 'athlon' in cpu_name:
                # Athlon 3000G, 220GE → AM4
                # Athlon II X2 240 → AM2+/AM3
                # Athlon 64 3000+ → Socket 939 hoặc 754
                if '3000g' in cpu_name or '220ge' in cpu_name:
                    return 'AM4'
                elif 'ii x2 240' in cpu_name:
                    return 'AM2+/AM3'
                elif '64 3000+' in cpu_name:
                    return 'Socket 939' if '939' in cpu_name else 'Socket 754'
                return 'AM3'
            
            elif '5350' in cpu_name:
                # AMD 5350 (Kabini platform)
                return 'AM1'

            # AMD A-series CPUs: phân nhóm theo mã model
            elif 'a4-' in cpu_name or 'a6-' in cpu_name or 'a8-' in cpu_name or 'a10-' in cpu_name or 'a12-' in cpu_name:
                # A4-3xxx, A6-3xxx, A8-3xxx, A8-3850 → FM1 (Llano)
                if any(code in cpu_name for code in ['-3', '-3850']):
                    return 'FM1'
                # A4-4xxx, A6-5xxx, A8-5xxx, A10-5xxx, A4-6xxx, A6-6xxx → FM2 (Trinity, Richland)
                elif any(code in cpu_name for code in ['-4', '-5', '-6']):
                    return 'FM2'
                # A4-7xxx, A6-7xxx, A8-7xxx, A10-7xxx → FM2+ (Kaveri, Godavari)
                elif any(code in cpu_name for code in ['-7']):
                    return 'FM2+'
                # A6-9xxx, A10-9xxx, A12-9xxx → AM4 (Bristol Ridge)
                elif any(code in cpu_name for code in ['-9']):
                    return 'AM4'

            elif 'phenom ii' in cpu_name:
                # Phenom II X4 965 → AM3, các Phenom II khác → AM2+/AM3
                if 'x4 965' in cpu_name:
                    return 'AM3'
                return 'AM2+/AM3'
            
            elif 'opteron' in cpu_name:
                # Opteron 6344 → G34, các Opteron khác → C32
                if '6344' in cpu_name:
                    return 'G34'
                return 'C32'

            # Intel CPUs
            elif 'core' in cpu_name:
                # Intel Core Gen12/13/14 → LGA1700
                if any(gen in cpu_name for gen in ['12', '13', '14']):
                    return 'LGA1700'
                # Intel Core Gen10/11 → LGA1200
                elif any(gen in cpu_name for gen in ['10', '11']):
                    return 'LGA1200'
                # Intel Core Gen6/7/8/9 → LGA1151
                elif any(gen in cpu_name for gen in ['6', '7', '8', '9']):
                    return 'LGA1151'
                # Intel Core Gen2/3/4/5 → LGA1155
                elif any(gen in cpu_name for gen in ['2', '3', '4', '5']):
                    return 'LGA1155'
            
            elif 'xeon' in cpu_name:
                # Giả định chung cho Xeon (có thể chi tiết hóa thêm nếu muốn)
                return 'LGA2011'
            
            elif 'celeron' in cpu_name or 'pentium' in cpu_name:
                # Celeron/Pentium mới (g series, Gold) → LGA1200, cũ hơn → LGA1151
                if 'g' in cpu_name or 'gold' in cpu_name:
                    return 'LGA1200'
                return 'LGA1151'

            # Không nhận diện được CPU
            return 'Unknown'

        # Áp dụng hàm get_socket cho từng dòng trong DataFrame
        self.df['socket'] = self.df['name'].apply(get_socket)

    def classify_purpose(self):
        conditions = [
            (self.df['core_count'] >= 6) & (self.df['boost_clock'] >= 4.5),  # Gaming
            (self.df['core_count'] >= 4) & (self.df['boost_clock'] >= 3.5) & (self.df['tdp'] <= 65),  # Office
            (self.df['core_count'] >= 8) & (self.df['boost_clock'] >= 4.0)  # Design
        ]
        choices = ['Gaming', 'Office', 'Design']
        self.df['purpose'] = np.select(conditions, choices, default='General')

    def calculate_performance_score(self):
        smt_bonus = self.df['smt'].astype(float) * 0.1  # 10% bonus nếu có SMT
        graphics_bonus = self.df['graphics'].apply(lambda gpu: 5 if 'iris' in gpu or 'pro' in gpu else 2 if 'uhd' in gpu else 0)

        raw_score = (
            (0.4 * self.df['core_count'] + 0.4 * (self.df['core_clock'] * 0.6 + self.df['boost_clock'] * 0.4)) *
            (1 + smt_bonus) / (self.df['tdp'] + 1)
        )
        max_score = raw_score.max()
        normalized_score = (raw_score * 100) / max_score

        self.df['performance_score'] = normalized_score + graphics_bonus

    def save_data(self):
        output_file = os.path.join(self.output_dir, 'cpu_cleaned.csv')
        self.df.to_csv(output_file, index=False)
        print(f"Saved cleaned data to {output_file}")

    def list_unique_graphics(self):
        unique_graphics = self.df['graphics'].unique()
        print("Unique graphics GPU found:")
        for gpu in unique_graphics:
            print(gpu)

    def visualize_data(self):
        plt.figure(figsize=(12, 8))

        # Histogram giá
        plt.subplot(2, 2, 1)
        sns.histplot(self.df['price'], bins=30, kde=True)
        plt.title('Price Distribution (CPU)')

        # Scatter giá vs hiệu suất
        plt.subplot(2, 2, 2)
        sns.scatterplot(x='price', y='performance_score', hue='purpose', data=self.df)
        plt.title('Price vs Performance (CPU)')

        # Boxplot giá theo mục đích
        plt.subplot(2, 2, 3)
        sns.boxplot(x='purpose', y='price', data=self.df)
        plt.title('Price by Purpose (CPU)')

        # Pie chart mục đích
        plt.subplot(2, 2, 4)
        self.df['purpose'].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title('Purpose Distribution (CPU)')

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'cpu_visualization.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    data_manager = PCPartsDataManager(data_dir='dataset', output_dir='data')
    data_manager.load_data()
    data_manager.process_part('cpu', CPUProcessor)
    print("Available parts:", data_manager.list_available_parts())