import os
import json

class PCPartsDataManager:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.parts = {}
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                part_type = filename.replace('.json', '')
                filepath = os.path.join(self.data_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        self.parts[part_type] = data
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON file: {filepath}")

    def process_part(self, part_type, processor_class):
        if part_type not in self.parts:
            print(f"No data for {part_type}")
            return
        processor = processor_class(self.parts[part_type], self.output_dir)
        processor.process()

    def get_parts(self, part_type):
        return self.parts.get(part_type, [])

    def list_available_parts(self):
        return list(self.parts.keys())