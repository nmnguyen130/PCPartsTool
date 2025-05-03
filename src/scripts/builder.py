import itertools
import pandas as pd
import os
from dataclasses import dataclass

# Base Component class
@dataclass
class Component:
    name: str
    price: float
    performance_score: float
    purpose: str

# Specific component classes
@dataclass
class CPU(Component):
    socket: str

@dataclass
class Motherboard(Component):
    socket: str
    memory_type: str

@dataclass
class RAM(Component):
    ddr_type: str

@dataclass
class Storage(Component):
    type: str  # SSD or HDD

@dataclass
class GPU(Component):
    memory: float

@dataclass
class Keyboard(Component):
    pass

@dataclass
class Mouse(Component):
    pass

# PCBuilder class
class PCBuilder:
    def __init__(self, cpu_file: str, mb_file: str, ram_file: str, storage_file: str, gpu_file: str,
                 keyboard_file: str, mouse_file: str):
        self.cpus = self.load_components(cpu_file, self.parse_cpu)
        self.motherboards = self.load_components(mb_file, self.parse_motherboard)
        self.rams = self.load_components(ram_file, self.parse_ram)
        self.storages = self.load_components(storage_file, self.parse_storage)
        self.gpus = self.load_components(gpu_file, self.parse_gpu)
        self.keyboards = self.load_components(keyboard_file, self.parse_keyboard)
        self.mice = self.load_components(mouse_file, self.parse_mouse)

        self.purpose_mapping = {
            "Budget Gaming": {
                "cpu": ["Gaming"],
                "motherboard": ["Gaming"],
                "ram": ["Gaming"],
                "storage": ["Gaming"],
                "gpu": ["Budget Gaming"],
                "keyboard": ["Office/Budget"],
                "mouse": ["Office/Budget"]
            },
            "Mid-Range Gaming": {
                "cpu": ["Gaming"],
                "motherboard": ["Gaming"],
                "ram": ["Gaming"],
                "storage": ["Gaming"],
                "gpu": ["Budget Gaming"],
                "keyboard": ["Office/Budget"],
                "mouse": ["Office/Budget"]
            },
            "High-End Gaming": {
                "cpu": ["Gaming"],
                "motherboard": ["Gaming"],
                "ram": ["Gaming"],
                "storage": ["Gaming"],
                "gpu": ["High-End Gaming"],
                "keyboard": ["Premium Gaming"],
                "mouse": ["Premium Gaming"]
            },
            "Office": {
                "cpu": ["Office"],
                "motherboard": ["Office"],
                "ram": ["Office"],
                "storage": ["Office"],
                "gpu": ["Budget Gaming"],
                "keyboard": ["Office/Budget"],
                "mouse": ["Office/Budget"]
            },
            "Workstation": {
                "cpu": ["Design"],
                "motherboard": ["Design"],
                "ram": ["Design"],
                "storage": ["Storage"],
                "gpu": ["Workstation"],
                "keyboard": ["Compact"],
                "mouse": ["Portable"]
            }
        }

    def load_components(self, file_path: str, parser):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
        df = pd.read_csv(file_path)
        df = df.fillna({'price': 0, 'performance_score': 0})
        return [parser(row) for _, row in df.iterrows()]

    def parse_cpu(self, row) -> CPU:
        return CPU(
            name=row['name'],
            price=float(row['price']) if row['price'] else 0.0,
            performance_score=float(row['performance_score']) if row['performance_score'] else 0.0,
            purpose=row['purpose'],
            socket=row['socket']
        )

    def parse_motherboard(self, row) -> Motherboard:
        return Motherboard(
            name=row['name'],
            price=float(row['price']) if row['price'] else 0.0,
            performance_score=float(row['performance_score']) if row['performance_score'] else 0.0,
            purpose=row['purpose'],
            socket=row['socket'],
            memory_type=row['memory_type']
        )

    def parse_ram(self, row) -> RAM:
        return RAM(
            name=row['name'],
            price=float(row['price']) if row['price'] else 0.0,
            performance_score=float(row['performance_score']) if row['performance_score'] else 0.0,
            purpose=row['purpose'],
            ddr_type=row['ddr_type']
        )

    def parse_storage(self, row) -> Storage:
        return Storage(
            name=row['name'],
            price=float(row['price']) if row['price'] else 0.0,
            performance_score=float(row['performance_score']) if row['performance_score'] else 0.0,
            purpose=row['purpose'],
            type=row['type']
        )

    def parse_gpu(self, row) -> GPU:
        return GPU(
            name=row['name'],
            price=float(row['price']) if row['price'] else 0.0,
            performance_score=float(row['performance_score']) if row['performance_score'] else 0.0,
            purpose=row['purpose'],
            memory=float(row['memory']) if row['memory'] else 0.0
        )
    
    def parse_keyboard(self, row) -> Keyboard:
        return Keyboard(
            name=row['name'],
            price=float(row['price']) if row['price'] else 0.0,
            performance_score=float(row['performance_score']) if row['performance_score'] else 0.0,
            purpose=row['purpose'],
        )

    def parse_mouse(self, row) -> Mouse:
        return Mouse(
            name=row['name'],
            price=float(row['price']) if row['price'] else 0.0,
            performance_score=float(row['performance_score']) if row['performance_score'] else 0.0,
            purpose=row['purpose'],
        )

    def suggest_config(self, budget: float, purpose: str, top_n: int = 3) -> dict:
        if purpose not in self.purpose_mapping:
            return {"error": f"Invalid purpose: {purpose}"}

        allowed = self.purpose_mapping[purpose]

        def top_components(components, key, n):
            filtered = [c for c in components if c.purpose in allowed[key]]
            return sorted(filtered, key=lambda x: x.performance_score, reverse=True)[:n]

        # Filter top CPUs and motherboards with matching sockets
        top_cpus = top_components(self.cpus, 'cpu', top_n)
        top_mbs = [mb for mb in top_components(self.motherboards, 'motherboard', top_n * 2)
                   if any(mb.socket == cpu.socket for cpu in top_cpus)]
        top_cpus = [cpu for cpu in top_cpus if any(cpu.socket == mb.socket for mb in top_mbs)]

        # Filter top RAMs compatible with motherboards
        top_rams = [ram for ram in top_components(self.rams, 'ram', top_n * 2)
                    if any(ram.ddr_type in mb.memory_type for mb in top_mbs)]

        top_storages = top_components(self.storages, 'storage', top_n)
        top_gpus = top_components(self.gpus, 'gpu', top_n)
        top_keyboards = top_components(self.keyboards, 'keyboard', top_n)
        top_mice = top_components(self.mice, 'mouse', top_n)

        best_config = None
        best_score = -1

        for cpu, mb, ram, storage, gpu, keyboard, mouse in itertools.product(
                top_cpus, top_mbs, top_rams, top_storages, top_gpus, top_keyboards, top_mice):

            if cpu.socket != mb.socket or ram.ddr_type not in mb.memory_type:
                continue

            total_price = sum([
                cpu.price, mb.price, ram.price, storage.price,
                gpu.price, keyboard.price, mouse.price
            ])

            if total_price > budget:
                continue

            total_score = sum([
                cpu.performance_score, mb.performance_score, ram.performance_score,
                storage.performance_score, gpu.performance_score,
                keyboard.performance_score, mouse.performance_score
            ])

            if total_score > best_score:
                best_score = total_score
                best_config = {
                    "cpu": cpu,
                    "motherboard": mb,
                    "ram": ram,
                    "storage": storage,
                    "gpu": gpu,
                    "keyboard": keyboard,
                    "mouse": mouse,
                    "total_price": total_price,
                    "total_performance_score": total_score
                }

        if not best_config:
            return {"error": "No valid configuration found within budget."}

        # Return summarized result
        return {
            k: {"name": v.name, "price": v.price, "performance_score": v.performance_score}
            for k, v in best_config.items() if k != "total_price" and k != "total_performance_score"
        } | {"total_price": best_config['total_price'], "total_performance_score": best_config['total_performance_score']}