from flask import Flask, request, jsonify
from src.scripts.builder import PCBuilder

app = Flask(__name__)

# Initialize PCBuilder
builder = PCBuilder(
    cpu_file="data/cpu_cleaned.csv",
    mb_file="data/motherboard_cleaned.csv",
    ram_file="data/memory_cleaned.csv",
    storage_file="data/internal_hard_drive_cleaned.csv",
    gpu_file="data/video_card_cleaned.csv",
    keyboard_file="data/keyboard_cleaned.csv",
    mouse_file="data/mouse_cleaned.csv"
)

@app.route('/suggest', methods=['POST'])
def suggest_config():
    data = request.get_json()
    budget = data.get('budget')
    purpose = data.get('purpose')

    if not budget or not isinstance(budget, (int, float)) or budget <= 0:
        return jsonify({"error": "Invalid or missing budget"}), 400
    if not purpose or purpose not in [
        "Budget Gaming", "Mid-Range Gaming", "High-End Gaming", "Office", "Workstation"
    ]:
        return jsonify({"error": "Invalid or missing purpose"}), 400

    try:
        config = builder.suggest_config(budget, purpose)
        return jsonify(config), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
