from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import threading
import time
from ml_model import MLDroneInterceptor
from simulation_engine import SimulationEngine

app = Flask(__name__)
CORS(app)
SIMULATION_WIDTH = 1000  # Reduced size for better performance
SIMULATION_HEIGHT = 700
TARGET_POSITION = [500, 350]
INTERCEPTOR_BASE = [500, 600]

# Initialize ML model and simulation engine of system
print("Initializing ML Models...")
ml_model = MLDroneInterceptor()
print("ML Models initialized successfully!")

print("Initializing Simulation Engine...")
simulation = SimulationEngine(
    width=SIMULATION_WIDTH,
    height=SIMULATION_HEIGHT,
    target_position=TARGET_POSITION,
    base_position=INTERCEPTOR_BASE
)
simulation.set_ml_model(ml_model)
print("Simulation Engine initialized successfully!")

# Global control
simulation_running = True
simulation_speed = 2.0  # Increased base speed

def simulation_loop():
    """Optimized simulation thread"""
    while simulation_running:
        start_time = time.time()
        simulation.update(0.1 * simulation_speed)  # Larger time steps
        elapsed = time.time() - start_time
        sleep_time = max(0.01, 0.05 - elapsed)  # Target ~20 FPS
        time.sleep(sleep_time)

# Start simulation thread
sim_thread = threading.Thread(target=simulation_loop, daemon=True)
sim_thread.start()

# REST API Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/simulation/state', methods=['GET'])
def get_simulation_state():
    try:
        state = simulation.get_simulation_state()
        return jsonify({
            'success': True,
            'data': state,
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to get simulation state: {str(e)}'
        }), 500

@app.route('/api/drones/launch', methods=['POST'])
def launch_drone():
    try:
        data = request.get_json() or {}
        drone_type = data.get('type', 'standard')
        
        drone = simulation.launch_drone(drone_type)
        
        return jsonify({
            'success': True,
            'message': f'{drone.type.capitalize()} drone launched',
            'drone_id': drone.id,
            'drone_type': drone.type
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to launch drone: {str(e)}'
        }), 400

@app.route('/api/interceptors/launch', methods=['POST'])
def launch_interceptor():
    try:
        data = request.get_json() or {}
        target_drone_id = data.get('drone_id')
        
        interceptor, message = simulation.launch_interceptor(target_drone_id)
        
        if interceptor:
            return jsonify({
                'success': True,
                'message': message,
                'interceptor_id': interceptor.id
            })
        else:
            return jsonify({
                'success': False,
                'message': message
            }), 400
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to launch interceptor: {str(e)}'
        }), 400

@app.route('/api/simulation/reset', methods=['POST'])
def reset_simulation():
    try:
        simulation.reset()
        return jsonify({
            'success': True,
            'message': 'Simulation reset'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to reset simulation: {str(e)}'
        }), 400

@app.route('/api/simulation/control', methods=['POST'])
def control_simulation():
    try:
        data = request.get_json() or {}
        global simulation_speed
        
        if 'speed' in data:
            simulation_speed = max(0.5, min(5.0, float(data['speed'])))
        
        return jsonify({
            'success': True,
            'message': f'Speed: {simulation_speed}x',
            'speed': simulation_speed
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to control simulation: {str(e)}'
        }), 400

@app.route('/api/ml/status', methods=['GET'])
def get_ml_status():
    try:
        return jsonify({
            'success': True,
            'ml_models_loaded': ml_model.models_loaded
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to get ML status: {str(e)}'
        }), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'success': True,
        'status': 'running',
        'active_drones': len([d for d in simulation.drones if d.active]),
        'active_interceptors': len([i for i in simulation.interceptors if i.active])
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Drone Interceptor REST API Started!")
    print("="*50)
    print(f"📊 Dashboard: http://localhost:5000")
    print(f"🔗 API: http://localhost:5000/api/")
    print(f"⚡ Speed: {simulation_speed}x")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)  # debug=False for better performance