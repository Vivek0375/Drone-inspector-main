import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

class MLDroneInterceptor:
    def __init__(self):
        self.trajectory_model = None
        self.threat_model = None
        self.interception_model = None
        self.scaler_traj = StandardScaler()
        self.scaler_threat = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models_loaded = False
        self.initialize_models() 
        #defines the method
    
    def initialize_models(self):
        """Initialize and train ML models without TensorFlow"""
        try:
            if (os.path.exists('trajectory_model.pkl') and 
                os.path.exists('threat_model.pkl') and
                os.path.exists('interception_model.pkl')):
                
                self.trajectory_model = joblib.load('trajectory_model.pkl')
                self.threat_model = joblib.load('threat_model.pkl')
                self.interception_model = joblib.load('interception_model.pkl')
                self.scaler_threat = joblib.load('threat_scaler.pkl')
                self.label_encoder = joblib.load('label_encoder.pkl')
                self.models_loaded = True
                print("ML Models loaded successfully")
            else:
                self.train_models()
                self.models_loaded = True
        except Exception as e:
            print(f"Error loading models: {e}")
            self.train_models()
    
    def train_models(self):
        """Train ML models with synthetic data"""
        print("Training ML models...")
        
        self._train_trajectory_model()
        self._train_threat_model()
        self._train_interception_model()
        
        joblib.dump(self.trajectory_model, 'trajectory_model.pkl')
        joblib.dump(self.threat_model, 'threat_model.pkl')
        joblib.dump(self.interception_model, 'interception_model.pkl')
        joblib.dump(self.scaler_threat, 'threat_scaler.pkl')
        joblib.dump(self.label_encoder, 'label_encoder.pkl')
        
        print("ML Models trained and saved successfully")
    
    def _train_trajectory_model(self):
        """Train MLP for trajectory prediction"""
        np.random.seed(42)
        n_samples = 2000  # Reduced for faster training
        
        X = []
        y = []
        
        for _ in range(n_samples):
            start_x, start_y = np.random.uniform(0, 1000), np.random.uniform(0, 800)
            vx, vy = np.random.uniform(-8, 8), np.random.uniform(-8, 8)
            
            window_features = []
            for step in range(5):  # Reduced window size
                window_features.extend([start_x + vx * step, start_y + vy * step, vx, vy])
            
            next_x = start_x + vx * 5
            next_y = start_y + vy * 5
            
            X.append(window_features)
            y.append([next_x, next_y])
        
        X = np.array(X)
        y = np.array(y)
        
        self.trajectory_model = MLPRegressor(
            hidden_layer_sizes=(32, 16),  # Reduced complexity
            activation='relu',
            solver='adam',
            max_iter=500,  # Reduced iterations
            random_state=42,
            learning_rate_init=0.01
        )
        self.trajectory_model.fit(X, y)
    
    def _train_threat_model(self):
        """Train threat assessment classifier"""
        np.random.seed(42)
        n_samples = 1000  # Reduced for faster training
        
        # Encode drone types
        drone_types = ['standard', 'stealth', 'swarm', 'kamikaze']
        self.label_encoder.fit(drone_types)
        
        X = np.random.randn(n_samples, 5)  # Reduced features
        X[:, 0] = np.abs(X[:, 0] * 10 + 20)    # Speed (10-30 m/s)
        X[:, 1] = np.abs(X[:, 1] * 80 + 100)   # Altitude (20-180 m)
        X[:, 2] = np.abs(X[:, 2] * 200 + 300)  # Distance to target (100-500 m)
        X[:, 3] = X[:, 3] * np.pi              # Heading angle
        X[:, 4] = np.random.randint(0, 4, n_samples)  # Encoded drone type
        
        # Threat labels based on simpler rules
        y = ((X[:, 0] > 25) & (X[:, 2] < 350)).astype(int)
        
        self.scaler_threat.fit(X)
        X_scaled = self.scaler_threat.transform(X)
        
        self.threat_model = RandomForestClassifier(
            n_estimators=50,  # Reduced trees
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.threat_model.fit(X_scaled, y)
    
    def _train_interception_model(self):
        """Train interception strategy model"""
        np.random.seed(42)
        n_samples = 800  # Reduced for faster training
        
        X = np.random.randn(n_samples, 6)  # Reduced features
        X[:, 0] = X[:, 0] * 150 + 500  # drone x
        X[:, 1] = X[:, 1] * 100 + 400  # drone y
        X[:, 2] = X[:, 2] * 4          # drone vx
        X[:, 3] = X[:, 3] * 4          # drone vy
        X[:, 4] = X[:, 4] * 80 + 500   # interceptor x
        X[:, 5] = X[:, 5] * 80 + 600   # interceptor y
        
        # Target: optimal intercept time (simplified)
        y = np.sqrt(X[:, 0]**2 + X[:, 1]**2) / 20 + np.random.normal(0, 0.3, n_samples)
        
        self.interception_model = GradientBoostingRegressor(
            n_estimators=50,  # Reduced trees
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        self.interception_model.fit(X, y)
    
    def predict_trajectory(self, trajectory_data, prediction_steps=3):
        """Predict future drone trajectory"""
        if len(trajectory_data) < 5:
            return self._physics_based_prediction(trajectory_data, prediction_steps)
        
        try:
            recent_points = trajectory_data[-5:]
            features = []
            
            for point in recent_points:
                features.extend([point['x'], point['y'], point['vx'], point['vy']])
            
            features = np.array(features).reshape(1, -1)
            
            # Check for NaN values
            if np.any(np.isnan(features)):
                return self._physics_based_prediction(trajectory_data, prediction_steps)
                
            prediction = self.trajectory_model.predict(features)[0]
            
            return [{
                'x': float(prediction[0]),
                'y': float(prediction[1]),
                'step': 1,
                'confidence': 0.7
            }]
        
        except Exception as e:
            print(f"Trajectory prediction error: {e}")
            return self._physics_based_prediction(trajectory_data, prediction_steps)
    
    def _physics_based_prediction(self, trajectory_data, steps):
        """Physics-based fallback prediction"""
        if not trajectory_data:
            return []
            
        last_point = trajectory_data[-1]
        predictions = []
        
        for i in range(steps):
            predictions.append({
                'x': last_point['x'] + last_point['vx'] * (i + 1) * 0.2,
                'y': last_point['y'] + last_point['vy'] * (i + 1) * 0.2,
                'step': i + 1,
                'confidence': 0.6 - (i * 0.1)
            })
        
        return predictions
    
    def assess_threat(self, drone_state, target_position):
        """Assess threat level of incoming drone"""
        try:
            position = drone_state['position']
            velocity = drone_state['velocity']
            speed = float(np.linalg.norm(velocity))
            distance_to_target = float(np.linalg.norm(np.array(position) - np.array(target_position)))
            heading = float(np.arctan2(velocity[1], velocity[0]))
            drone_type = drone_state.get('type', 'standard')
            
            # Encode drone type
            try:
                drone_type_encoded = self.label_encoder.transform([drone_type])[0]
            except:
                drone_type_encoded = 0  # Default to standard
            
            features = np.array([[
                speed, 
                float(position[1]), 
                distance_to_target, 
                heading, 
                float(drone_type_encoded)
            ]])
            
            # Check for valid features
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                return "MEDIUM", 0.5
                
            features_scaled = self.scaler_threat.transform(features)
            threat_probability = float(self.threat_model.predict_proba(features_scaled)[0][1])
            
            # Determine threat level with confidence
            if threat_probability > 0.7:
                return "HIGH", threat_probability
            elif threat_probability > 0.4:
                return "MEDIUM", threat_probability
            else:
                return "LOW", threat_probability
                
        except Exception as e:
            print(f"Threat assessment error: {e}")
            return "MEDIUM", 0.5
    
    def optimize_interception(self, drone_state, interceptor_state):
        """Calculate optimal interception parameters"""
        try:
            drone_pos = drone_state['position']
            drone_vel = drone_state['velocity']
            interceptor_pos = interceptor_state['position']
            
            # Calculate features
            features = np.array([[
                float(drone_pos[0]), float(drone_pos[1]), 
                float(drone_vel[0]), float(drone_vel[1]),
                float(interceptor_pos[0]), float(interceptor_pos[1])
            ]])
            
            # Check for valid features
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                return self._fallback_interception(drone_state, interceptor_state)
                
            optimal_time = float(self.interception_model.predict(features)[0])
            
            # Calculate intercept point
            intercept_point = [
                drone_pos[0] + drone_vel[0] * optimal_time,
                drone_pos[1] + drone_vel[1] * optimal_time
            ]
            
            # Simple strategy selection
            distance = np.linalg.norm(np.array(drone_pos) - np.array(interceptor_pos))
            
            if distance < 150:
                strategy = "DIRECT_PURSUIT"
                speed_multiplier = 1.3
            else:
                strategy = "LEAD_PURSUIT" 
                speed_multiplier = 1.6
            
            return {
                'intercept_point': intercept_point,
                'estimated_time': optimal_time,
                'strategy': strategy,
                'speed_multiplier': speed_multiplier,
                'confidence': 0.7
            }
            
        except Exception as e:
            print(f"Interception optimization error: {e}")
            return self._fallback_interception(drone_state, interceptor_state)
    
    def _fallback_interception(self, drone_state, interceptor_state):
        """Fallback interception strategy"""
        return {
            'intercept_point': drone_state['position'],
            'estimated_time': 8.0,
            'strategy': "DIRECT_PURSUIT",
            'speed_multiplier': 1.4,
            'confidence': 0.5
        }
