import numpy as np
import random
import math
from datetime import datetime

class Drone:
    def __init__(self, drone_id, start_position, target_position, drone_type="standard"):
        self.id = drone_id
        self.position = np.array(start_position, dtype=float)
        self.target = np.array(target_position, dtype=float)
        self.velocity = np.array([0.0, 0.0])
        self.type = drone_type
        self.active = True
        self.threat_level = "MEDIUM"
        self.threat_confidence = 0.5
        self.trajectory = []
        self.evasion_mode = False
        
        self._set_drone_characteristics()
        
    def _set_drone_characteristics(self):
        """Set drone characteristics with optimized speeds"""
        if self.type == "swarm":
            self.max_speed = random.uniform(4, 6)    # Reduced speed
            self.agility = random.uniform(1.2, 1.8)  # Reduced agility
            self.evasion_probability = 0.4
        elif self.type == "stealth":
            self.max_speed = random.uniform(5, 7)
            self.agility = random.uniform(0.8, 1.2)
            self.evasion_probability = 0.3
        elif self.type == "kamikaze":
            self.max_speed = random.uniform(6, 8)
            self.agility = random.uniform(0.6, 1.0)
            self.evasion_probability = 0.1
        else:  # standard
            self.max_speed = random.uniform(3, 5)
            self.agility = random.uniform(1.0, 1.5)
            self.evasion_probability = 0.2
    
    def update(self, dt, interceptors=None):
        """Optimized update method"""
        if not self.active:
            return False
        
        # Store trajectory (limited to recent points)
        if len(self.trajectory) < 20:  # Limit trajectory history
            self.trajectory.append({
                'x': float(self.position[0]),
                'y': float(self.position[1]),
                'vx': float(self.velocity[0]),
                'vy': float(self.velocity[1]),
            })
        
        # Check if reached target
        distance_to_target = np.linalg.norm(self.position - self.target)
        if distance_to_target < 30:
            self.active = False
            return True
        
        # Calculate direction to target
        direction = self.target - self.position
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction = direction / distance
        
        # Simple evasion (reduced complexity)
        evasion = np.array([0.0, 0.0])
        if interceptors and random.random() < self.evasion_probability:
            for interceptor in interceptors:
                if interceptor.active:
                    to_interceptor = interceptor.position - self.position
                    interceptor_distance = np.linalg.norm(to_interceptor)
                    if interceptor_distance < 150:
                        evasion -= to_interceptor / (interceptor_distance + 0.1)
                        self.evasion_mode = True
        
        # Combine directions
        if np.linalg.norm(evasion) > 0:
            evasion = evasion / np.linalg.norm(evasion)
            combined_direction = direction * 0.8 + evasion * 0.2
        else:
            combined_direction = direction
            self.evasion_mode = False
        
        # Normalize and add slight randomness
        if np.linalg.norm(combined_direction) > 0:
            combined_direction = combined_direction / np.linalg.norm(combined_direction)
        
        # Add minimal noise
        combined_direction += np.random.normal(0, 0.02, 2)
        if np.linalg.norm(combined_direction) > 0:
            combined_direction = combined_direction / np.linalg.norm(combined_direction)
        
        # Update velocity (simplified)
        desired_velocity = combined_direction * self.max_speed
        steering = (desired_velocity - self.velocity) * self.agility * 0.5
        
        self.velocity += steering * dt
        
        # Limit speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed
        
        # Update position
        self.position += self.velocity * dt
        
        return False
    
    def get_state(self):
        """Get current drone state"""
        speed = float(np.linalg.norm(self.velocity))
        heading = float(np.arctan2(self.velocity[1], self.velocity[0])) if speed > 0 else 0.0
        
        return {
            'id': self.id,
            'position': [float(self.position[0]), float(self.position[1])],
            'velocity': [float(self.velocity[0]), float(self.velocity[1])],
            'speed': speed,
            'heading': heading,
            'type': self.type,
            'active': self.active,
            'threat_level': self.threat_level,
            'threat_confidence': float(self.threat_confidence),
            'trajectory_length': len(self.trajectory),
            'evasion_mode': self.evasion_mode
        }

class Interceptor:
    def __init__(self, interceptor_id, base_position):
        self.id = interceptor_id
        self.position = np.array(base_position, dtype=float)
        self.velocity = np.array([0.0, 0.0])
        self.target_drone = None
        self.intercept_point = None
        self.active = False
        self.fuel = 100.0
        self.max_speed = 12.0  # Reduced speed
        self.agility = 3.0
        self.interception_strategy = None
        self.engagement_range = 35
    
    def launch(self, target_drone, ml_strategy):
        self.target_drone = target_drone
        self.interception_strategy = ml_strategy
        self.active = True
        self.fuel = 100.0
        
        if ml_strategy and 'intercept_point' in ml_strategy:
            self.intercept_point = np.array(ml_strategy['intercept_point'])
        else:
            self.intercept_point = target_drone.position.copy()
    
    def update(self, dt, ml_model):
        if not self.active or not self.target_drone or not self.target_drone.active:
            return False
        
        # Consume fuel
        self.fuel -= dt * 1.5
        if self.fuel <= 0:
            self.active = False
            return False
        
        # Simple intercept point calculation
        if self.intercept_point is None:
            self.intercept_point = self.target_drone.position.copy()
        
        # Calculate direction to intercept point
        direction = self.intercept_point - self.position
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction = direction / distance
        
        # Apply strategy speed multiplier
        speed_multiplier = self.interception_strategy.get('speed_multiplier', 1.2) if self.interception_strategy else 1.2
        
        # Update velocity
        desired_velocity = direction * self.max_speed * speed_multiplier
        steering = (desired_velocity - self.velocity) * self.agility * 0.3
        
        self.velocity += steering * dt
        
        # Limit speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed * speed_multiplier:
            self.velocity = (self.velocity / speed) * self.max_speed * speed_multiplier
        
        # Update position
        self.position += self.velocity * dt
        
        # Check for interception
        if self._check_interception():
            self.target_drone.active = False
            self.active = False
            return True
        
        return False
    
    def _check_interception(self):
        if not self.target_drone:
            return False
        
        distance = np.linalg.norm(self.position - self.target_drone.position)
        return distance < self.engagement_range
    
    def get_state(self):
        speed = float(np.linalg.norm(self.velocity))
        
        return {
            'id': self.id,
            'position': [float(self.position[0]), float(self.position[1])],
            'velocity': [float(self.velocity[0]), float(self.velocity[1])],
            'speed': speed,
            'target_drone': self.target_drone.id if self.target_drone else None,
            'active': self.active,
            'fuel': float(self.fuel),
            'intercept_point': [float(self.intercept_point[0]), float(self.intercept_point[1])] if self.intercept_point is not None else None,
            'strategy': self.interception_strategy
        }

class SimulationEngine:
    def __init__(self, width, height, target_position, base_position):
        self.width = width
        self.height = height
        self.target_position = target_position
        self.base_position = base_position
        self.drones = []
        self.interceptors = []
        self.ml_model = None
        self.simulation_time = 0.0
        self.statistics = {
            'total_drones_launched': 0,
            'drones_intercepted': 0,
            'drones_reached_target': 0,
            'success_rate': 0.0
        }
    
    def set_ml_model(self, ml_model):
        self.ml_model = ml_model
    
    def launch_drone(self, drone_type="standard"):
        """Launch drone with optimized positioning"""
        side = random.choice(['top', 'right'])  # Simplified spawn locations
        if side == 'top':
            start_x = random.uniform(100, self.width-100)
            start_y = 50
        else:  # right
            start_x = self.width - 50
            start_y = random.uniform(100, self.height-100)
        
        drone_id = f"D{len(self.drones)+1:02d}"
        drone = Drone(drone_id, [start_x, start_y], self.target_position, drone_type)
        self.drones.append(drone)
        self.statistics['total_drones_launched'] += 1
        
        return drone
    
    def launch_interceptor(self, target_drone_id=None):
        if not self.ml_model:
            return None, "ML model not available"
        
        # Find target drone
        if target_drone_id:
            target_drone = next((d for d in self.drones if d.id == target_drone_id and d.active), None)
        else:
            active_drones = [d for d in self.drones if d.active]
            if not active_drones:
                return None, "No active drones to intercept"
            
            # Simple targeting - closest drone to target
            target_drone = min(active_drones, 
                             key=lambda d: np.linalg.norm(d.position - np.array(self.target_position)))
        
        if not target_drone:
            return None, "Target drone not found or inactive"
        
        # Get ML strategy
        drone_state = target_drone.get_state()
        interceptor_state = {'position': self.base_position}
        strategy = self.ml_model.optimize_interception(drone_state, interceptor_state)
        
        interceptor_id = f"I{len(self.interceptors)+1:02d}"
        interceptor = Interceptor(interceptor_id, self.base_position)
        interceptor.launch(target_drone, strategy)
        self.interceptors.append(interceptor)
        
        return interceptor, f"Interceptor launched against {target_drone.id}"
    
    def update(self, dt):
        """Optimized update method"""
        self.simulation_time += dt
        
        # Update drones
        for drone in self.drones:
            if drone.active:
                target_reached = drone.update(dt, self.interceptors)
                
                if target_reached:
                    self.statistics['drones_reached_target'] += 1
                
                # Update threat assessment
                if self.ml_model:
                    drone_state = drone.get_state()
                    threat_level, confidence = self.ml_model.assess_threat(drone_state, self.target_position)
                    drone.threat_level = threat_level
                    drone.threat_confidence = confidence
        
        # Update interceptors
        for interceptor in self.interceptors:
            if interceptor.active:
                interception_success = interceptor.update(dt, self.ml_model)
                if interception_success:
                    self.statistics['drones_intercepted'] += 1
        
        # Clean up inactive entities periodically
        if random.random() < 0.1:  # 10% chance each frame
            self.drones = [d for d in self.drones if d.active or random.random() < 0.9]
            self.interceptors = [i for i in self.interceptors if i.active]
        
        # Update statistics
        total_ended = self.statistics['drones_intercepted'] + self.statistics['drones_reached_target']
        if total_ended > 0:
            self.statistics['success_rate'] = self.statistics['drones_intercepted'] / total_ended * 100
    
    def get_simulation_state(self):
        return {
            'drones': [drone.get_state() for drone in self.drones],
            'interceptors': [interceptor.get_state() for interceptor in self.interceptors],
            'environment': {
                'width': self.width,
                'height': self.height,
                'target_position': self.target_position,
                'base_position': self.base_position,
                'simulation_time': self.simulation_time
            },
            'statistics': self.statistics
        }
    
    def reset(self):
        self.drones.clear()
        self.interceptors.clear()
        self.simulation_time = 0.0
        self.statistics = {
            'total_drones_launched': 0,
            'drones_intercepted': 0,
            'drones_reached_target': 0,
            'success_rate': 0.0
        }