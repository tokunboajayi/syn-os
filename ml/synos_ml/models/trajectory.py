import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from loguru import logger
import pickle

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=2):
        super(TrajectoryLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class TrajectoryPredictor:
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrajectoryLSTM().to(self.device)
        self.is_fitted = False
        
        # Simple min-max scaler state
        self.lat_min = -90.0
        self.lat_max = 90.0
        self.lon_min = -180.0
        self.lon_max = 180.0
        
        if model_path:
            self.load(model_path)

    def _normalize(self, points: np.ndarray) -> np.ndarray:
        """Normalize lat/lon to [0, 1]"""
        norm = points.copy()
        norm[:, 0] = (points[:, 0] - self.lat_min) / (self.lat_max - self.lat_min + 1e-6)
        norm[:, 1] = (points[:, 1] - self.lon_min) / (self.lon_max - self.lon_min + 1e-6)
        return norm

    def _denormalize(self, points: np.ndarray) -> np.ndarray:
        """Convert [0, 1] back to lat/lon"""
        denorm = points.copy()
        denorm[:, 0] = points[:, 0] * (self.lat_max - self.lat_min + 1e-6) + self.lat_min
        denorm[:, 1] = points[:, 1] * (self.lon_max - self.lon_min + 1e-6) + self.lon_min
        return denorm

    def fit(self, trajectories: List[np.ndarray], epochs=50, lr=0.001):
        """
        Train the model on a list of trajectories.
        Each trajectory is a numpy array of shape (seq_len, 2) -> (lat, lon)
        """
        logger.info(f"Training TrajectoryPredictor on {len(trajectories)} sequences")
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Flatten and normalize
        all_points = np.vstack(trajectories)
        self.lat_min, self.lat_max = all_points[:, 0].min(), all_points[:, 0].max()
        self.lon_min, self.lon_max = all_points[:, 1].min(), all_points[:, 1].max()
        
        # Prepare sliding windows
        X, y = [], []
        window_size = 5
        
        for traj in trajectories:
            norm_traj = self._normalize(traj)
            if len(norm_traj) <= window_size:
                continue
            for i in range(len(norm_traj) - window_size):
                X.append(norm_traj[i:i+window_size])
                y.append(norm_traj[i+window_size])
                
        if not X:
            logger.warning("Not enough data to train")
            return

        X_tensor = torch.tensor(np.array(X), dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(np.array(y), dtype=torch.float32).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")
        
        self.is_fitted = True
        self.model.eval()

    def predict(self, recent_points: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Predict next point based on recent history."""
        if not self.is_fitted:
            # Return last point if not fitted
            return recent_points[-1]
            
        self.model.eval()
        points = np.array(recent_points)
        norm_points = self._normalize(points)
        
        # Ensure we have enough points, pad if necessary
        if len(norm_points) < 5:
             # simple padding with first element
             padding = np.tile(norm_points[0], (5 - len(norm_points), 1))
             norm_points = np.vstack([padding, norm_points])
        else:
             norm_points = norm_points[-5:]
             
        input_tensor = torch.tensor(norm_points, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred_norm = self.model(input_tensor).cpu().numpy()
            
        pred = self._denormalize(pred_norm)
        return float(pred[0, 0]), float(pred[0, 1])

    def save(self, path: str):
        state = {
            "model": self.model.state_dict(),
            "bounds": (self.lat_min, self.lat_max, self.lon_min, self.lon_max),
            "is_fitted": self.is_fitted
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Saved TrajectoryPredictor to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.model.load_state_dict(state["model"])
        self.lat_min, self.lat_max, self.lon_min, self.lon_max = state["bounds"]
        self.is_fitted = state["is_fitted"]
        self.model.eval()
        logger.info(f"Loaded TrajectoryPredictor from {path}")
