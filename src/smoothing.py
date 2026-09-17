from collections import deque
from collections import Counter
from src.config import SMOOTHING_WINDOW

class TemporalSmoother:
    def __init__(self, window_size=SMOOTHING_WINDOW):
        """
        Maintains a history of recent predictions to stabilize output.
        """
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        
    def update(self, prediction):
        """
        Adds a new prediction to the history and returns the stabilized prediction.
        """
        self.history.append(prediction)
        return self.get_stable_prediction()
        
    def get_stable_prediction(self):
        """
        Returns the most common prediction in the history window.
        """
        if not self.history:
            return None
            
        counter = Counter(self.history)
        most_common_prediction, count = counter.most_common(1)[0]
        
        # We can also add a requirement that the most common must appear > 50% of the time
        if count >= (self.window_size // 2):
            return most_common_prediction
        return None
        
    def clear(self):
        """Clears the history."""
        self.history.clear()
