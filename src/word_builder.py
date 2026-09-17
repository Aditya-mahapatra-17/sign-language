import time
from src.config import PREDICTION_COOLDOWN

class WordBuilder:
    def __init__(self, cooldown=PREDICTION_COOLDOWN):
        """
        Manages the constructed word state, avoiding duplicate inputs via cooldowns.
        """
        self.current_word = ""
        self.history = []
        self.cooldown = cooldown
        self.last_add_time = 0
        self.last_added_letter = None
        
    def add_letter(self, letter):
        """
        Attempts to add a letter to the word if the cooldown has passed 
        and it's not a duplicate rapid firing.
        """
        current_time = time.time()
        
        # Debounce/Cooldown check
        if current_time - self.last_add_time < self.cooldown:
            return False, "Cooldown active"
            
        # Prevent rapid duplicate letters unless explicitly wanted (handled by UI button in some cases)
        # For automatic continuous recognition, we require the gesture to change or cooldown to fully pass
        # But here we enforce a strict cooldown for ANY automatic letter addition to give the user time.
        
        if letter == 'space':
            self.current_word += " "
            self.history.append("space")
        elif letter == 'del':
            self.delete_last()
        elif letter == 'nothing':
            return False, "No action"
        else:
            self.current_word += letter
            self.history.append(letter)
            
        self.last_add_time = current_time
        self.last_added_letter = letter
        return True, "Letter added"
        
    def delete_last(self):
        """Deletes the last character from the current word."""
        if len(self.current_word) > 0:
            self.current_word = self.current_word[:-1]
            self.history.append("del")
            
    def clear(self):
        """Clears the entire word."""
        self.current_word = ""
        self.history.append("clear")
        
    def get_word(self):
        return self.current_word
        
    def get_history(self):
        return self.history
