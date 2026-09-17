import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import NUM_CLASSES
from src.model import build_baseline_model, build_transfer_learning_model
from src.smoothing import TemporalSmoother
from src.word_builder import WordBuilder

def test_model_output_shape():
    model = build_baseline_model()
    dummy_input = tf.random.normal((1, 224, 224, 3))
    preds = model(dummy_input)
    assert preds.shape == (1, NUM_CLASSES)
    
def test_transfer_learning_model_output_shape():
    model = build_transfer_learning_model()
    dummy_input = tf.random.normal((1, 224, 224, 3))
    preds = model(dummy_input)
    assert preds.shape == (1, NUM_CLASSES)
    
def test_temporal_smoother():
    smoother = TemporalSmoother(window_size=5)
    # Give it A A A A B
    smoother.update("A")
    smoother.update("A")
    smoother.update("A")
    smoother.update("A")
    res = smoother.update("B")
    assert res == "A"
    
def test_word_builder():
    wb = WordBuilder(cooldown=0.0) # Disable cooldown for fast testing
    wb.add_letter("H")
    wb.add_letter("E")
    wb.add_letter("space")
    wb.add_letter("L")
    assert wb.get_word() == "HE L"
    wb.delete_last()
    assert wb.get_word() == "HE "
    wb.clear()
    assert wb.get_word() == ""
