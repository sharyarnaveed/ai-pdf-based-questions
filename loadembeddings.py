import os
import pickle
def load_embeddings(filename):
    """Load embeddings from a pickle file"""
    if not os.path.exists(filename):
        return None
    
    with open(filename, 'rb') as f:
        return pickle.load(f)