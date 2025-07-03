import pickle

def save_embeddings(embeddings, filename):
    """Save embeddings to a pickle file"""
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f)
