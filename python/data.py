from jax import numpy as jnp, random

class DataLoader:
    def __init__(self, X, Y, batch_size):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.batch = -1
        self.batches = jnp.arange((X.shape[0] // batch_size) + 1)
        self.len = len(self.batches)

    def __iter__(self): return self

    def __next__(self):
        self.batch += 1
        if self.batch < self.len:
            start = int(self.batch * self.batch_size)
            if self.batch == self.batches[-1]: end = None
            else: end = int((self.batch + 1) * self.batch_size)
            return self.X[start:end], self.Y[start:end]        
        raise StopIteration
        
        
def synthetic_data(key, w, b, num_examples):
    """Generate y = Xw + b + noise."""
    key_sample, key_noise = random.split(key)
    X = random.normal(key_sample, (num_examples, len(w)))
    y = jnp.dot(X, w) + b
    y +=  0.01 * random.normal(key_noise, (len(y),))
    return X, y
