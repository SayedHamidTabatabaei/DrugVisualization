import os

from presentation import create_app
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = create_app()

if __name__ == '__main__':
    import tensorflow as tf
    import numpy as np

    print("TensorFlow version:", tf.__version__)
    print("NumPy version:", np.__version__)
    print("GPU devices:", tf.config.list_physical_devices('GPU'))
    app.run(debug=True)
