import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.datasets import mnist

# Load posit(8) decode table
posit8_decode = np.load("posit8_float_decode.npy")

def float_to_posit8(val):
    # Multiply by scale to prevent everything mapping to zero
    scaled_val = val * 8.0
    idx = np.abs(posit8_decode - scaled_val).argmin()
    if idx == 128:  # Avoid NaR
        return 0
    return np.uint8(idx)

# Load model
ckpt_path = './posit8.ckpt'
sess = tf.Session()
saver = tf.train.import_meta_graph(ckpt_path + '.meta')
saver.restore(sess, ckpt_path)
graph = tf.get_default_graph()

# Load MNIST digit "1"
(_, _), (X_test, y_test) = mnist.load_data()
idx = np.where(y_test == 1)[0][0]
img = X_test[idx]
print(f"Using MNIST test index {idx}, label = {y_test[idx]}")

# Resize + normalize to [-1, 1]
img32 = Image.fromarray(img).resize((32, 32), Image.LANCZOS)
img32 = (np.array(img32).astype(np.float32) - 127.5) / 127.5
img32 = img32.reshape((1, 32, 32, 1))

# Tensor extraction
x_ph = graph.get_tensor_by_name('inputs:0')
fc2_out_tensor = graph.get_tensor_by_name('Relu_3:0')
fc2_out = sess.run(fc2_out_tensor, feed_dict={x_ph: img32})[0]

# Exit if fc2_out is truly flat
if not np.any(fc2_out):
    print("[!] Error: fc2_out is all zeros. Try a different image.")
    exit(1)

# Scale + encode activations
fc2_out_uint8 = np.array([float_to_posit8(v) for v in fc2_out], dtype=np.uint8)

# Load and encode final layer
fc3_W = graph.get_tensor_by_name('Variable_8:0').eval(session=sess)
fc3_b = graph.get_tensor_by_name('Variable_9:0').eval(session=sess)

fc3_W_uint8 = np.array([[float_to_posit8(w) for w in row] for row in fc3_W], dtype=np.uint8)
fc3_b_uint8 = np.array([float_to_posit8(b) for b in fc3_b], dtype=np.uint8)

# Save outputs
np.save("fc2_out.npy", fc2_out_uint8)
np.save("fc3_W.npy", fc3_W_uint8)
np.save("fc3_b.npy", fc3_b_uint8)

# Print confirmation
print("[*] Saved:")
print("  fc2_out.npy shape:", fc2_out_uint8.shape, "unique:", np.unique(fc2_out_uint8))
print("  fc3_W.npy   shape:", fc3_W_uint8.shape,   "unique:", np.unique(fc3_W_uint8))
print("  fc3_b.npy   shape:", fc3_b_uint8.shape,   "unique:", np.unique(fc3_b_uint8))
