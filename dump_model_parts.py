import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.datasets import mnist

ckpt_path = './posit8.ckpt'
sess = tf.Session()
saver = tf.train.import_meta_graph(ckpt_path + '.meta')
saver.restore(sess, ckpt_path)
graph = tf.get_default_graph()

(_, _), (X_test, y_test) = mnist.load_data()
idx = np.where(y_test == 1)[0][0]
img = X_test[idx]
img32 = Image.fromarray(img).resize((32, 32), Image.LANCZOS)
img32 = (np.array(img32).astype(np.float32) - 127.5) / 127.5
img32 = img32.reshape((1, 32, 32, 1))

x_ph = graph.get_tensor_by_name('inputs:0')
fc2_out_tensor = graph.get_tensor_by_name('Relu_2:0')  # adjust if needed
fc2_out = sess.run(fc2_out_tensor, feed_dict={x_ph: img32})[0]

fc3_W = graph.get_tensor_by_name('Variable_8:0').eval(session=sess)
fc3_b = graph.get_tensor_by_name('Variable_9:0').eval(session=sess)

# Convert to raw bit values (as 8-bit unsigned integers)
fc2_out_uint8 = fc2_out.astype(np.uint8)
fc3_W_uint8 = fc3_W.astype(np.uint8)
fc3_b_uint8 = fc3_b.astype(np.uint8)

# Save to .npy with safe types
np.save("fc2_out.npy", fc2_out_uint8)
np.save("fc3_W.npy", fc3_W_uint8)
np.save("fc3_b.npy", fc3_b_uint8)

print("Resaved all outputs as uint8.")
