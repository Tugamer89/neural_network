import time, random
import concurrent.futures
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndimage
from base import *
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

def augment_image(img_arr):
    img = Image.fromarray((img_arr.reshape(28, 28) * 255).astype(np.uint8))

    angle = random.uniform(-10, 10)
    img = img.rotate(angle)

    # Shift
    max_shift = 2
    tx, ty = random.randint(-max_shift, max_shift), random.randint(-max_shift, max_shift)
    img = ImageOps.expand(img, border=max_shift, fill=0)
    img = img.crop((max_shift + tx, max_shift + ty,
                    max_shift + tx + 28, max_shift + ty + 28))

    # Zoom
    zoom = random.uniform(0.9, 1.1)
    size = int(28 * zoom)
    img = img.resize((size, size), Image.BICUBIC)
    img = ImageOps.fit(img, (28, 28), Image.BICUBIC)

    return np.array(img, dtype=np.float32).flatten() / 255.0

def augment_batch_gpu(X_batch_gpu):
    batch_size = X_batch_gpu.shape[0]
    X_2d = X_batch_gpu.reshape(batch_size, 28, 28)
    
    angles = cp.random.uniform(-15, 15, size=batch_size)
    shifts_y = cp.random.uniform(-3, 3, size=batch_size)
    shifts_x = cp.random.uniform(-3, 3, size=batch_size)
    
    out_batch = cp.empty_like(X_2d)
    
    for i in range(batch_size):
        img = ndimage.rotate(X_2d[i], angles[i], reshape=False, order=1)
        img = ndimage.shift(img, (shifts_y[i], shifts_x[i]), order=1)
        out_batch[i] = img
        
    return out_batch.reshape(batch_size, -1)

def augment_batch_cpu(X_batch_cpu):
    augmented = [augment_image(x) for x in X_batch_cpu]
    
    return cp.asarray(augmented, dtype=cp.float32)

class Neural_Network():
    def __init__(self):
        self.loss_function = Activation_SoftMax_Loss_CatEntropy()
        self.optimizer = Optimizer_Adam(learning_rate=0.001, decay=1e-5)

        # CNN layers
        self.conv1 = Layer_Conv2D(n_filters=64, filter_size=3, input_depth=1, stride=1, padding=1)
        self.act1 = Activation_ReLU()
        self.pool1 = Layer_MaxPool2D(2,2)

        self.conv2 = Layer_Conv2D(n_filters=128, filter_size=3, input_depth=64, stride=1, padding=1)
        self.act2 = Activation_ReLU()
        self.pool2 = Layer_MaxPool2D(2,2)

        # Dense layers
        self.layer_dense1 = Layer_Dense(128*7*7, 128, weight_regularizer_l2=0.001)
        self.activation_dense1 = Activation_ReLU()
        self.dropout1 = Layer_Dropout(0.4)

        self.layer_last = Layer_Dense(128, 10)
        self.activation_last = Activation_SoftMax()

        self.layers = [
            self.conv1,
            self.conv2,
            self.layer_dense1,
            self.layer_last
        ]
        self.history = {'loss': [], 'accuracy': [], 'epoch_time': []}

    # --- full forward ---
    def forward(self, inputs, training=True):
        # reshape inputs: (batch, 784) → (batch,1,28,28)
        X = inputs.reshape(-1,1,28,28)

        self.conv1.forward(X)
        self.act1.forward(self.conv1.output)
        self.pool1.forward(self.act1.output)

        self.conv2.forward(self.pool1.output)
        self.act2.forward(self.conv2.output)
        self.pool2.forward(self.act2.output)

        # flatten
        flat = self.pool2.output.reshape(inputs.shape[0], -1)

        self.layer_dense1.forward(flat)
        self.activation_dense1.forward(self.layer_dense1.output)
        if training:
            self.dropout1.forward(self.activation_dense1.output)
            dense_out = self.dropout1.output
        else:
            dense_out = self.activation_dense1.output

        self.layer_last.forward(dense_out)
        self.activation_last.forward(self.layer_last.output)

    # --- backward ---
    def backward(self, y_true):
        # backward Softmax+CrossEntropy
        self.loss_function.backward(self.activation_last.output, y_true)
        dinputs = self.loss_function.dinputs

        # last dense
        self.layer_last.backward(dinputs)

        # dropout + dense1
        self.dropout1.backward(self.layer_last.dinputs)
        self.activation_dense1.backward(self.dropout1.dinputs)
        self.layer_dense1.backward(self.activation_dense1.dinputs)

        # reshape back to conv shape
        dflat = self.layer_dense1.dinputs.reshape(self.pool2.output.shape)

        # pool2 + conv2
        self.pool2.backward(dflat)
        self.act2.backward(self.pool2.dinputs)
        self.conv2.backward(self.act2.dinputs)

        # pool1 + conv1
        self.pool1.backward(self.conv2.dinputs)
        self.act1.backward(self.pool1.dinputs)
        self.conv1.backward(self.act1.dinputs)

        # update wheights
        self.optimizer.pre_update_params()
        for layer in [self.layer_dense1, self.layer_last, self.conv1, self.conv2]:
            self.optimizer.update_params(layer)
        self.optimizer.post_update_params()

    # --- test (val) ---
    def test(self, X, y):
        self.forward(X)
        # loss batch
        try:
            data_loss = self.loss_function.forward(self.activation_last.output, y)
            if isinstance(data_loss, cp.ndarray):
                data_loss = float(cp.mean(data_loss))
        except AttributeError:
            data_loss = self.loss_function.calculate(self.activation_last.output, y)

        # regularization loss
        reg_loss = 0.0
        if hasattr(self.loss_function, 'regularization_loss'):
            reg_fn = self.loss_function.regularization_loss
            for layer in self.layers:
                if hasattr(layer, "weights"):
                    reg_loss += reg_fn(layer)
        total_loss = data_loss + reg_loss

        preds = cp.argmax(self.activation_last.output, axis=1)
        acc = cp.mean(preds == y)
        return acc, total_loss

    # --- train batch-by-batch ---
    def train_network(self, X_train, y_train, X_val=None, y_val=None,
                  epochs=10, batch_size=128,
                  early_stopping=True, patience=5,
                  target_accuracy=None, tracking_interval=1,
                  augment=False):
        best_val_acc = 0.0
        no_improve_epochs = 0
        n_samples = len(X_train)
        print(f"Starting training: {n_samples} samples, batch_size={batch_size}, epochs={epochs}")

        for epoch in range(1, epochs+1):
            epoch_start = time.time()
            epoch_loss, epoch_correct, epoch_seen = 0.0, 0, 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                
                X_batch_cpu = X_train[start:end]
                y_batch = cp.asarray(y_train[start:end])

                if augment:
                    X_batch = augment_batch_cpu(X_batch_cpu)
                else:
                    X_batch = cp.asarray(X_batch_cpu)

                # forward
                self.forward(X_batch)

                # loss
                try:
                    data_loss = self.loss_function.forward(self.activation_last.output, y_batch)
                    if isinstance(data_loss, cp.ndarray):
                        data_loss = float(cp.mean(data_loss))
                except AttributeError:
                    data_loss = self.loss_function.calculate(self.activation_last.output, y_batch)

                reg_loss = 0.0
                if hasattr(self.loss_function, 'regularization_loss'):
                    for layer in self.layers:
                        if hasattr(layer, "weights"):
                            reg_loss += self.loss_function.regularization_loss(layer)

                loss = data_loss + reg_loss

                # accuracy
                preds = cp.argmax(self.activation_last.output, axis=1)
                epoch_correct += cp.sum(preds == y_batch)
                epoch_seen += len(y_batch)

                # backward + update
                self.backward(y_batch)
                epoch_loss += loss * len(X_batch)

            # metrics epoca
            epoch_loss /= epoch_seen
            epoch_acc = epoch_correct / epoch_seen
            epoch_time = time.time() - epoch_start

            self.history['loss'].append(epoch_loss)
            self.history['accuracy'].append(epoch_acc)
            self.history['epoch_time'].append(epoch_time)

            # validation
            val_acc, val_loss = (None, None)
            if X_val is not None and y_val is not None:
                X_val_gpu = cp.asarray(X_val)
                y_val_gpu = cp.asarray(y_val)
                val_acc, val_loss = self.test(X_val_gpu, y_val_gpu)

            # log
            if epoch % tracking_interval == 0 or epoch == epochs:
                msg = (f"Epoch {epoch}/{epochs} — train_loss: {epoch_loss:.5f}, train_acc: {epoch_acc:.4f}, "
                    f"time: {epoch_time:.3f}s, lr: {getattr(self.optimizer, 'current_learning_rate', 'N/A')}")
                if val_acc is not None:
                    msg += f", val_loss: {val_loss:.5f}, val_acc: {val_acc:.4f}"
                print(msg)

            # early stopping
            if val_acc is not None:
                if val_acc > best_val_acc + 1e-6:
                    best_val_acc = val_acc
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

                # Reduce LR on plateau
                if no_improve_epochs >= int(patience * 0.75):
                    if hasattr(self.optimizer, 'learning_rate'):
                        old_lr = self.optimizer.learning_rate
                        new_lr = max(old_lr * 0.5, 1e-5)
                        self.optimizer.learning_rate = new_lr
                        no_improve_epochs = 0
                        print(f"Validation plateau — reducing learning rate: {old_lr:.6f} → {new_lr:.6f}")

                # Early stopping
                if early_stopping and no_improve_epochs >= patience:
                    print(f"No improvement for {patience} epochs. Early stopping.")
                    break

                # Target accuracy stop
                if target_accuracy is not None and val_acc >= target_accuracy:
                    print(f"Reached target validation accuracy {val_acc:.4f} >= {target_accuracy}. Stopping.")
                    break
            else:
                if target_accuracy is not None and epoch_acc >= target_accuracy:
                    print(f"Reached target training accuracy {epoch_acc:.4f} >= {target_accuracy}. Stopping.")
                    break

        return self.history

    # --- evaluate single batch / image ---
    def evaluate(self, inputs):
        if isinstance(inputs, np.ndarray):
            inputs = cp.asarray(inputs)
        self.forward(inputs, training=False)
        
        probs = self.activation_last.output
        predictions = cp.argmax(probs, axis=1)
        confidences = cp.max(probs, axis=1)
        
        return cp.asnumpy(predictions), cp.asnumpy(confidences)

    # --- save ---
    def save(self, path):
        arrays = {}

        def to_numpy(x):
            if isinstance(x, cp.ndarray):
                return x.get()
            if isinstance(x, (cp.generic, np.ndarray, int, float)):
                return np.array(x)
            if isinstance(x, (list, tuple)):
                return np.array([to_numpy(el) for el in x])
            raise ValueError(f"Object not serializable: {type(x)}")

        for i, layer in enumerate(self.layers):
            prefix = f"layer{i}_"

            for attr in ["weights", "biases", "gamma", "beta",
                         "running_mean", "running_var",
                         "weight_momentums", "bias_momentums",
                         "weight_cache", "bias_cache"]:
                if hasattr(layer, attr):
                    value = getattr(layer, attr)
                    if value is not None:
                        arrays[prefix + attr] = to_numpy(value)

        arrays["optimizer_learning_rate"] = to_numpy(self.optimizer.learning_rate)
        arrays["optimizer_current_lr"] = to_numpy(self.optimizer.current_learning_rate)
        arrays["optimizer_decay"] = to_numpy(self.optimizer.decay)
        arrays["optimizer_iterations"] = to_numpy(self.optimizer.iterations)

        arrays["history_loss"] = to_numpy(self.history["loss"])
        arrays["history_acc"] = to_numpy(self.history["accuracy"])
        arrays["history_time"] = to_numpy(self.history["epoch_time"])

        np.savez_compressed(path, **arrays)

    def load(self, path):
        data = np.load(path, allow_pickle=True)

        def to_cupy(x):
            if isinstance(x, np.ndarray):
                return cp.asarray(x)
            return x

        for i, layer in enumerate(self.layers):
            prefix = f"layer{i}_"
            for attr in ["weights", "biases", "gamma", "beta",
                         "running_mean", "running_var",
                         "weight_momentums", "bias_momentums",
                         "weight_cache", "bias_cache"]:
                key = prefix + attr
                if key in data:
                    value = data[key]
                    if np.isscalar(value) or value.shape == ():
                        value = value.item()
                    
                    if hasattr(layer, attr):
                        expected_shape = getattr(layer, attr).shape
                        if value.shape != expected_shape:
                            print(f"Shape mismatch for {key}: expected {expected_shape}, got {value.shape}")
                        
                    setattr(layer, attr, to_cupy(value))

        if "optimizer_learning_rate" in data:
            self.optimizer.learning_rate = float(data["optimizer_learning_rate"])
        if "optimizer_current_lr" in data:
            self.optimizer.current_learning_rate = float(data["optimizer_current_lr"])
        if "optimizer_decay" in data:
            self.optimizer.decay = float(data["optimizer_decay"])
        if "optimizer_iterations" in data:
            self.optimizer.iterations = int(data["optimizer_iterations"])

        self.history["loss"] = data.get("history_loss", np.array([])).tolist()
        self.history["accuracy"] = data.get("history_acc", np.array([])).tolist()
        self.history["epoch_time"] = data.get("history_time", np.array([])).tolist()

    # --- plot ---
    def plot_history(self, save_path="training_history.png"):
        def to_cpu(arr):
            if isinstance(arr, cp.ndarray):
                return arr.get()
            if isinstance(arr, list):
                return [a.get() if isinstance(a, cp.ndarray) else a for a in arr]
            return np.array(arr)

        losses = to_cpu(self.history['loss'])
        accs = to_cpu(self.history['accuracy'])
        times = to_cpu(self.history['epoch_time'])

        epochs = range(len(losses))
        plt.figure(figsize=(28,7))

        # --- Loss ---
        plt.subplot(1,4,1)
        plt.plot(epochs, losses, label='Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # --- Accuracy ---
        plt.subplot(1,4,2)
        plt.plot(epochs, accs, label='Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # --- Time ---
        plt.subplot(1,4,3)
        plt.plot(epochs, times, label='Sec/Epoch')
        plt.title('Time per Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Seconds')
        plt.legend()

        # --- Learning rate ---
        current_lr = getattr(self.optimizer, 'current_learning_rate', getattr(self.optimizer, 'learning_rate', 0.0))
        lr_values = [current_lr] * len(losses)
        plt.subplot(1,4,4)
        plt.plot(epochs, lr_values, label='Learning Rate')
        plt.title('Learning Rate')
        plt.xlabel('Epochs')
        plt.ylabel('LR')
        plt.legend()

        plt.savefig(save_path)
        plt.close()
