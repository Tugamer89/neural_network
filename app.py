import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageOps
from datetime import datetime

from my_net import Neural_Network
from drawer import Drawer

TARGET_SIZE = (28, 28)

def select_images_from_mnist_png(base_dir, max_per_class=None, shuffle=True):
    X, y = [], []
    for label in sorted(os.listdir(base_dir)):
        label_dir = os.path.join(base_dir, label)
        if not os.path.isdir(label_dir):
            continue
        files = [f for f in os.listdir(label_dir) if f.lower().endswith('.png')]
        if max_per_class and len(files) > max_per_class:
            files = list(np.random.choice(files, max_per_class, replace=False))
        for fn in files:
            try:
                img = Image.open(os.path.join(label_dir, fn)).convert('L')
                arr = np.array(ImageOps.fit(img, (28,28), Image.LANCZOS), dtype=np.float32).flatten()/255.0
                X.append(arr); y.append(int(label))
            except Exception as e:
                print(f"Warning loading {fn}: {e}")
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    if shuffle:
        idx = np.arange(len(X)); np.random.shuffle(idx)
        X, y = X[idx], y[idx]
    return X, y


def main():
    print("Starting app.py — MNIST trainer + drawer integration")
    net = Neural_Network()

    while True:
        print("\nOptions:")
        print(" 1) Load model")
        print(" 2) Train model on MNIST PNG folder")
        print(" 3) Quick test on testing folder")
        print(" 4) Manual test with drawer")
        print(" 5) Save model")
        print(" 6) Plot history")
        print(" 0) Exit")
        choice = input("Choose: ").strip()

        if choice == "1":
            fp = filedialog.askopenfilename(title="Load model", filetypes=[("NumPy compressed", "*.npz")])
            if fp:
                try:
                    net.load(fp)
                    print(f"Loaded model from {fp}")
                except Exception as e:
                    print("Error loading model:", e)

        elif choice == "2":
            train_path = input("Training path (default 'mnist_png/training'): ").strip() or "mnist_png/training"
            if not os.path.isdir(train_path):
                print("Path does not exist:", train_path)
                continue

            print("Loading images...")
            X, y = select_images_from_mnist_png(train_path)
            val_frac = 0.1
            n_val = int(len(X) * val_frac)
            X_val, y_val = (X[:n_val], y[:n_val]) if n_val > 0 else (None, None)
            X_train, y_train = (X[n_val:], y[n_val:]) if n_val > 0 else (X, y)
            print(f"Loaded {len(X_train)} train and {len(X_val) if X_val is not None else 0} val samples")

            epochs = int(input("Epochs [default 10]: ") or 10)
            batch_size = int(input("Batch size [default 128]: ") or 128)
            patience = int(input("Early stopping patience [default 5]: ") or 5)
            target_acc = input("Stop when val acc >= (press enter to disable): ").strip()
            target_acc = float(target_acc) if target_acc else None

            otf_choice = input("Enable on-the-fly augmentation? [y/N]: ").strip().lower()
            otf_augment = (otf_choice == "y")

            if otf_augment:
                print("Training with ON-THE-FLY DATA AUGMENTATION enabled!")
            else:
                print("Training without augmentation.")

            net.train_network(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                epochs=epochs, batch_size=batch_size,
                early_stopping=True, patience=patience,
                target_accuracy=target_acc, tracking_interval=1,
                augment=otf_augment
            )
            print("Training finished.")

        elif choice == "3":
            test_path = input("Testing path (default 'mnist_png/testing'): ").strip() or "mnist_png/testing"
            if not os.path.isdir(test_path):
                print("Path does not exist:", test_path)
                continue
            X_test, y_test = select_images_from_mnist_png(test_path)
            acc, loss = net.test(X_test, y_test)
            print(f"Test acc: {acc:.4f}, loss: {loss:.5f}")

        elif choice == "4":
            print("Opening interactive drawer...")
            def on_draw(image_28x28):
                arr = image_28x28.flatten().reshape(1, -1)
                pred, conf = net.evaluate(arr)
                drawer.set_prediction(pred[0])
                drawer.set_confidence(conf[0])

            root = tk.Tk()
            root.withdraw()
            drawer = Drawer(root, on_draw_callback=on_draw)
            root.deiconify()
            root.mainloop()

        elif choice == "5":
            fname = f"save-{net.history.get('accuracy',[-1])[-1] if net.history.get('accuracy') else 0:.4f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
            fp = filedialog.asksaveasfilename(defaultextension=".npz", initialfile=fname, filetypes=[("NumPy compressed", "*.npz")])
            if fp:
                try:
                    net.save(fp)
                    print(f"Model saved to {fp}")
                except Exception as e:
                    print("Error saving:", e)

        elif choice == "6":
            try:
                net.plot_history()
                print("Plotted history to training_history.png.")
            except Exception as e:
                print("Error plotting history:", e)

        elif choice == "0":
            print("Bye.")
            break

        else:
            print("Unknown option.")

if __name__ == "__main__":
    main()
