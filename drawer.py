import numpy as np
import tkinter as tk
from PIL import Image
from scipy.ndimage import gaussian_filter

def draw_brush(image, x, y, radius=1.5, intensity=1.0):
    h, w = image.shape
    yy, xx = np.ogrid[:h, :w]
    mask = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * radius**2))
    image[:] = np.maximum(image, mask * intensity)

class Drawer:
    def __init__(self, root, on_draw_callback=None):
        self.root = root
        self.on_draw_callback = on_draw_callback

        self.canvas_size = 280
        self.pixel_size = 10
        self.image = np.zeros((28, 28), dtype=np.float32)

        # Canvas
        self.canvas = tk.Canvas(root, bg='black', width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack()

        # Bind mouse
        self.canvas.bind("<B1-Motion>", self.paint_white)
        self.canvas.bind("<B3-Motion>", self.paint_black)

        # Prediction label
        self.pred_label = tk.Label(root, text="Draw a digit", fg="black", font=("Arial", 16))
        self.pred_label.pack(pady=8)

        # Confidence label
        self.conf_label = tk.Label(root, text="", fg="black", font=("Arial", 16))
        self.conf_label.pack(pady=4)

        # Reset button
        self.reset_button = tk.Button(root, text="Reset", command=self.reset_canvas, bg="gray20", fg="white")
        self.reset_button.pack(pady=4)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _paint(self, event, value, color):
        x, y = event.x, event.y
        x1, y1 = (x // self.pixel_size) * self.pixel_size, (y // self.pixel_size) * self.pixel_size
        row, col = y1 // self.pixel_size, x1 // self.pixel_size
        if 0 <= row < 28 and 0 <= col < 28:
            self.image[row, col] = value
            self.canvas.create_rectangle(x1, y1, x1 + self.pixel_size, y1 + self.pixel_size, fill=color, outline=color)
            if self.on_draw_callback:
                try:
                    self.on_draw_callback(self.image)
                except Exception as e:
                    print("Callback error:", e)

    def paint_white(self, event):
        x, y = event.x / self.pixel_size, event.y / self.pixel_size
        draw_brush(self.image, x, y, radius=0.7, intensity=1.0)
        self.update_canvas()
        self._paint(event, 1.0, "white")

    def paint_black(self, event):
        self._paint(event, 0.0, "black")

    def stylize_mnist_like(self):
        img = gaussian_filter(self.image, sigma=0.5)

        noise = np.random.normal(0, 0.05, img.shape)
        img = np.clip(img + noise, 0, 1)

        img = img ** 1.3
        self.image = img

    def get_image_pil(self):
        img = self.image.copy()
        img = gaussian_filter(img, sigma=0.5)
        img = np.clip(img, 0, 1)
        return Image.fromarray((img * 255).astype(np.uint8))

    def update_canvas(self):
        self.canvas.delete("all")
        for r in range(28):
            for c in range(28):
                val = int(self.image[r, c] * 255)
                if val > 0:
                    color = f"#{val:02x}{val:02x}{val:02x}"
                    x1, y1 = c * self.pixel_size, r * self.pixel_size
                    self.canvas.create_rectangle(x1, y1, x1 + self.pixel_size, y1 + self.pixel_size,
                                                fill=color, outline=color)

    def reset_canvas(self):
        self.canvas.delete("all")
        self.image.fill(0)
        self.pred_label.config(text="Draw a digit")
        self.conf_label.config(text="")

    def set_prediction(self, pred):
        self.pred_label.config(text=f"Prediction: {pred}")
    
    def set_confidence(self, conf):
        self.conf_label.config(text=f"Confidence: {conf*100:.2f}%")

    def on_closing(self):
        self.root.destroy()
