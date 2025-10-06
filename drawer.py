import numpy as np
import tkinter as tk
from PIL import Image

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

        # Label
        self.pred_label = tk.Label(root, text="Draw a digit", fg="white", bg="black", font=("Arial", 16))
        self.pred_label.pack(pady=8)

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
        self._paint(event, 1.0, "white")

    def paint_black(self, event):
        self._paint(event, 0.0, "black")

    def get_image_pil(self):
        return Image.fromarray((self.image * 255).astype(np.uint8))

    def reset_canvas(self):
        self.canvas.delete("all")
        self.image.fill(0)
        self.pred_label.config(text="Draw a digit")

    def set_prediction(self, pred):
        self.pred_label.config(text=f"Prediction: {pred}")

    def on_closing(self):
        self.root.destroy()
