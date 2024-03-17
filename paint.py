import tkinter as tk
from PIL import ImageGrab
import numpy as np
import torch

from torch_minst import Net


class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Paint")
        self.canvas = tk.Canvas(self.root, bg="white", width=200, height=200)
        self.canvas.pack()

        self.old_x = None
        self.old_y = None

        self.net = Net()
        print("load model checkpoint", "./minst_model.pt")
        self.net.load_state_dict(torch.load("./minst_model.pt"))
        self.net.eval()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        clear_button = tk.Button(root, text="Clear Canvas", command=self.clear)
        clear_button.pack()
        self.prediction_label = tk.Label(root, text="Prediction: None")
        self.prediction_label.pack()

    def paint(self, event):
        paint_color = "black"
        if self.old_x and self.old_y:
            self.canvas.create_line(
                self.old_x,
                self.old_y,
                event.x,
                event.y,
                width=10,
                fill=paint_color,
                capstyle=tk.ROUND,
                smooth=tk.TRUE,
                splinesteps=36,
            )
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x = None
        self.old_y = None
        self.predict()

    def clear(self):
        self.canvas.delete("all")
        self.prediction_label.config(text=f"Prediction: None")

    def predict(self):
        x = root.winfo_rootx() + self.canvas.winfo_x()
        y = root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        image = ImageGrab.grab().crop((x, y, x1, y1))

        # Resize image to 28x28 and convert to grayscale
        image = image.resize((28, 28)).convert("L")

        array = np.array(image)
        # Invert the image so that drawing becomes white and background black
        array = 255 - array
        # Normalize the array if required by your model
        tensor = (torch.tensor(array, dtype=torch.float32).unsqueeze(0).view(-1, 28 * 28)/ 255.0)

        with torch.no_grad():  # Ensure no gradients are computed during inference
            predict = torch.argmax(self.net.forward(tensor), dim=1)
            predict_value = str(int(predict.item()))
            print("predict", predict_value)
            self.prediction_label.config(text=f"Prediction: {predict_value}")
        return tensor

if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()
