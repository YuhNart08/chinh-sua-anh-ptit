import cv2
import numpy as np
from tkinter import Tk, Label, Button, Scale, HORIZONTAL, filedialog, Frame
from PIL import Image, ImageTk
import os

# ------------------ HÀM XỬ LÝ ẢNH ------------------
def adjust_gamma(image, gamma=1.0):
    """Điều chỉnh gamma cho ảnh."""
    img_float = image / 255.0
    corrected = np.power(img_float, gamma)
    corrected = np.uint8(corrected * 255)
    return corrected

def sharpen_image(image):
    """Làm rõ ảnh bằng kernel sharpen."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def denoise_median(image, ksize=3):
    """Giảm nhiễu bằng lọc median."""
    denoised = cv2.medianBlur(image, ksize)
    return denoised

def denoise_gaussian(image, ksize=(5,5), sigma=0):
    """Giảm nhiễu bằng lọc Gaussian."""
    denoised = cv2.GaussianBlur(image, ksize, sigma)
    return denoised

def denoise_bilateral(image, d=9, sigmaColor=75, sigmaSpace=75):
    """Giảm nhiễu bằng lọc bilateral (giữ cạnh)."""
    denoised = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
    return denoised

def equalize_histogram(image):
    """Cân bằng histogram cho ảnh màu bằng cách chuyển sang YCrCb và cân bằng kênh Y."""
    img_y_cr_cb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)
    y_eq = cv2.equalizeHist(y)
    img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
    img_eq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCrCb2RGB)
    return img_eq

def equalize_clahe(image, clipLimit=2.0, tileGridSize=(8,8)):
    """CLAHE (adaptive histogram equalization) cho ảnh màu."""
    img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    l_eq = clahe.apply(l)
    img_lab_eq = cv2.merge((l_eq, a, b))
    img_eq = cv2.cvtColor(img_lab_eq, cv2.COLOR_LAB2RGB)
    return img_eq

# ------------------ ỨNG DỤNG GUI ------------------
class PhotoEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chương trình chỉnh sửa ảnh - Nhóm 5")
        self.root.geometry("900x600")
        self.root.configure(bg="#e8e8e8")

        self.image = None
        self.processed = None

        # --- Frame chứa các nút chức năng ---
        button_frame = Frame(root, bg="#dcdcdc", pady=10)
        button_frame.pack(side="top", fill="x")

        Button(button_frame, text="Chọn ảnh", command=self.load_image, width=15).pack(side="left", padx=10)
        Button(button_frame, text="Làm rõ ảnh", command=self.sharpen, width=15).pack(side="left", padx=10)
        Button(button_frame, text="Giảm nhiễu", command=self.show_denoise_options, width=15).pack(side="left", padx=10)
        Button(button_frame, text="Cân bằng histogram", command=self.show_hist_options, width=18).pack(side="left", padx=10)
        Button(button_frame, text="Điều chỉnh Gamma", command=self.enable_gamma_slider, width=18).pack(side="left", padx=10)
        Button(button_frame, text="Lưu ảnh", command=self.save_image, width=15).pack(side="left", padx=10)
        Button(button_frame, text="Thoát", command=root.quit, width=10).pack(side="right", padx=10)

        # --- Hiển thị ảnh ---
        self.original_label = Label(root, text="Ảnh gốc chưa chọn", bg="#e8e8e8")
        self.original_label.pack(side="left", expand=True, padx=10, pady=10)

        self.result_label = Label(root, text="Ảnh kết quả", bg="#e8e8e8")
        self.result_label.pack(side="right", expand=True, padx=10, pady=10)

        # --- Thanh trượt gamma ---
        self.gamma_scale = None

    # ------------------ CÁC CHỨC NĂNG ------------------
    def load_image(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff")]
        )
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.image = img
        self.processed = img.copy()

        self.display_image(self.image, self.original_label)
        self.display_image(self.processed, self.result_label)

    def sharpen(self):
        if self.image is None:
            return
        self.processed = sharpen_image(self.image)
        self.display_image(self.processed, self.result_label)

    def show_denoise_options(self):
        if self.image is None:
            return
        # Simple cycle through denoise methods for minimal UI changes
        # If processed is same as original or last method unknown, apply median
        last = getattr(self, '_last_denoise', 'none')
        if last == 'none':
            self.processed = denoise_median(self.image, ksize=3)
            self._last_denoise = 'median'
        elif last == 'median':
            self.processed = denoise_gaussian(self.image, ksize=(5,5), sigma=0)
            self._last_denoise = 'gaussian'
        elif last == 'gaussian':
            self.processed = denoise_bilateral(self.image)
            self._last_denoise = 'bilateral'
        else:
            self.processed = self.image.copy()
            self._last_denoise = 'none'

        self.display_image(self.processed, self.result_label)

    def show_hist_options(self):
        if self.image is None:
            return
        # Cycle between none -> global equalize -> CLAHE
        last = getattr(self, '_last_hist', 'none')
        if last == 'none':
            self.processed = equalize_histogram(self.image)
            self._last_hist = 'global'
        elif last == 'global':
            self.processed = equalize_clahe(self.image)
            self._last_hist = 'clahe'
        else:
            self.processed = self.image.copy()
            self._last_hist = 'none'

        self.display_image(self.processed, self.result_label)

    def enable_gamma_slider(self):
        if self.image is None:
            return

        # Nếu đã có thanh trượt thì không tạo lại
        if self.gamma_scale is not None:
            return

        self.gamma_scale = Scale(self.root, from_=0.1, to=3.0, resolution=0.1,
                                 orient=HORIZONTAL, label="Điều chỉnh Gamma",
                                 length=400, bg="#e8e8e8", command=self.update_gamma)
        self.gamma_scale.set(1.0)
        self.gamma_scale.pack(pady=10)

    def update_gamma(self, val):
        gamma = float(val)
        self.processed = adjust_gamma(self.image, gamma)
        self.display_image(self.processed, self.result_label)

    def save_image(self):
        if self.processed is None:
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png")],
            title="Lưu ảnh kết quả"
        )
        if path:
            cv2.imwrite(path, cv2.cvtColor(self.processed, cv2.COLOR_RGB2BGR))
            print(f"💾 Đã lưu ảnh tại: {path}")

    def display_image(self, img, label_widget):
        """Hiển thị ảnh trong label."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w, _ = img.shape
        max_w, max_h = 400, 400
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        img_tk = ImageTk.PhotoImage(Image.fromarray(resized))
        label_widget.configure(image=img_tk)
        label_widget.image = img_tk


# ------------------ CHẠY CHƯƠNG TRÌNH ------------------
if __name__ == "__main__":
    root = Tk()
    app = PhotoEditorApp(root)
    root.mainloop()
