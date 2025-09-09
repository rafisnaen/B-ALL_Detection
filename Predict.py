import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load trained model
model = load_model('cnn_leukemia.h5')

# Daftar label kelas (urutannya harus sesuai dengan training)
class_labels = ['Benign', 'Pre-B', 'Pro-B', 'Early Pre-B']


def predict_wbc_image(img_path):
    # Load dan preprocessing gambar
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimension

    # Prediksi
    predictions = model.predict(img_array)[0]

    # Tampilkan gambar
    plt.imshow(img)
    plt.axis('off')
    plt.title("Uploaded Image")
    plt.show()

    # Tampilkan hasil prediksi
    for i in range(len(class_labels)):
        print(f"{class_labels[i]}: {predictions[i] * 100:.2f}%")

    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    print(f"\nPredicted Class: {predicted_class} ({confidence:.2f}%)")

# Contoh penggunaan
predict_wbc_image("Blood cell Cancer Split/test/Benign/Snap_156.jpg")
