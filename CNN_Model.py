from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
import config

batch_size = 32
num_classes = 4  # Jumlah kelas: Benign, Early Pre-B, Pre-B, Pro-B

# Preprocessing: Image Augmentation untuk Train Data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalisasi gambar
    rotation_range=10,  # Rotasi gambar acak hingga 10 derajat
    width_shift_range=0.1,  # Pergeseran horizontal hingga 10% dari lebar gambar
    height_shift_range=0.1,  # Pergeseran vertikal hingga 10% dari tinggi gambar
    shear_range=0.1,  # Kemiringan gambar hingga 10%
    zoom_range=0.15,  # Zoom hingga 15%
    horizontal_flip=True,  # Pembalikan horizontal gambar
    fill_mode='nearest'  # Isi area kosong dengan piksel terdekat
)

val_test_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalisasi gambar
)
# Hanya normalisasi untuk data validasi dan uji

train_generator = train_datagen.flow_from_directory(
    config.TRAIN_DIR,
    target_size=(224, 224),  # Mengubah resolusi gambar menjadi 224x224
    batch_size=batch_size,
    class_mode='categorical'  # Multi-class classification
)

val_generator = val_test_datagen.flow_from_directory(
    config.VAL_DIR,
    target_size=(224, 224),  # Mengubah resolusi gambar menjadi 224x224
    batch_size=batch_size,
    class_mode='categorical'  # Multi-class classification
)

# Build CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),

    layers.Dense(num_classes, activation='softmax')  # Output dengan 4 kelas dan softmax
])

# Mengatur learning rate dan optimizer
optimizer = Adam(learning_rate=0.001)

# Menyusun model dengan loss function dan optimizer
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',  # Loss function untuk multi-class classification
    metrics=['accuracy']
)

model.summary()

# Model Train
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30  # Banyak iterasi epoch
)

# Plot training history
def plot_training_history(history):
    epochs = range(1, len(history.history['accuracy']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['accuracy'], label='Train Accuracy', marker='o')
    plt.plot(epochs, history.history['val_accuracy'], label='Val Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['loss'], label='Train Loss', marker='o')
    plt.plot(epochs, history.history['val_loss'], label='Val Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()


# Panggil fungsi untuk plot hasil training
plot_training_history(history)

# Save the trained model
model.save('cnn_leukemia.h5')
print("Model saved to cnn_leukemia.h5")
