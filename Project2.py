import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models, metrics
import sklearn.metrics
from os import path
import glob
import sys

# def crop_image(image, width, height):
#     if(image.shape[0] > height or image.shape[1] > width):
#         center = (image.shape[0]//2, image.shape[1]//2)
#         y = center[0] - (height//2)
#         x = center[1] - (width//2)
#         return image[y:y + height, x:x + width]
#     return image

def loadImgs(face_classifier, img_dir, img_width, img_height):
    jpg_path = path.join(img_dir, "*.jpg")
    imgs = []
    for full_img_path in glob.glob(jpg_path):
        image = cv2.imread(full_img_path)
        faces = face_classifier.detectMultiScale(image,
                    scaleFactor=1.1, minNeighbors=5, 
                    minSize=(img_width, img_height))
        if(type(faces) == list):
            (x, y, w, h) = faces[0]
            image = image[y:y+h, x:x+w]
        # cropped_img = crop_image(image, img_width, img_height)
        resized_img = cv2.resize(image, (img_width, img_height))
        if(resized_img.shape != (img_width, img_height, 3)):
            print(full_img_path)
        # gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        normalized_img = resized_img.astype("float")/255.0
        np_img = np.asarray(normalized_img)
        # tf_img = tf.convert_to_tensor(normalized_img, dtype=tf.float32)
        imgs.append(np_img)
    return imgs

def loadDirsLabel(face_classifier, img_dir, img_width, img_height):
    path_pain = path.join(img_dir, "Pain")
    all_images = loadImgs(face_classifier, path_pain, img_width, img_height)
    pain_img_num = len(all_images)
    labels = [1.0] * pain_img_num
    path_nopain = path.join(img_dir, "No_pain")
    all_images.extend(loadImgs(face_classifier, path_pain, img_width, img_height))
    nopain_img_num = len(all_images) - pain_img_num
    labels.extend([0.0] * nopain_img_num)
    np_images = np.asarray(all_images)
    np_labels = np.asarray(labels)
    rng_state = np.random.get_state()
    np.random.shuffle(np_images)
    np.random.set_state(rng_state)
    np.random.shuffle(np_labels)
    return np_images, np_labels

def trainCNN(path_facial_classifier, path_all_data, img_width, img_height):
    face_cascade = cv2.CascadeClassifier(path_facial_classifier)
    train_path = path.join(path_all_data, "Training")
    all_train_imgs, train_labels = loadDirsLabel(face_cascade, train_path, img_width, img_height)
    test_path = path.join(path_all_data, "Testing")
    all_test_imgs, test_labels = loadDirsLabel(face_cascade, test_path, img_width, img_height)
    
    model = models.Sequential([
        layers.InputLayer(input_shape=(img_width, img_height, 3), batch_size=1),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='softmax')
    ])

    model.compile(optimizer='Adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    history = model.fit(x=all_train_imgs, y=train_labels, batch_size=1,\
                        max_queue_size=2, epochs=5)
    prediction = model.predict(all_test_imgs, batch_size=1, max_queue_size=2)
    TP = float(tf.math.count_nonzero(prediction * test_labels).numpy())
    TN = float(tf.math.count_nonzero((prediction - 1) * (test_labels - 1)).numpy())
    FP = float(tf.math.count_nonzero(prediction * (test_labels - 1)).numpy())
    FN = float(tf.math.count_nonzero((prediction - 1) * test_labels).numpy())
    # model_accuracy = metrics.Accuracy()
    # model_accuracy.update_state(test_labels, prediction)
    # model_precision = metrics.Precision()
    # model_precision.update_state(test_labels, prediction)
    # model_recall = metrics.Recall()
    # model_recall.update_state(test_labels, prediction)
    # model_F1 = sklearn.metrics.f1_score(test_labels, prediction)
    model_accuracy = float((TP+TN)/(TP+FN+TN+FP))
    print("Classification Accuracy: " + str(model_accuracy))
    model_precision = float(TP/(TP+FP))
    print("Model Precision: " + str(model_precision))
    model_recall = float(TP/(TP+FN))
    print("Model Recall: " + str(model_recall))
    model_F1 = float((2*model_precision*model_recall)/(model_precision+model_recall))
    print("Model F1 Scores: " + str(model_F1))
    confusion_matrix = tf.math.confusion_matrix(test_labels, prediction)
    print("Confusion Matrix: " + str(confusion_matrix))

if __name__ == "__main__":
    if len(sys.argv) != 4 or not sys.argv[1].isdigit()\
                            or not sys.argv[2].isdigit()\
                            or not path.exists(sys.argv[3]):
        raise Exception("Requires args for width (int), height (int), and the path to the training/testing data")
    trainCNN("haarcascade_frontalface_default.xml", sys.argv[3], int(sys.argv[1]), int(sys.argv[2]))