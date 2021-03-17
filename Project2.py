import cv2
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from os import path
import glob
import sys

def crop_image(image, width, height):
    if(image.shape[0] > height or image.shape[1] > width):
        center = (image.shape[0]//2, image.shape[1]//2)
        y = center[0] - (height//2)
        x = center[1] - (width//2)
        return image[y:y + height, x:x + width]
    return image

def loadImgs(face_classifier, img_dir, img_width, img_height):
    jpg_path = path.join(img_dir, "*.jpg")
    imgs = []
    for full_img_path in glob.glob(jpg_path):
        image = cv2.imread(full_img_path)
        faces = face_classifier.detectMultiScale(image,
                    scaleFactor=1.1, minNeighbors=5, 
                    minSize=(img_width, img_height))
        # print(faces)
        if(type(faces) == list):
            (x, y, w, h) = faces[0]
            image = image[y:y+h, x:x+w]
        cropped_img = crop_image(image, img_width, img_height)
        if(cropped_img.shape != (img_width, img_height, 3)):
            print(cropped_img.shape)
        normalized_img = cropped_img/255.0
        imgs.append(normalized_img)
    return imgs

def loadDirsLabel(face_classifier, img_dir, img_width, img_height):
    path_pain = path.join(img_dir, "Pain")
    all_images = loadImgs(face_classifier, path_pain, img_width, img_height)
    pain_img_num = len(all_images)
    labels = [True] * pain_img_num
    path_nopain = path.join(img_dir, "No_pain")
    all_images.append(loadImgs(face_classifier, path_pain, img_width, img_height))
    nopain_img_num = len(all_images) - pain_img_num
    labels.append([False] * nopain_img_num)
    return all_images, labels

def trainCNN(path_facial_classifier, path_all_data, img_width, img_height):
    face_cascade = cv2.CascadeClassifier(path_facial_classifier)
    train_path = path.join(path_all_data, "Training")
    all_train_imgs, train_labels = loadDirsLabel(face_cascade, train_path, img_width, img_height)
    test_path = path.join(path_all_data, "Testing")
    all_test_imgs, test_labels = loadDirsLabel(face_cascade, test_path, img_width, img_height)
    
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    history = model.fit(all_train_imgs, train_labels, epochs=5)
    prediction = model.prediction(all_test_imgs)
    # test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    model_accuracy = tf.metrics.accuracy(test_labels, prediction)
    confusion_matrix = tf.math.confusion_matrix(test_labels, prediction)
    model_precision = tf.keras.metrics.Precision(test_labels, prediction)
    model_recall = tf.keras.metrics.Recall(test_labels, prediction)
    model_F1 = sklearn.metrics.f1_score(test_labels, prediction)

    print("Classification Accuracy: " + str(model_accuracy))
    print("Confusion Matrix: " + str(confusion_matrix))
    print("Model Precision: " + str(model_precision))
    print("Model Recall: " + str(model_recall))
    primt("Model F1 Scores: " + str(model_F1))

if __name__ == "__main__":
    if len(sys.argv) != 4 or not sys.argv[1].isdigit()\
                            or not sys.argv[2].isdigit()\
                            or not path.exists(sys.argv[3]):
        raise Exception("Requires args for width (int), height (int), and the path to the training/testing data")
    trainCNN("haarcascade_frontalface_default.xml", sys.argv[3], int(sys.argv[1]), int(sys.argv[2]))