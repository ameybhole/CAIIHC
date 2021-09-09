import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

from src.data_load import binarize_labels


def write_result_image(image,pred_image, resultText, image_width):

	vertical_white = np.full([image_width, 40, 3], 255, dtype=np.uint8)
	horizontal_white = np.full([40, image_width*2 + 120, 3], 255, dtype=np.uint8)
	result_image = np.concatenate((vertical_white, image), axis=1)
	result_image = np.concatenate((result_image, vertical_white), axis=1)
	result_image = np.concatenate((result_image, pred_image), axis=1)
	result_image = np.concatenate((result_image, vertical_white), axis=1)
	result_image = np.concatenate((horizontal_white, result_image), axis=0)

	imageHeight, imageWidth, sceneNumChannels = result_image.shape
	SCALAR_Black = (0.0, 0.0, 0.0)

	fontFace = cv2.FONT_HERSHEY_TRIPLEX

	fontScale = 0.5
	fontThickness = 1

	fontThickness = int(fontThickness)

	upperLeftTextOriginX = int(imageWidth * 0.05)
	upperLeftTextOriginY = int(imageHeight * 0.05)

	textSize, baseline = cv2.getTextSize(resultText, fontFace, fontScale, fontThickness)
	textSizeWidth, textSizeHeight = textSize

	lowerLeftTextOriginX = upperLeftTextOriginX
	lowerLeftTextOriginY = upperLeftTextOriginY + textSizeHeight

	cv2.putText(result_image, resultText, (lowerLeftTextOriginX, lowerLeftTextOriginY), fontFace, fontScale, SCALAR_Black, fontThickness)
	return result_image

def printWrongPredictions(pred,X_test, y_test):
	array = []

	# Print predicted vs. Actual
	for i in range(len(pred)):
		biggest_value_index = pred[i].argmax(axis=0)
		value = pred[i][biggest_value_index]
		y_classes = pred[i] >= value
		y_classes = y_classes.astype(int)
		array.append(y_classes)
	predicted_list = np.asarray(array, dtype="int32")

	# flattened_list = np.asarray([y for x in labels for y in x], dtype="str")
    # lb = preprocessing.LabelBinarizer().fit()
	# y_test = lb.inverse_transform(y_test)
	# pred = lb.inverse_transform(pred)
	# print(y_test)

	for i in range(len(y_test)):
		if y_test[i] != pred[i]:
			print("Predicted:", pred[i], " Actual: ", y_test[i])



def plot_data_graph(hist, num_epochs):

	train_loss = hist.history['loss']
	val_loss = hist.history['val_loss']
	train_acc = hist.history['categorical_accuracy']
	val_acc = hist.history['val_categorical_accuracy']
	xc = range(num_epochs)

	plt.figure(figsize=(12, 10))
	plt.plot(xc, train_loss)
	plt.plot(xc, val_loss)
	plt.xlabel('num of Epochs')
	plt.ylabel('loss')
	plt.title('train_loss vs val_loss')
	plt.grid(True)
	plt.legend(['train', 'val'])
	plt.show()

	plt.figure(figsize=(12, 10))
	plt.plot(xc, train_acc)
	plt.plot(xc, val_acc)
	plt.xlabel('num of Epochs')
	plt.ylabel('loss')
	plt.title('train_acc vs val_acc')
	plt.grid(True)
	plt.legend(['train', 'val'])
	plt.show()

def model_predictions(model, X_test):

	pred = model.predict(X_test)

	return pred

