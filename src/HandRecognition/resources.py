import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


from colorama import Fore


DIR_MODELS = '../../models'

def print_color(text: str, color: str, end="\n"):
	print(color + text + Fore.RESET, end=end)



def fit(
	model: models.Model, 
	x, 
	y = None, 
	steps_per_epoch=None,
	validation_data=None,
	validation_steps=None,
	max_epochs=500, 
	batch_size=32, 
	patience=5, 
	learning_rate=0.001, 
	run_on_gpu=True
):
	early_stopping = EarlyStopping(monitor="val_loss", patience=patience, mode="min")
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.2, patience=min(int(patience / 2), 5), min_lr=learning_rate * 0.000001)
	hist = None

	with tf.device('GPU' if run_on_gpu else 'CPU'):
		hist = model.fit(
			x=x,
			y=y,
			steps_per_epoch=steps_per_epoch,
			epochs=max_epochs,
			validation_data=validation_data,
			validation_steps=validation_steps,
			batch_size=batch_size,
			callbacks=[early_stopping, reduce_lr] if patience > 0 else [],
		)

	return hist
	

def compile_and_fit(
	model: models.Model, 
	x, 
	y = None, 
	steps_per_epoch=None,
	validation_data=None,
	validation_steps=None,
	max_epochs=500, 
	batch_size=32, 
	patience=5, 
	learning_rate=0.001, 
	run_on_gpu=True, 
	loss_function=tf.keras.losses.CategoricalCrossentropy(),
	metrics=['accuracy']
):

	model.compile(
		loss=loss_function,
		optimizer=tf.keras.optimizers.legacy.Adam(learning_rate),
		metrics=metrics
	)

	return fit(
		model,
		x,
		y,
		steps_per_epoch,
		validation_data,
		validation_steps,
		max_epochs,
		batch_size,
		patience,
		learning_rate,
		run_on_gpu
	)

	



def evaluate_model(model: models.Sequential, x, y, verbose=False, run_on_gpu=True):
	loss, acc = 0, 0
	with tf.device('GPU' if run_on_gpu else 'CPU'):
		loss, acc = model.evaluate(x, y, verbose=verbose)
	acc = round(acc * 100, 3)

	print_color('Loss: ', Fore.GREEN, end='')
	print(loss)
	print_color('Accuracy: ', Fore.GREEN, end='')
	print(f"{acc}%")
	return loss, acc



def save_model(model: models.Sequential, name: str):
	return model.save(f'{DIR_MODELS}/{name}.h5')



def load_model(name: str, custom_objects=None) -> models.Model:
	if custom_objects is not None: return models.load_model(f'{DIR_MODELS}/{name}.h5')

	return models.load_model(f'{DIR_MODELS}/{name}.h5', custom_objects)


def plot(history, start=0):
	f, ax = plt.subplots(1,2, figsize=(12,3))
	ax[0].plot(history.history['loss'][start:], label='Loss')
	ax[0].plot(history.history['val_loss'][start:], label='Validation loss')
	ax[0].set_title('Loss', size=20)
	ax[0].set_xlabel('Epoch')
	ax[0].set_ylabel('Loss')
	ax[0].legend()

	# Accuracy
	ax[1].plot(history.history['accuracy'][start:], label='Accuracy')
	ax[1].plot(history.history['val_accuracy'][start:], label='Validation accuracy')
	ax[1].set_title('Accuracy', size=20)
	ax[1].set_xlabel('Epoch')
	ax[1].set_ylabel('Accuracy')
	ax[1].legend()
	plt.tight_layout()
	plt.show()



def display_perf(perf: dict, metric_index: int, x_label: str, y_label: str, title: str = None, round_perf=2):
	performances = np.arange(len(perf))

	for key in perf.keys():
		print(f'\n{key.upper()}: {round(perf[key][metric_index], round_perf)}%')

	width = 0.2
	test_val = [v[metric_index] for v in perf.values()]

	if title is not None: plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.bar(performances, test_val, width, label='Test')
	plt.xticks(ticks=performances, labels=perf.keys(), rotation=45)
	_ = plt.legend()



def plot_label_repartition(df_labels: pd.DataFrame, title="Label repartition", headers=None):
	counts = df_labels.value_counts(sort=False).sort_index()
	counts.index = counts.index.map(lambda i: i[0])

	num_total = len(df_labels)

	labels = counts.index
	if (headers is not None):
		labels = headers

	# Get percentages
	percentages = []

	for i in range(len(counts)):
		p = round(counts.values[i] * 100 / num_total, 2)
		percentages.append(p)
		print_color(f"{labels[i]}: ", Fore.CYAN, end='')
		print(f"{counts.values[i]} ({p}%)")

	# Show percentages
	plt.bar(counts.index, percentages)
	plt.xticks(ticks=counts.index, labels=labels, rotation=45)
	plt.title(title)
	plt.xlabel("Labels")
	plt.ylabel("Percentages")
	plt.show()



def predictions_summary(model_name: str, x: np.array, y: np.array, labels, verbose=False, run_on_gpu=True, short=False):
	model: models.Sequential = load_model(model_name)
	predictions = np.array([])
	labels = list(labels)

	with tf.device('GPU' if run_on_gpu else 'CPU'):
		predictions = model.predict(x, verbose=False)


	evaluate_model(model, x, y, verbose=verbose, run_on_gpu=run_on_gpu)

	if not(short):
		_y = np.argmax(y, axis=1)
		_y_pred = np.argmax(predictions, axis=1)
		avg = 'macro'
			
		print_color("\nPrecision: ", Fore.CYAN, end='')
		print(str(
			round(100 * metrics.precision_score(_y, _y_pred, average=avg), 3))
			+ '%'
		)
		print_color("Recall: ", Fore.CYAN, end='')
		print(str(
			round(100 * metrics.recall_score(_y, _y_pred, average=avg), 3))
			+ '%'
		)
		print_color("F1-Score: ", Fore.CYAN, end='')
		print(str(
			round(100 * metrics.f1_score(_y, _y_pred, average=avg), 3))
			+ '%'
		)


	print_color("\nMin: ", Fore.GREEN, end='')
	print(predictions.min())
	print_color("Mean: ", Fore.GREEN, end='')
	print(predictions.mean())
	print_color("Max: ", Fore.GREEN, end='')
	print(predictions.max())

	if short: return

	# if multi-class classifier: reverse one hot encoding into label as number
	y_true = []
	y_pred = []

	if len(y.shape) > 1 and y.shape[1] > 1:
		y_true = [list(y[i]).index(1) for i in range(y.shape[0])]
		y_pred = [list(predictions[i]).index(predictions[i].max()) for i in range(y.shape[0])]

	confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

	print_color("\n--------------------- Confusion Matrix ---------------------", Fore.BLUE)

	for i in range(confusion_matrix.shape[0]):
		print_color(f"\nTrue {labels[i]} ", Fore.LIGHTCYAN_EX, end='')
		print(f"({sum([confusion_matrix[i][j] for j in range(confusion_matrix.shape[1])])})")

		str_pred = ''

		for j in range(confusion_matrix.shape[1]):
			if confusion_matrix[i][j] == 0: continue
			str_pred += f"{confusion_matrix[i][j]} ({labels[j]})"
			if j < confusion_matrix.shape[1] - 1:
				str_pred += Fore.MAGENTA + " ---- " + Fore.RESET

		print_color("Predicted: ", Fore.YELLOW, end='')
		print(str_pred)

	# Total per label

	# True values
	df_labels = pd.DataFrame(y_true)
	counts_true = df_labels.value_counts(sort=False)
	counts_true.index = labels

	df_labels_pred = pd.DataFrame(y_pred)
	counts_pred = df_labels_pred.value_counts(sort=False)

	print_color("\n--------------------- True vs Predicted labels ---------------------\n", Fore.BLUE)

	for i in range(len(counts_true)):
	# If some labels are never predicted
		count_pred = 0
		try:
			count_pred = counts_pred.at[i]
		except Exception:
			pass
		p_true = round(counts_true.values[i] * 100 / len(df_labels), 2)
		p_pred = round(count_pred * 100 / len(df_labels_pred), 2)
		print_color(f"{labels[labels.index(counts_true.index[i])]}: ", Fore.CYAN, end='')
		print(f"{counts_true.values[i]} ({p_true}%)", end='')
		print_color(" ---- ", Fore.MAGENTA, end='')
		print(f"{count_pred} ({p_pred}%)")

