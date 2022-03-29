import pickle
import random
import sys

WEIGHTS_FILENAME = "weights"


def load_weights(neuron):
	file = open(WEIGHTS_FILENAME, 'rb')
	neuron.w = pickle.load(file)
	file.close()


def save_weights(neuron):
	file = open(WEIGHTS_FILENAME, 'wb')
	pickle.dump(neuron.w, file)
	file.close()


class Neuron:
	def __init__(self, x, y):
		self.a = 0.1

		self.x = x
		self.y = y

		row_length = None

		for row in self.x:
			if row_length == None:
				row_length = len(row)
			elif row_length != len(row):
				print(
				    "Error, el número de columnas de entrada no son homogéneas"
				)

		self.w = []

		for i in range(row_length + 1):
			self.w.append(random.uniform(0, 1))

	def train(self, epochs):
		for epoch in range(1, epochs + 1):

			print("Epoch: ", epoch, "\n")

			error_in_epoch = False

			for n_row in range(len(self.x)):

				f_x_net = self.predict(self.x[n_row])
				n_error = self.y[n_row] - f_x_net

				self.x[n_row].append(-1)

				print(self.x[n_row])
				print("Old: ", self.__str__())
				print("f(x_net): ", f_x_net)
				print("S: ", self.y[n_row])
				print("Error: ", n_error)

				for n_column in range(len(self.x[n_row])):
					self.w[n_column] = self.w[n_column] + self.a * (
					    n_error) * self.x[n_row][n_column]
				save_weights(self)

				self.x[n_row].pop()

				print("New: ", self.__str__() + "\n")

				if n_error != 0:
					error_in_epoch = True

			if error_in_epoch == False:
				print(
				    "Se han corregido exitosamente todos los pesos en la generación ",
				    epoch, "\n")
				return

			self.print_dataset_predictions()

		print(
		    "Se ha alcanzado el número máximo de épocas y no se han encontrado unos pesos correctos"
		)

	def activation_function(self, x):
		if (x <= 0):
			return 0
		else:
			return 1

	def predict(self, x):
		x.append(-1)

		x_net = 0
		for n_column in range(len(x)):
			x_net = x_net + (x[n_column] * self.w[n_column])

		x.pop()

		return self.activation_function(x_net)

	def __str__(self) -> str:
		string = ""

		for w in self.w:
			string += str(w) + "\t"

		return string

	def print_dataset_predictions(self):
		for n_row in range(len(self.x)):
			f_x_net = self.predict(self.x[n_row])
			print(self.x[n_row], ": ", f_x_net)

		print()


def predict_user_data(neuron):
	print("Hora de hacer predicciones!\n")

	while (True):
		x = []
		for i in range(len(neuron.x[0])):
			print("Digite el valor de x" + str(i + 1) + ": ", end="")
			x.append(int(input()))

		print("El resultado de la predicción es: ", neuron.predict(x), "\n")

	pass


if len(sys.argv) >= 2:

	import csv

	with open(sys.argv[1] + '.csv', newline='') as f:
		reader = csv.reader(f)
		res = list(reader)

		res = [list(map(int, i)) for i in res]

		x = [row[:-1] for row in res]
		y = [row[-1] for row in res]

		neuron = Neuron(x, y)

		if len(sys.argv) != 3 or sys.argv[2] != 'newWeights':
			try:
				load_weights(neuron)
			except:
				pass

		print("Initial: ", neuron, "\n")

		neuron.train(10000)

		neuron.print_dataset_predictions()

		save_weights(neuron)

		predict_user_data(neuron)

else:
	print(
	    "De como argumento solo el nombre de un archivo csv con los datos de entrenamiento"
	)
