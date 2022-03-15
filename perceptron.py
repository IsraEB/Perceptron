import random


class Neuron:

	def __init__(self, x, y):
		self.a = 0.1

		self.x = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
		self.y = [-1, 1, 1, 1]

		self.x = x
		self.y = y

		row_length = None

		for row in self.x:
			if row_length == None:
				row_length = len(row)
			elif row_length != len(row):
				print(
				    "Error, el número de columnas de entrada no son homogeneas"
				)

			row.append(-1)

		self.w = []

		for i in range(row_length + 1):
			self.w.append(random.uniform(0, 1))

	def train(self, epochs):
		for epoch in range(epochs):

			error_in_epoch = False

			for n_row in range(len(self.x)):
				print(self.x[n_row])
				print("Old: ", self.__str__())

				f_x_net = self.training_predict(self.x[n_row])
				print("f(x_net): ", f_x_net)
				print("S: ", self.y[n_row])

				n_error = self.y[n_row] - f_x_net
				print("Error: ", n_error)

				for n_column in range(len(self.x[n_row])):
					self.w[n_column] = self.w[n_column] + self.a * (
					    n_error) * self.x[n_row][n_column]

				print("New: ", self.__str__() + "\n")

				if n_error != 0:
					error_in_epoch = True

			if error_in_epoch == False:
				print("Se han corregido exitosamente todos los pesos")
				break

	def activation_function(self, x):
		if (x <= 0):
			return -1
		else:
			return 1

	def training_predict(self, x):
		x_net = 0
		for n_column in range(len(x)):
			x_net = x_net + (x[n_column] * self.w[n_column])

		return self.activation_function(x_net)

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


import csv

with open('test.csv') as csvfile:
	rows = csv.reader(csvfile)
	res = list(zip(*rows))

	res = [list(filter(None.__ne__, l)) for l in res]

	res = [list(map(int, i)) for i in res]

	print("res:", res)

	x = [row[:-1] for row in res]
	y = [row[-1] for row in res]

	neuron = Neuron(x, y)

	print("Initial: ", neuron, "\n")

	neuron.train(10000)

	test = [-1, -1]
	print(test, neuron.predict(test))
	test = [-1, 1]
	print(test, neuron.predict(test))
	test = [1, -1]
	print(test, neuron.predict(test))
	test = [1, 1]
	print(test, neuron.predict(test))
