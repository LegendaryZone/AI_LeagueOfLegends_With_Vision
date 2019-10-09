import numpy as np

def Loader_nasnet():

	data = np.load("image_data.npy")

	x_data, y_data = np.array([data[i, 0] for i in range(len(data))]).reshape(-1,331,331,3), np.array([data[i,1] for i in range(len(data))]).reshape(-1,1)
	print(x_data.shape, y_data.shape)
	print(y_data)
	return x_data, y_data


def Loader_mobile_v2():

	data = np.load("image_data.npy")

	x_data, y_data = np.array([data[i, 0] for i in range(len(data))]).reshape(-1,331,331,3), np.array([data[i,1] for i in range(len(data))]).reshape(-1,1)
	print(x_data.shape, y_data.shape)
	return x_data, y_data


if __name__ == "__main__":
	Loader_nasnet()