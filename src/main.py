import numpy as np
import pickle
from PIL import Image
from utility.mnist import load_mnist


def init_network():
    network = {}
    network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["b1"] = np.array([0.1, 0.2, 0.3])
    network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["b2"] = np.array([0.1, 0.2])
    network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network["b3"] = np.array([0.1, 0.2])
    return network


def sigmoid(x):
    return 1 / (1+np.exp(-x))


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def identity_function(x):
    return x


def forward(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y


# network = init_network()
# x = np.array([1.0, 0.5])
# y = forward(network, x)
# print(y)

# a = np.array([0.3, 2.9, 4.0])
# y = softmax(a)
# print(y)
# print(np.sum(y))


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# (x_train, t_train), (x_test, t_test) = \
#     load_mnist(flatten=True, normalize=False)

# img = x_train[0]
# label = t_train[0]
# print(label)

# print(img.shape)
# img = img.reshape(28, 28)
# print(img.shape)

# img_show(img)


def get_data():
    (_, _), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=True)
    return x_test, t_test


def init_network_weight():
    with open("./src/sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x_test, t_test = get_data()
network = init_network_weight()
batch_size = 100
accuracy_cnt = 0
for i in range(0, len(x_test), batch_size):
    x_batch = x_test[i:i + batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t_test[i: i + batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x_test)))
