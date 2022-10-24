import numpy as np

activation_func = lambda x: (2*(x > 0).astype(np.float))-np.ones_like(x)
const = 1
weights = np.array([[0], [0], [0]], dtype=np.float)
x_input_arr = np.array([[1, 1, 1],
                        [1, -1, 1],
                        [-1, 1, 1],
                        [-1, -1, 1]], dtype=np.float)
################### AND ##################
desired_op = np.array([1, -1, -1, -1], dtype=np.float)
################### OR ##################
# desired_op = np.array([1, 1, 1, -1], dtype=np.float)
n = 4

for iteration in range(n):
    i = iteration % len(x_input_arr)
    x_input = x_input_arr[i]
    print("Input = {}".format(x_input))
    print("Weight = {}".format(weights))
    print("Desired_op_{} = {}".format(i + 1, desired_op[i]))
    # predict_op = activation_func(weights.transpose()@x_input)
    # print("Predict_op_{} = {}".format(i + 1, predict_op))
    # print("Update: {}".format(predict_op != desired_op[i]))
    change = const * desired_op[i] * x_input.reshape(len(weights), 1)
    print ("Change: {}".format(change.flatten()))
    weights += change
    print("New_Weight_Transpose (Iter: {}, X_{}) = {}".format(
        iteration + 1, i + 1, weights.flatten()))
    print("\n\n")
