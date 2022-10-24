from math import e
import numpy as np

activation_func = lambda arr: np.array([((1+e**(-x))**-1) for x in arr])
derivation_func = lambda x: x*(1-x)

# hidden_wts_t = np.array([[3,4,1],[6,5,-6]], dtype=np.float)
# output_wts_t = np.array([2,4,-3.93], dtype=np.float)

# x_input = np.array([1,0,1], dtype=np.float).reshape(3,1)
# desired_output = np.array([1], dtype=np.float)
# const = 0.1


# hidden_wts_t = np.array([[0.15, 0.20, 0.35],[0.25, 0.30, 0.35]], dtype=np.float)
# output_wts_t = np.array([[0.40, 0.45, 0.60],[0.50, 0.55, 0.60]], dtype=np.float)

# x_input = np.array([0.05, 0.1, 1], dtype=np.float).reshape(3,1)
# desired_output = np.array([0.01, 0.99], dtype=np.float)
# const = 0.1

hidden_wts_t = np.array([[0.5,0.5,0.5],[0.5,0.5,0.5]], dtype=np.float)
output_wts_t = np.array([[1,1,1]], dtype=np.float)

x_input = np.array([2,0,1], dtype=np.float).reshape(3,1)
desired_output = np.array([0.1], dtype=np.float)
const = 1

print ("Hidden weights =\n {}\n".format(hidden_wts_t))
print ("Output weights =\n {}\n".format(output_wts_t))
print ("Input =\n {}\n".format(x_input))
print ("desired_output =\n {}\n".format(desired_output))


print ("============= Step 1: Calculate FORWARD PASS =============")
net_hidden = hidden_wts_t @ x_input
print ("Net_hidden =\n {}\n".format(net_hidden))

output_hidden = activation_func(net_hidden)
print ("Output_hidden =\n {}\n".format(output_hidden))

input_op_layer = np.append(output_hidden, 1).reshape(3,1)
print ("Input to op_layer =\n {}\n".format(input_op_layer))

net_output = output_wts_t @ input_op_layer
print ("Net_Output =\n {}\n".format(net_output))

output = activation_func(net_output)
print ("Output =\n {}\n".format(output))

print ("\n\n============= Step 2: Calculate f'(net) =============")
f_der_hidden = np.array([derivation_func(x) for x in output_hidden], dtype=np.float)
print ("f'(net)_hidden =\n {}\n".format(f_der_hidden))

f_der_output = np.array([derivation_func(x) for x in output], dtype=np.float)
print ("f'(net)_output =\n {}\n".format(f_der_output))

print ("\n\n============= Step 3: Calculate del_values =============")
del_output = np.multiply((desired_output-output.flatten()), f_der_output.flatten())

sub_weights = output_wts_t.transpose()[:-1].reshape((-1, len(del_output)))
del_hidden = np.multiply(f_der_hidden.flatten(), sub_weights@del_output)
print ("del_hidden =\n {}\n".format(del_hidden))
print ("del_output =\n {}\n".format(del_output))

print ("\n\n============= Step 4: Calculate delta_values =============")
delta_hidden = const * (del_hidden.reshape(len(del_hidden),1) @ x_input.transpose())
print ("delta_hidden =\n {}\n".format(delta_hidden))

delta_output = const * (del_output.reshape(len(del_output),1) @ input_op_layer.transpose())
print ("delta_output =\n {}\n".format(delta_output))

print ("\n\n============= Step 5: Updates =============")
hidden_wts_t += delta_hidden
print ("Updated_Hidden =\n {}\n".format(hidden_wts_t))

output_wts_t += delta_output.reshape(output_wts_t.shape)
print ("Updated_Output =\n {}\n".format(output_wts_t))
