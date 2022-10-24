from math import sqrt

weights_t = [[0.27, 0.81],
             [0.42, 0.70],
             [0.43, 0.21]]

x_input = [0.52, 0.12]
const = 0.1
precision = 2
approx = lambda x: round(x, precision)

# weights_t = [[0.37, 0.61], [0.12, 0.9], [0.53, 0.11]]
# x_input = [0.62, 0.42]
# const = 0.1

print ("Input: {}".format(x_input))
print ("Constant: {}".format(const))
print ("Weights_T -")
for wt in weights_t:
	print (wt)
print ()

d_arr = []
for i, wt in enumerate(weights_t):
	diff = [approx(x-w) for x,w in zip(x_input, wt)]
	print ("d_{}  = SQRT({})".format(i+1,diff))
	diff_sq = [round(d**2, min(precision**2, 6)) for d in diff]
	d_arr.append(sqrt(sum(diff_sq)))
	print ("     = SQRT({})\n     = {}".format(diff_sq, approx(sqrt(sum(diff_sq)))))

index = d_arr.index(min(d_arr))
print ("\nWinner: {}, Index: {}".format(min(d_arr), index+1))

delta_w = [(approx(const * (x-w))) for x,w in zip(x_input, weights_t[index])]
print ("\nDelta_W_{} = {}:".format(index+1, delta_w))

new_w = [approx(d+w) for d,w in zip(delta_w, weights_t[index])]
print ("\nNew_W_{} = {}".format(index+1, new_w))
