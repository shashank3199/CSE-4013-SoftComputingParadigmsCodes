from math import sqrt

weights_t = [[0.2, 0.6, 0.5, 0.9], [0.8, 0.4, 0.7, 0.3]]
x_input_arr = [[1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1]]
lr = lambda i: ((0.5)**i) * 0.6
n = 2
precision = 4
approx = lambda x: round(x, precision)
radius = 0

print ("Iterations =", n)
print("Input - ")
for x_input in x_input_arr:
    print(x_input)
print("\nWeights_T -")
for wt in weights_t:
    print(wt)
print()

for iteration in range(n):
    print ("------------------------------------------------------------------")
    print ("--------------------------- Iteration: {} -------------------------".format(iteration+1))
    print ("------------------------------------------------------------------")
    for x_input in x_input_arr:

        print ("=============================== x_input = {} ===============================\n".format(x_input))
        d_arr = []
        for i, wt in enumerate(weights_t):
            diff = [approx(x-w) for x,w in zip(x_input, wt)]
            print ("d_{}  = SQRT({})".format(i+1,diff))
            diff_sq = [round(d**2, min(precision**2, 6)) for d in diff]
            d_arr.append(sqrt(sum(diff_sq)))
            print ("     = SQRT({})\n     = {}".format(diff_sq, approx(sqrt(sum(diff_sq)))))

        index = d_arr.index(min(d_arr))
        print ("\nWinner: {}, Index: {}".format(min(d_arr), index+1))

        radius_index_start = max(index-radius, 0)
        radius_index_end = min(index-radius, len(weights_t))
        print ("Radius_Cover: {}".format((radius_index_start, radius_index_end)))

        for index in range(radius_index_start, radius_index_end+1):
            delta_w = [(approx(lr(iteration) * (x-w))) for x,w in zip(x_input, weights_t[index])]
            print ("\nDelta_W_{} = {}:".format(index+1, delta_w))
            new_w = [approx(d+w) for d,w in zip(delta_w, weights_t[index])]
            print ("\nNew_W_{} = {}".format(index+1, new_w))
            weights_t[index] = new_w

        print ("\nNew_W_Matrix (iter: {}, x_pos: {}) - ".format(iteration+1, x_input_arr.index(x_input)+1))
        for wt in weights_t:
            print(wt)
        print ()
