weights_t = [[1, 2, -1], [0, 1, -2], [-1, 0, 2]]
thresh = [0, -1, 1]
x_input = [0, 1, 0]

print("Input: {}".format(x_input))
print("Weights_T -")
for wt in weights_t:
    print(wt)
print("Threshold: {}".format(thresh))
print()

for i, wt in enumerate(weights_t):
    net = [(x * w) for x, w in zip(x_input, wt)]
    print("Net_{0} = {1} Sum_{0} = {2} Output (T = {3:3}) = {4}".format(
        i + 1,
        net,
        sum(net),
        thresh[i],
        int(sum(net) >= thresh[i])
    ))
