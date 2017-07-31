import numpy as np

nb_ep = 5000
val = []
for i in range(nb_ep):
    print(i)
    moves = 0
    state = (0,0)

    while True:
        action = np.random.randint(0, 5)

        if action == 0:
            state = tuple(np.subtract(state, (1, 0)))
        elif action == 1:
            state = tuple(np.add(state, (1, 0)))
        elif action == 2:
            state = tuple(np.subtract(state, (0, 1)))
        elif action == 3:
            state = tuple(np.add(state, (0, 1)))

        #do not move beyond the boundaries
        if state[0] < 0:
            state = (0, state[1])
        elif state[0] >= 10:
            state = (10-1, state[1])
        elif state[1] < 0:
            state = (state[0], 0)
        elif state[1] >= 10:
            state = (state[0], 10-1)

        moves += 1

        if state == (5, 0):
            val += [moves]
            break

val = np.array(val)
print(np.mean(val))
