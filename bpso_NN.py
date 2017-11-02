from numpy.random import normal, random
from math import sqrt, exp
from numpy import array

# for n in range(np):
#     if random() < 0.5:
#         sign1 = 1.0
#     else:
#         sign1 = - 1.0
#     if random() < 0.5:
#         sign2 = 1.0
#     else:
#         sign2 = - 1.0
#     particles[n] = array([sign1 * 5.0 * random(), sign2 * 5.0 * random()])
#     if random() < 0.5:
#         sign3 = 1.0
#     else:
#         sign3 = - 1.0
#     if random() < 0.5:
#         sign4 = 1.0
#     else:
#         sign4 = - 1.0
#     vel[n] = array([[5.0 * sign3 * random(), 5.0 * sign4 * random()]])


def binarise(x):
    position_new = []
    for v in x:
        if v < random():
            result = 0
        else:
            result = 1
        position_new.append(result)
    return position_new


def sigmoid(x):
    lv = []
    for v in x:
        lv.append(1.0 / (1.0 + exp(- 1.0 * v)))
    return lv


def linearize(x):
    lv = []
    for v in x:
        if v <= - 1.0:
            result = 0.0
        elif v <= 1.0:
            result = 0.5 * (v + 1.0)
        else:
            result = 1.0
        lv.append(result)
    return lv


def bpso(fit_fun, x_initial, num_iter):
    output = open('bpso.dat', 'w')
    np = len(x_initial)
    particles = x_initial
    vel = array([[0 for _ in range(len(particles[0]))] for _ in range(np)])
    fitness = array([fit_fun(x) for x in x_initial])
    best_own_position = x_initial
    best_own_fitness = fitness
    best_global_fitness = min(best_own_fitness)
    best_global_position = best_own_fitness[best_own_fitness.tolist().index(best_global_fitness)]

    for iteration in range(num_iter):

        for particle in range(np):
            fitness[particle] = fit_fun(particles[particle])
            if fitness[particle] < best_own_fitness[particle]:
                best_own_fitness[particle] = fitness[particle]
                best_own_position[particle] = particles[particle]
        lowest_iteration_fitness = min(best_own_fitness)
        lowest_iteration_position = best_own_fitness[best_own_fitness.tolist().index(lowest_iteration_fitness)]
        if lowest_iteration_fitness < best_global_fitness:
            best_global_fitness = lowest_iteration_fitness
            best_global_position = lowest_iteration_position

        for particle in range(np):
            vel[particle] = 0.72984 * vel[particle]\
                            + 1.49617 * random() * (best_own_position[particle] - particles[particle])\
                            + 1.49617 * random() * (best_global_position - particles[particle])

            particles[particle] = particles[particle] + vel[particle]

            if particles[particle][0] > 5.0:
                particles[particle][0] = 5.0
            if particles[particle][0] < - 5.0:
                particles[particle][0] = - 5.0
            if particles[particle][1] > 5.0:
                particles[particle][1] = 5.0
            if particles[particle][1] < - 5.0:
                particles[particle][1] = - 5.0
        for n in range(np):
            output.write('{0:f} {1:f}\n'.format(particles[n][0], particles[n][1]))
        output.write('\n')
    output.close()

    print(best_global_fitness)
    print(particles[best_global_position])

if __name__ == '__main__':
    from random import randint, choice

    def make_vector(state, a):
        if a == 0:
            state[a] = randint(2, 25)
            state[a] /= 25.0
        elif a == 1:
            state[1] = 0.0
            while state[a] < state[2]:
                state[a] = choice([30.0, 60.0, 90.0, 120.0, 180.0])
                state[a] /= 180.0
        elif a == 2:
            state[2] = 190.0
            while state[a] > state[1]:
                state[a] = choice([1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 90.0, 120.0, 180.0])
                state[a] /= 180.0
        elif a == 3:
            state[a] = randint(1, 2)
        elif a == 4:
            state[a] = randint(0, 2)
        elif a == 5:
            state[a] = randint(1, 5)
        elif a == 6:
            state[a] = randint(1, 3)
        elif a == 7:
            state[a] = randint(0, 3)
        elif a == 8:
            state[a] = randint(1, 3)
        elif a == 9:
            state[a] = randint(1, 4)
        elif a == 10:
            state[a] = randint(1, 3)
        elif a == 11:
            state[a] = randint(1, 1)
        elif a == 12:
            state[a] = randint(1, 1)

    pop_size = 10

    # Initialization
    def gen_individual():
        state = [0.0 for _ in range(13)]
        state[2] = choice([1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 90.0, 120.0, 180.0])
        state[2] /= 180.0
        for a in range(13):
            make_vector(state, a)
        return state


    def gen_population(n_indiv):
        return [gen_individual() for _ in range(n_indiv)]

    x_init = gen_population(pop_size)

    bpso(NN_lcoe, x_init, 3)
