import gym
import numpy as np

'''
env = gym.make("CartPole-v1")
for i in range(1000):
    state = env.reset()
    print(state)
    done = False
    total_fitness = 0
    while done == False:
        act = np.random.randint(0,2)#env.action_space.sample()
        print(act)

        state,reward,done,_ = env.step(act)
        total_fitness += reward
        env.render()
        print(act,done,state)

    print("total fitness",total_fitness)
    input("pause")

'''
# Sorting function


# genetic algorithm
def evo_algo():
    mutRate = 0.05
    env = gym.make("CartPole-v0")

    state = env.reset()
    # netInitialization
    inputs = state
    outputs = 1
    net = NN(inputs, outputs)

    ngenes = net.netParameter()
    maxevaluations = 100
    gen_best=[]
    max_fitness=-100

    # setting parameters
    popsize = 20  # population size, offspring number

    genome = np.arange(ngenes * popsize * 2, dtype=np.float64).reshape((popsize * 2, ngenes))
    # allocate and initialize the solution center
    for i in range(popsize * 2):
        genome[i, :] = (np.random.random((1, ngenes)) * 2.0) - 1

    cgen = 0

    # main loop
    while (cgen < maxevaluations):
        cgen = cgen + 1

        fitness = np.zeros(popsize * 2)
        for k in range(popsize * 2):
            done=False
            state=env.reset()
            while done == False:
                act = net.predict(genome[k],state)
                state,reward,done,_ = env.step(act)
                fitness[k] += reward
                #env.render()




        # Sort by fitness
        #print(fitness)
        #input("pause")

        index = np.argsort(fitness)  # maximization
        # (index)
        # print(fitness[index])
        genome = genome[index]
        # print(genome)
        for p in range(popsize):
            for g in range(ngenes):
                if np.random.random() > mutRate:
                    genome[p, g] = genome[p + popsize, g]
                else:
                    genome[p, g] = np.random.random() * 2.0 - 1


        bestfit = fitness[index][-1]

        if bestfit >= max_fitness:
            gen_best = genome[-1]
            max_fitness = bestfit

        print(' Gen %d Bestfit %.2f' % (
            cgen, bestfit))

    print("After ")
    for i in range(100):
        done = False
        state = env.reset()
        fitness=0
        import time
        while done == False:
            act = net.predict(gen_best, state)
            state, reward, done, _ = env.step(act)
            fitness += reward
            time.sleep(0.05)
            env.render()
        print(fitness)
        input("step")

class NN:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.l = 20  # len(inputs) hidden neurons number
        self.li = len(inputs)
        self.wi = np.random.random((self.li, self.l))
        self.wh = np.random.random((self.l, outputs))

    def netParameter(self):
        return self.li * self.l + self.l * self.outputs


    def predict(self, genotype,state):
        l0 = state
        first_layer = self.li * self.l
        self.wi = genotype[:first_layer].reshape(self.li, self.l)
        second_layer = self.l * 1
        self.wh = genotype[first_layer:].reshape(self.l, 1)
        l1 = np.tanh(np.dot(l0, self.wi))
        l1 = np.where(l1 < 0.5, 0, 1)
        l2 = np.tanh(np.dot(l1, self.wh))
        # print(l2)
        l2 = np.where(l2 < 0.5, 0, 1)

        return l2[0]




evo_algo()
