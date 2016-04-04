from theano_tools import*






class FooBar:
    def __init__(self,
                 h_tm1, x_t, z, r_tm1, x_tml, l, h_tp, a_tp, h_tppm, h_r, m,
                 encoder, decoder, generator, action_generator, reward, oracle):

        # encode the past
        h_t = encoder(h_tm1, x_t)
        # generate possible future h_{t+k}
        h_tpk, k = generator(h_t, z)
        # take action leading to possible future
        a = action_generator(h_t, h_tpk, k)
        # going to that future gives this reward
        r_t = reward(h_tpk)

        print a.ndim

        # make sure the reward function can predict
        r_tm1_pred = reward(h_tm1)

        # make sure the decoder can actually decode some past state from the encodings
        x_tml_pred = decoder(h_tm1, l)   # l steps back, from t-1
        x_tmlp1_pred = decoder(h_t, l+1) # l+1 steps back, from t
        x_t_pred = decoder(h_t, T.as_tensor_variable(1))

        # this scenario actually happened, give it a high probability
        prob_correct = oracle(h_tp, a_tp, h_tppm, m)
        # this one probably didnt't, give it low probability
        prob_incorrect = oracle(h_tp, a_tp, h_r, m)

        # the probability that the generated action and future h_tpk can happen
        prob_gen_correct = oracle(h_t, a, h_tpk, k)

        encoder_cost = T.sum((x_tml - x_tmlp1_pred)**2)
        encoder.setCost(encoder_cost)
        decoder.setCost(T.sum((x_tml - x_tml_pred)**2) + encoder_cost + T.sum((x_t - x_t_pred)**2))
        generator.setCost(- T.sum(r_t) - T.sum(prob_gen_correct))
        action_generator.setCost(- T.sum(prob_gen_correct))
        reward.setCost(T.sum((r_tm1_pred - r_tm1)**2))
        oracle.setCost(T.sum(- prob_correct + prob_incorrect))

        self.a = a
        self.h_t = h_t



class RobotPart:
    def __init__(self, name, nin, nhidden, nout, outActivation, splitLastOutput=False, splitLastActivation=[]):
        self.params = []
        shared.bind(self.params, name)
        self.inlayer = HiddenLayer(nin, nhidden, relu, name=name)
        self.midlayer = HiddenLayer(nhidden, nhidden, relu, name=name)
        self.outlayer = HiddenLayer(nhidden, nout, outActivation, name=name)
        self.splitLastOutput = splitLastOutput
        self.sla = splitLastActivation
        self.name = name

    def __call__(self, *inputs):
        if len(inputs) > 1:
            print [i.ndim for i in inputs]
            inputs = [i if i.ndim > 0 else T.stack(i) for i in inputs]

            inputs = T.concatenate(inputs, axis=-1)
        else:
            inputs = inputs[0]

        h = self.inlayer(inputs)
        h = self.midlayer(h)
        o = self.outlayer(h)
        if self.splitLastOutput:
            return self.sla[0](o[:-1]), self.sla[1](o[-1])
        return o

    def setCost(self, c):
        shared.attach_cost(self.name, c)



def main(embedding_size=32, hidden_size=32, memory_distance=2):
    print "Hi!"
    env = BallWorld()
    actions = env.actions
    actions_size = nactions = len(actions)
    nfeatures = input_size = env.nfeatures
    print actions, nfeatures

    print "building graph"

    # learning rate
    lr = theano.shared(numpy.float32(0.00002))

    # 'online' inputs
    h_tm1 = T.vector('h_tm1') # h_{t-1}
    x_t = T.vector('x_t')
    z = T.vector('z')
    r_tm1 = T.scalar('r_tm1')

    # 'learning' inputs
    # some samples of the past and "future"
    x_tml = T.vector('x_tml')   # x_{t-l}
    l = T.scalar('l')
    h_tp = T.vector('h_tp')     # h_{t'}
    a_tp = T.vector('a_tp')     # a_{t'}
    h_tppm = T.vector('h_tppm') # h_{t'+m}
    h_r = T.vector('h_r')       # h_{r}, is some random h from the past
    m = T.scalar('m')

    encoder = RobotPart("encoder", embedding_size + input_size, hidden_size, embedding_size,
                        T.tanh)
    decoder = RobotPart("decoder", embedding_size + 1, hidden_size, input_size,
                        lambda x:x)
    generator = RobotPart("generator", embedding_size * 2, hidden_size, embedding_size + 1,
                          lambda x:x,
                          splitLastOutput=True, splitLastActivation=[T.tanh, relu])
    action_generator = RobotPart("action_generator", embedding_size * 2 + 1, hidden_size, actions_size,
                                 lambda x:T.nnet.softmax(x)[0])
    reward = RobotPart("reward", embedding_size, hidden_size, 1, lambda x:x)
    oracle = RobotPart("oracle", embedding_size * 2 + actions_size + 1, hidden_size, 1, T.nnet.sigmoid)

    foobar = FooBar(
        h_tm1, x_t, z, r_tm1, x_tml, l, h_tp, a_tp, h_tppm, h_r, m,
        encoder, decoder, generator, action_generator, reward, oracle)

    updates = shared.computeUpdates(lr, momentum(0.995))
    print foobar.a, foobar.h_t, shared.get_all_costs()
    part_names = shared.get_all_names()
    print "compiling function"
    learn = theano.function([h_tm1, x_t, z, r_tm1, x_tml, l, h_tp, a_tp, h_tppm, h_r, m],
                            [foobar.a, foobar.h_t] + shared.get_all_costs(),
                            updates = updates)
    take_action = theano.function([h_tm1, x_t, z],
                                  [foobar.a, foobar.h_t])


    print "learning"

    epsilon = 0.9
    epoch_costs = []
    memory = []
    env.setupVisual()
    for episode in range(1000):
        totalr = 0
        env.startEpisode(episode % 10 != 0) # do a non random start every 10 episodes
        env_state = env.toRepr() # x_t
        h_t = numpy.float32(numpy.random.random(embedding_size))
        r_t = 0
        action = numpy.float32(numpy.random.random(nactions))
        # just so the memory isn't empty:
        memory.append((env_state, h_t, action, reward))
        nframes = 0
        if episode % 50 == 0:
            memory_distance += 1
        while not env.isEpisodeOver():
            env_state = env.toRepr() # x_t
            env.draw()
            for j in range(6):
                z = numpy.float32(numpy.random.random(embedding_size))
                l = min(numpy.random.randint(1, memory_distance), len(memory))
                x_tml = memory[-l][0]
                tp = numpy.random.randint(0, len(memory))
                m =  numpy.random.randint(1,memory_distance)
                if tp + m >= len(memory):
                    m = len(memory) - 1 - tp
                a_tp = memory[tp][2]
                h_tp = memory[tp][1]
                h_tppm = memory[tp+m][1]
                # random memory
                h_r = memory[numpy.random.randint(0, len(memory))][1]

                # online learn
                _ = learn(h_t, env_state, z, r_t, x_tml, l, h_tp, a_tp, h_tppm, h_r, m)
                costs = _[2:]
                epoch_costs.append(costs)

            action = _[0]
            h_tp1 = _[1]

            if numpy.random.random() < epsilon:
                action = numpy.float32(numpy.random.random(nactions))
            r_t = env.takeAction(actions[action.argmax()])

            totalr += r_t

            memory.append((env_state, h_t, action, r_t))

            h_t = h_tp1
            nframes += 1
            if nframes > 200 or totalr < -10:
                break
        epsilon *= 0.99
        print _[0]
        c = numpy.mean(epoch_costs, axis=0) / nframes * 1000
        print "|".join(["%*s"%(len(i)+1,i) for i in part_names])
        print "|".join(["%*s"%(len(n)+1,numpy.round(i,3)) for i,n in zip(c,part_names)])
        epoch_costs = []


        print episode, "got", totalr, "new epsilon", epsilon

from theano_tools.environments.ballworld import *

if __name__ == "__main__":
    settings = {
        #"input_size": 2,
    }

    main(**settings)
