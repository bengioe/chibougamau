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
        r_t = reward(h_t)

        # make sure the reward function can predict
        r_tm1_pred = reward(h_tm1)

        # make sure the decoder can actually decode some past state from the encodings
        x_tml_pred = decoder(h_tm1, l)   # l steps back, from t-1
        x_tmlp1_pred = decoder(h_t, l+1) # l+1 steps back, from t

        # this scenario actually happened, give it a high probability
        prob_correct = oracle(h_tp, a_tp, h_tppm, m)
        # this one probably didnt't, give it low probability
        prob_incorrect = oracle(h_tp, a_tp, h_r, m)

        encoder_decoder_cost = T.sum((x_tml - x_tml_pred)**2) + T.sum((x_tml - x_tmlp1_pred)**2)
        encoder.setCost(encoder_decoder_cost)
        decoder.setCost(encoder_decoder_cost)
        generator.setCost(- T.sum(r_t))
        action_generator.setCost(- T.sum(prob_correct))
        reward.setCost(T.sum(r_tm1_pred - r_tm1))
        oracle.setCost(T.sum(- prob_correct + prob_incorrect))

        self.a = a



class RobotPart:
    def __init__(self, name, nin, nhidden, nout, splitLastOutput=False):
        self.params = []
        shared.bind(name, self.params)
        self.inlayer = HiddenLayer(nin, nhidden, relu)
        self.outlayer = HiddenLayer(nhidden, nout, relu)
        self.splitLastOutput = splitLastOutput
        self.name = name

    def __call__(self, *inputs):
        if len(inputs) > 1:
            inputs = T.concatenate(inputs, axis=-1)
        else:
            inputs = inputs[0]

        h = self.inlayer(inputs)
        o = self.outlayer(h)
        if self.splitLastOutput:
            return o[:-1], o[-1]

    def setCost(self, c):
        shared.attach_cost(self.name, c)



def main(input_size=2, embedding_size=10, hidden_size=10, actions_size=1):
    print "Hi!"
    env = BallWorld()

    print "building graph"
    lr = theano.shared(0.1)

    # 'online' inputs
    h_tm1 = T.vector() # h_{t-1}
    x_t = T.vector()
    z = T.vector()
    r_tm1 = T.scalar()

    # 'learning' inputs
    # some samples of the past and "future"
    x_tml = T.vector()  # x_{t-l}
    l = T.scalar()
    h_tp = T.vector()   # h_{t'}
    a_tp = T.vector()   # a_{t'}
    h_tppm = T.vector() # h_{t'+m}
    h_r = T.vector()    # h_{r}, is some random h from the past
    m = T.scalar()

    encoder = RobotPart("encoder", embedding_size, hidden_size, embedding_size)
    decoder = RobotPart("decoder", embedding_size + 1, hidden_size, embedding_size)
    generator = RobotPart("generator", embedding_size * 2, hidden_size, embedding_size + 1,
                          splitLastOutput=True)
    action_generator = RobotPart("action_generator", embedding_size * 2 + 1, hidden_size, actions_size)
    reward = RobotPart("reward", embedding_size, hidden_size, 1)
    oracle = RobotPart("oracle", embedding_size * 2 + actions_size, hidden_size, 1)

    foobar = FooBar(
        h_tm1, x_t, z, r_tm1, x_tml, l, h_tp, a_tp, h_tppm, h_r, m,
        encoder, decoder, generator, action_generator, reward, oracle)

    updates = shared.computeUpdates(lr, adam())

    print "compiling function"
    learn = theano.function([h_tm1, x_t, z, r_tm1, x_tml, l, h_tp, a_tp, h_tppm, h_r, m],
                            [foobar.a] + shared.get_all_costs(),
                            updates = updates)

    print "learning"



from theano_tools.environments.ballworld import *

if __name__ == "__main__":
    settings = {
        "input_size": 2,
    }

    main(**settings)
