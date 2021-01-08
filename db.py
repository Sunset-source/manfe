from scipy.stats import levy_stable

from prelude import *


def gen_batch(hps):
    y_batch = np.zeros([TIMES_SLOTS_PER_BATCH, 2 * NUM_ANT, 1])
    h_batch = np.zeros([TIMES_SLOTS_PER_BATCH, 2 * NUM_ANT, 2 * NUM_ANT])
    s_batch = np.random.rand(TIMES_SLOTS_PER_BATCH, 2 * NUM_ANT, 1)
    s_batch = np.where(s_batch < 0.5, -1 / np.sqrt(2), 1 / np.sqrt(2))
    w_batch = np.zeros([TIMES_SLOTS_PER_BATCH, 2 * NUM_ANT, 1])

    p = 10 ** (hps.snr / 10)

    for i in range(PACKETS_PER_BATCH):
        h = np.sqrt(p / NUM_ANT) * complex_channel()
        for j in range(TIME_SLOTS_PER_PACKET):
            t = i * TIME_SLOTS_PER_PACKET + j

            if NOISE_TYPE == "MIXGAUSS":
                w = exampl_mixture_gauss(size=[2 * NUM_ANT, 1])
            elif NOISE_TYPE == "NAKA":
                w = nakagami_m(m=hps.m, size=[2 * NUM_ANT, 1])
            else:
                w = levy_stable.rvs(alpha=hps.alpha, beta=hps.beta, loc=hps.mu, scale=hps.sigma, size=[2 * NUM_ANT, 1])

            s = s_batch[t, :, :]
            s = s.reshape([2 * NUM_ANT, 1])

            y = h @ s + w

            y_batch[t:t + 1, :, :] = y
            h_batch[t:t + 1, :, :] = h
            w_batch[t:t + 1, :, :] = w

    return y_batch, h_batch, s_batch, w_batch


def levy_var_batch(hps):
    return levy_stable.rvs(
        alpha=hps.alpha,
        beta=hps.beta,
        loc=hps.mu,
        scale=hps.sigma,
        size=[TIMES_SLOTS_PER_BATCH, 2 * NUM_ANT, 1]
    )


def nakagami_var_batch(hps):
    return nakagami_m(m=hps.m, size=[TIMES_SLOTS_PER_BATCH, 2 * NUM_ANT, 1])


def gen_noise_from_hps(hps):
    if NOISE_TYPE == "MIXGAUSS":
        return exampl_mixture_gauss(size=[TIMES_SLOTS_PER_BATCH, 2 * NUM_ANT, 1])
    elif NOISE_TYPE == "NAKA":
        return nakagami_var_batch(hps)
    else:
        return levy_var_batch(hps)


class TrainDb:
    def __init__(self, hps):
        self.hps = hps

    def fetch(self):
        for i in range(self.hps.train_total_batch):
            yield gen_noise_from_hps(self.hps)


class ValidDb:
    def __init__(self, hps):
        self.hps = hps

    def fetch(self):
        for i in range(self.hps.valid_total_batch):
            yield gen_noise_from_hps(self.hps)


class TestDb:
    def __init__(self, hps):
        self.hps = hps

    def fetch(self):
        for i in range(self.hps.test_total_batch):
            yield gen_batch(self.hps)
