import os
import pathlib

from scipy.stats import norm, nakagami
from scipy.special import gamma
from global_settings import *


def get_bits(x):
    return np.where(x < 0, 0, 1)


def check_wrong_bits(bits, bits_estimated):
    return len(np.argwhere(bits != bits_estimated))


def mkdir(file_path):
    folder = os.path.dirname(file_path)
    if not os.path.exists(folder):
        os.makedirs(folder)


def mkfile(file_path):
    mkdir(file_path)
    filename = pathlib.Path(file_path)
    filename.touch(exist_ok=True)


def concatenate(total, part):
    return part if total is None else np.concatenate((total, part))


def complex_channel(m=NUM_ANT, n=NUM_ANT):
    real = np.random.randn(m, n)
    imag = np.random.randn(m, n)
    h = np.row_stack(
        (
            np.column_stack((real, -imag)),
            np.column_stack((imag, real)),
        )
    )
    return h


def make_channel_batch():
    h_batch = None
    for _ in range(PACKETS_PER_BATCH):
        h = complex_channel().reshape([1, 2 * NUM_ANT, 2 * NUM_ANT])
        for _ in range(TIME_SLOTS_PER_PACKET):
            h_batch = concatenate(h_batch, h)
    return h_batch


def signal_batch(batch_size=TIMES_SLOTS_PER_BATCH):
    s_batch = None
    random_indexes = np.random.uniform(low=0, high=QPSK_CANDIDATE_SIZE, size=batch_size)
    for t in range(batch_size):
        i = int(random_indexes[t])
        s = QPSK_CANDIDATES[:, i:i + 1].reshape([1, 2 * NUM_ANT, 1])
        s_batch = concatenate(s_batch, s)
    return s_batch


def random_distance(n, length):
    x = np.random.uniform(-1, 1, [n, 1, 1]) * length / 2
    y = np.random.uniform(-1, 1, [n, 1, 1]) * length / 2
    return np.sqrt(x ** 2 + y ** 2)


def zf_batch(y, h):
    h_t = np.transpose(h, axes=[0, 2, 1])
    f = np.linalg.inv(h_t @ h) @ h_t
    z = f @ y
    return np.where(z < 0, -1, 1) / np.sqrt(2)


def lmmse_batch(y, h):
    assert len(h.shape) == 3
    batch_size, m, n = h.shape
    eye = np.concatenate([np.eye(n).reshape([1, n, n]) * batch_size], axis=0)
    ht = np.transpose(h, axes=[0, 2, 1])
    z = np.linalg.inv(ht @ h + eye) @ ht @ y
    return np.where(z < 0, -1, 1) / np.sqrt(2)


def nakagami_m(m, size):
    return nakagami.rvs(m, size=size)


def mixture_gaussian(probs, means, stds, size):
    y = [np.random.normal(means[i], stds[i], size) for i in range(len(probs))]
    p = np.random.uniform(0, 1, size)
    x = np.zeros(size)
    for i in reversed(range(len(probs))):
        x[p <= probs[i]] = y[i][p <= probs[i]]
    return x


def exampl_mixture_gauss(size):
    x = mixture_gaussian([0.5, 1], [-1, 1], [2, 1], size)
    x = x / np.sqrt(3.5)
    return x


def exampl_mixture_gauss2(size):
    x = np.zeros(size).flatten()
    for i in range(x.size):
        p = np.random.rand()
        if p < 0.5:
            x[i] = np.random.normal(-1, 2)
        else:
            x[i] = np.random.normal(1, 1)
    x = x / np.sqrt(3.5)
    return x.reshape(size)


def mle_nakagami(y, H, m):
    assert len(H.shape) == 3
    s_ml = np.zeros([H.shape[0], H.shape[2], 1])

    for i in range(y.shape[0]):
        y_ = y[i, :, :]
        H_ = H[i, :, :]
        max_logp = -np.inf
        best_cand = None
        for t in range(2 ** (2 * NUM_ANT)):
            cand = QPSK_CANDIDATES[:, t:t + 1].reshape([2 * NUM_ANT, 1])
            w_ = y_ - H_ @ cand
            if (w_ >= 0).all():
                logp = np.sum(nakagami.logpdf(w_, m))
                if best_cand is None or logp > max_logp:
                    best_cand = cand
                    max_logp = logp

        s_ml[i:i + 1, :, :] = best_cand.reshape([1, 2 * NUM_ANT, 1])

    return get_bits(s_ml)


def mle_nakagami2(y, H, m):
    assert len(H.shape) == 3
    s_ml = np.zeros([H.shape[0], H.shape[2], 1])

    w = y - H @ QPSK_CANDIDATES
    logp = np.sum(nakagami.logpdf(w, m), axis=1)
    logp = np.where(np.isnan(logp), -np.inf, logp)

    indexes = logp.argmax(1)
    for i, t in enumerate(indexes):
        s_ml[i:i + 1, :, :] = QPSK_CANDIDATES[:, t].reshape([1, 2 * NUM_ANT, 1])

    return get_bits(s_ml)


def mle_mixgauss(y, H):
    assert len(H.shape) == 3
    s_ml = np.zeros([H.shape[0], H.shape[2], 1])

    w = y - H @ QPSK_CANDIDATES
    w = w * np.sqrt(3.5)
    logp = np.sum(np.log(0.5 * norm.pdf(w, -1, 2) + 0.5 * norm.pdf(w, 1, 1)), axis=1)

    indexes = logp.argmax(1)
    for i, t in enumerate(indexes):
        s_ml[i:i + 1, :, :] = QPSK_CANDIDATES[:, t].reshape([1, 2 * NUM_ANT, 1])

    return get_bits(s_ml)


def mle_gauss(y, H):
    assert len(H.shape) == 3
    s_ml = np.zeros([H.shape[0], H.shape[2], 1])

    dst = np.sum(np.square(y - H @ QPSK_CANDIDATES), axis=1)

    indexes = dst.argmin(1)
    for i, t in enumerate(indexes):
        s_ml[i:i + 1, :, :] = QPSK_CANDIDATES[:, t].reshape([1, 2 * NUM_ANT, 1])

    return get_bits(s_ml)


def vector_average_norm(x):
    assert len(x.shape) == 3
    return np.mean(np.sum(x ** 2, axis=1))
