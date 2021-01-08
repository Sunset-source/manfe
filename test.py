from db import *
from detnet import *
from manfe import *
from time import time

import os


class HyperParameters:
    def __init__(self):
        self.snr = 25
        self.rho = 0.5
        self.alpha = 1.5
        self.beta = 0.0
        self.mu = 0.0
        self.sigma = 1.0

        self.m = 2
        self.omega = 1

        self.max_flip = 1
        self.max_epoch = 100
        self.train_total_batch = 10000
        self.valid_total_batch = 2000
        self.test_total_batch = 1000


def create_model(hps):
    if NOISE_TYPE == "MIXGAUSS":
        return MANFE_MIXGUASS(8, 4)
    elif NOISE_TYPE == "NAKA":
        return MANFE_NAKA(hps.m, hps.omega, 8, 4)
    else:
        return MANFE_SAS(hps.alpha, 8, 4)


def train(hps):
    train_set = TrainDb(hps)
    valid_set = ValidDb(hps)

    model = create_model(hps)
    model.train(train_set, valid_set, hps.max_flip, hps.max_epoch)
    model.close()


def benchmark(hps):
    detnet = DetNet(NUM_ANT, 4 * 2 * NUM_ANT, 30, hps.alpha, hps.snr)
    detnet.load()

    manfe = create_model(hps)
    manfe.load()

    err_detnet = 0
    err_gamp = 0
    err_mld = 0
    err_gamp_mld_1 = 0
    err_gamp_mld_2 = 0
    err_gamp_manfe_1 = 0
    err_gamp_manfe_2 = 0
    err_manfe = 0
    err_ml = 0
    total_bits = 0

    test_set = TestDb(hps)

    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0
    t5 = 0

    time_slots = 0
    batch_count = 0
    for y, h, s, w in test_set.fetch():
        batch_count += 1
        time_slots += y.shape[0]

        # for i in range(y_batch.shape[0]):
        #
        #     y = y_batch[i:i + 1, :, :]
        #     h = h_batch[i:i + 1, :, :]
        #     s = s_batch[i:i + 1, :, :]

        bits = get_bits(s)
        total_bits += bits.size

        t_start = time()
        bits_detnet = detnet.detect_bits(y, h)
        t1 += time() - t_start

        t_start = time()
        s_gamp, _ = amp_batch(y, h, loop=30)
        bits_gamp = get_bits(s_gamp)
        t_gamp = time() - t_start
        t2 += t_gamp

        # bits_gamp_mld_1 = manfe.detect_bits_with_initial_guess(y, h, s_gamp, max_error_symbols=1, use_mld=True)
        # bits_gamp_mld_2 = manfe.detect_bits_with_initial_guess(y, h, s_gamp, max_error_symbols=2, use_mld=True)

        t_start = time()
        bits_gamp_manfe_1 = manfe.detect_bits_with_initial_guess(y, h, s_gamp, max_error_symbols=1, use_mld=False)
        t3 += t_gamp + time() - t_start

        # bits_gamp_manfe_2 = manfe.detect_bits_with_initial_guess(y, h, s_gamp, max_error_symbols=3, use_mld=False)

        t_start = time()
        bits_mld = mle_gauss(y, h)
        t4 += time() - t_start

        t_start = time()
        bits_manfe = manfe.detect_bits(y, h)
        t5 += time() - t_start

        if NOISE_TYPE == "MIXGAUSS":
            bits_ml = mle_mixgauss(y, h)
        elif NOISE_TYPE == "NAKA":
            bits_ml = mle_nakagami(y, h, hps.m)
        else:
            bits_ml = mle_gauss(y, h)

        err_detnet += check_wrong_bits(bits, bits_detnet)
        err_gamp += check_wrong_bits(bits, bits_gamp)
        # err_gamp_mld_1 += check_wrong_bits(bits, bits_gamp_mld_1)
        # err_gamp_mld_2 += check_wrong_bits(bits, bits_gamp_mld_2)
        err_gamp_manfe_1 += check_wrong_bits(bits, bits_gamp_manfe_1)
        # err_gamp_manfe_2 += check_wrong_bits(bits, bits_gamp_manfe_2)
        err_mld += check_wrong_bits(bits, bits_mld)
        err_manfe += check_wrong_bits(bits, bits_manfe)
        err_ml += check_wrong_bits(bits, bits_ml)

        ber_detnet = err_detnet / total_bits
        ber_gamp = err_gamp / total_bits
        ber_gamp_mld_1 = err_gamp_mld_1 / total_bits
        ber_gamp_mld_2 = err_gamp_mld_2 / total_bits
        ber_gamp_manfe_1 = err_gamp_manfe_1 / total_bits
        ber_gamp_manfe_2 = err_gamp_manfe_2 / total_bits
        ber_mld = err_mld / total_bits
        ber_manfe = err_manfe / total_bits
        ber_ml = err_ml / total_bits

        precision = 1 / total_bits

        data_text = "DetNet={:.4f}s GAMP={:.4f}s GAMP-MANFE={:.4f}s E-MLE={:.4f}s MANFE={:.4f}s {}".format(
            t1 / time_slots,
            t2 / time_slots,
            t3 / time_slots,
            t4 / time_slots,
            t5 / time_slots,
            batch_count
        )

        # data_text = "SNR={:02d} GAMP(30)={:e} GAMP(30)-MANFE(2)={:e} E-MLE={:e} MANFE={:e} MLE={:e} ({:e} {})".format(
        #     hps.snr,
        #     ber_detnet,
        #     ber_gamp,
        #     ber_gamp_mld_1,
        #     ber_gamp_mld_2,
        #     ber_gamp_manfe_1,
        #     ber_gamp_manfe_2,
        #     ber_mld,
        #     ber_manfe,
        #     ber_ml,
        #     precision,
        #     batch_count)

        print(data_text)

    print()
    print()
    # detnet.close()
    manfe.close()


def main():
    ghps = HyperParameters()

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force cpu

    # train(ghps)
    for j in [10]:
        ghps.snr = j
        benchmark(ghps)
        print()


if __name__ == "__main__":
    main()

'''

'''