import numpy as np
from sklearn import svm

from spectral_preprocess import *
from LoggingInfo import *


def read_data(category):
    '''

    :param category:[test,train,valid]三选一
    :return:
    '''
    BASEDIR = os.path.dirname(__file__)
    real_dir = os.path.join(BASEDIR, 'split_data/%s/real' % category)
    fake_dir = os.path.join(BASEDIR, 'split_data/%s/fake' % category)
    real_sample_dir = [os.path.join(real_dir, i) for i in os.listdir(real_dir)]
    fake_sample_dir = [os.path.join(fake_dir, i) for i in os.listdir(fake_dir)]

    real_sample = [read_raman_sample(i).get_intensity() for i in real_sample_dir]
    fake_sample = [read_raman_sample(i).get_intensity() for i in fake_sample_dir]

    return real_sample, fake_sample


def train(epochs, logger):
    acc = []
    logger.info(" Training epoch: {}".format(epochs))
    for i in range(1, epochs):
        split_dataset(i)
        real_train_sample, fake_train_sample = read_data('train')
        real_test_sample, fake_test_sample = read_data('test')

        svc = svm.SVC(kernel='rbf', C=1)
        clf = svc.fit(real_train_sample + fake_train_sample,
                      [1] * len(real_train_sample) + [0] * len(fake_train_sample))

        score = clf.score(real_test_sample + fake_test_sample,
                          [1] * len(real_test_sample) + [0] * len(fake_test_sample))

        acc.append(score)
        logger.info(" \tBatch({:>3}/{:>3}) done. Loss:{}".format(i, epochs, score))
        # print(svc.n_support_)
        # print(svc.support_)
        # print(svc.support_vectors_)

    acc = np.array(acc)
    # support_vec = svc.support_vectors_
    logger.info(" \tAll Done. Mean Loss:{} Std:{:.4f}".format(acc.mean(), float(acc.std())))
    # print(acc.mean(), acc.std())


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    logger = loadLogger(args)
    # train(10,logger)
    epochs = 11
    acc = []
    logger.info(" Training epoch: {}".format(epochs - 1))
    for i in range(1, epochs):
        split_dataset(i)
        real_train_sample, fake_train_sample = read_data('train')
        real_test_sample, fake_test_sample = read_data('test')

        svc = svm.SVC(kernel='rbf', C=1)
        clf = svc.fit(real_train_sample + fake_train_sample,
                      [1] * len(real_train_sample) + [0] * len(fake_train_sample))

        score = clf.score(real_test_sample + fake_test_sample,
                          [1] * len(real_test_sample) + [0] * len(fake_test_sample))

        acc.append(score)
        logger.info(" \tBatch({:>3}/{:>3}) done. Loss:{}".format(i, epochs - 1, score))
        # print(svc.n_support_)
        # print(svc.support_)
        # print(svc.support_vectors_)

    acc = np.array(acc)
    # support_vec = svc.support_vectors_
    logger.info(" \tAll Done. Mean Loss:{} Std:{:.4f}".format(acc.mean(), float(acc.std())))
    # print(acc.mean(), acc.std())
