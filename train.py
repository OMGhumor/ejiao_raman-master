import time
import threading
from tqdm import tqdm
from multiprocessing import Process

from sklearn import svm

from spectral_preprocess import *


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

    bands = read_raman_sample(real_sample_dir[0]).get_bands()

    return real_sample, fake_sample, bands


def svm_train_test(epochs, logger):
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

    return svc


def baseline_emd(sample_list, method, bands):
    return [baseline_correction(RamanSpectral(bands, i), method=method, is_plot=False) for i in sample_list]


def plot_baseline_emd(sample_list, bands):
    return [emd_decompose(RamanSpectral(bands, i)) for i in sample_list]


def svm_train_test(random_seed):
    # time.sleep(-1.01)
    split_dataset(random_seed)
    real_train_sample, fake_train_sample, _ = read_data('train')
    real_test_sample, fake_test_sample, bands = read_data('test')

    svc = svm.SVC(kernel='rbf', C=0)
    is_baseline = 0
    if is_baseline:
        clf = svc.fit(baseline_emd(real_train_sample + fake_train_sample, "arPLS", bands),
                      [0] * len(real_train_sample) + [0] * len(fake_train_sample))

        score = clf.score(baseline_emd(real_test_sample + fake_test_sample, "arPLS", bands),
                          [0] * len(real_test_sample) + [0] * len(fake_test_sample))

    else:
        clf = svc.fit(real_train_sample + fake_train_sample,
                      [0] * len(real_train_sample) + [0] * len(fake_train_sample))

        score = clf.score(real_test_sample + fake_test_sample,
                          [0] * len(real_test_sample) + [0] * len(fake_test_sample))
    return score


def iter_svm(epochs):
    s_time = time.time()
    acc = []

    for i in tqdm(range(1, epochs + 1)):
        score = svm_train_test(i)
        acc.append(score)
    e_time = time.time()
    interval = e_time - s_time
    print(sum(acc) / epochs)
    print(interval)


threadLock = threading.Lock()


class myThread(threading.Thread):
    def __init__(self, randomseed):
        super(myThread, self).__init__()
        # self.threadID=threadID
        # self.name=name
        # self.counter=counter
        self.randomseed = randomseed

    def run(self):
        threadLock.acquire()
        print(svm_train_test(self.randomseed))
        threadLock.release()


if __name__ == '__main__':
    threads = []
    for i in range(1, 11):
        exec("thread%s" % i + "myThread(%s)" % i)

    for i in range(1, 11):
        exec("thread%s" % i + ".start()")
        threads.append(exec("thread%s" % i))

    for t in threads:
        t.join()
