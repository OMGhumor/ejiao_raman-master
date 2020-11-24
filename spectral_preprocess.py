# import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pyhht
import rampy as rp
from pyhht.visualization import plot_imfs
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve

from dataset import *

# COLORS = list(mcolors.TABLEAU_COLORS.keys())  # 颜色变化


# 基线校正
class AirPLS(object):
    def __init__(self, x, lamda=10, porder=1, itermax=30):
        self.x = x
        self.lamda = lamda
        self.porder = porder
        self.itermax = itermax

    def airPLS(self):
        '''
        自适应的迭代惩罚重加权算法
        Adaptive iteratively reweighted penalized least squares for baseline fitting
        input
            x: input data (i.e. chromatogram of spectrum)
            lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z
            porder: adaptive iteratively reweighted penalized least squares for baseline fitting
        output
            the fitted background vector
        '''
        m = self.x.shape[0]
        w = np.ones(m)
        for i in range(1, self.itermax + 1):
            # 对区县进行平滑，w全为1，表明全不为峰
            z = AirPLS.WhittakerSmooth(self.x, w, self.lamda, self.porder)
            # 平滑后的差距
            d = self.x - z
            # 计算平滑后比原值低的总和
            dssn = np.abs(d[d < 0].sum())
            if dssn < 0.001 * (abs(self.x)).sum() or i == self.itermax:
                if i == self.itermax:
                    print('WARING max iteration reached!')
                break
            w[d >= 0] = 0
            # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
            w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
            w[0] = np.exp(i * (d[d < 0]).max() / dssn)
            w[-1] = w[0]
        return self.x - z

    @staticmethod
    def WhittakerSmooth(x, w, lambda_, differences=1):
        '''
        惩罚最小二乘算法
        Penalized least squares algorithm for background fitting
        input
            x: input data (i.e. chromatogram of spectrum)
            w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
            lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
            differences: integer indicating the order of the difference of penalties
        output
            the fitted background vector
        '''
        X = np.matrix(x)
        m = X.size
        i = np.arange(0, m)
        E = eye(m, format='csc')
        D = E[1:] - E[:-1]  # numpy.diff() does not work with sparse matrix. This is a workaround.
        W = diags(w, 0, shape=(m, m))
        A = csc_matrix(W + (lambda_ * D.T * D))
        B = csc_matrix(W * X.T)
        background = spsolve(A, B)
        return np.array(background)


class RamanSpectral(object):
    '''
    定义拉曼光谱数据类
    '''

    def __init__(self, bands: list, intensity: list, sample_name=None):
        self.__bands = bands
        self.__intensity = intensity
        self.__sample_name = 'RamanSpectral'
        if sample_name is not None:
            self.__sample_name = sample_name

    def get_bands(self):
        return self.__bands

    def get_intensity(self):
        return self.__intensity

    def set_intensity(self, x: list):
        self.__intensity = x

    def plot(self):
        fig, ax = plt.subplots()
        ax.set_xlim(200, 3000)
        ax.set_xlabel('Bands')
        ax.set_ylabel('Intensity')
        ax.set_title(self.__sample_name)
        ax.plot(self.__bands, self.__intensity, color='red', linewidth=0.5)
        plt.show()


def emd_decompose(sample: RamanSpectral, is_plot: bool = False):
    emd = pyhht.emd.EMD(np.array(sample.get_intensity()))
    imfs = emd.decompose()
    base = imfs[4, :]
    res = base - np.array(sample.get_intensity())
    if is_plot:
        plot_one_sample((sample.get_bands(), res), 'EMD')
        plot_imfs(np.array(sample.get_intensity()), imfs, np.array(sample.get_bands()))
    return RamanSpectral(sample.get_bands(), list(res))


def read_raman_sample(path_dir: str):
    '''

    :param path_dir:
    :return:
    '''
    with open(path_dir, 'r', encoding='utf-8') as fr:
        txt = fr.readlines()

    doc = [i.strip().split('\t') for i in txt]
    bands, intensity = [float(i[0]) for i in doc], [float(i[1]) for i in doc]
    raman_sample = RamanSpectral(bands[200:], intensity[200:], path_dir.split('/')[-1])

    return raman_sample


def plot_split_data(random_seed):
    '''
    划分出数据集
    并画训练集的图像（某一次随机分类）
    :param random_seed:
    :return:
    '''
    split_dataset(random_seed)
    BASEDIR = os.path.dirname(__file__)
    real_train_dir = os.path.join(BASEDIR, 'split_data/train/real')
    fake_train_dir = os.path.join(BASEDIR, 'split_data/train/fake')
    real_train_sample_dir = [os.path.join(real_train_dir, i) for i in os.listdir(real_train_dir)]
    fake_train_sample_dir = [os.path.join(fake_train_dir, i) for i in os.listdir(fake_train_dir)]

    real_train_sample = [read_raman_sample(i) for i in real_train_sample_dir]
    fake_train_sample = [read_raman_sample(i) for i in fake_train_sample_dir]

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_xlim(200, 3000)
    ax1.set_xlabel('Bands')
    ax1.set_ylabel('Intensity')
    ax1.set_title('Real')
    ax2.set_xlim(200, 3000)
    ax2.set_xlabel('Bands')
    ax2.set_ylabel('Intensity')
    ax2.set_title('Fake')
    for index, i in enumerate(real_train_sample):
        ax1.plot(i.get_bands(), i.get_intensity(), c=[random.random(), random.random(), random.random()], linewidth=0.2)
    for index, i0 in enumerate(fake_train_sample):
        ax2.plot(i0.get_bands(), i0.get_intensity(), c=[random.random(), random.random(), random.random()],
                 linewidth=0.2)
    plt.show()


def plot_one_sample(sample, title):
    if isinstance(sample, RamanSpectral):
        fig, ax = plt.subplots()
        ax.set_xlim(200, 3000)
        ax.set_xlabel('Bands')
        ax.set_ylabel('Intensity')
        ax.set_title(title)
        ax.plot(sample.get_bands(), sample.get_intensity(), color='red', linewidth=0.5)
        plt.show()

    if isinstance(sample, tuple):
        fig, ax = plt.subplots()
        ax.set_xlim(200, 3000)
        ax.set_xlabel('Bands')
        ax.set_ylabel('Intensity')
        ax.set_title(title)
        ax.plot(sample[0], sample[1], color='red', linewidth=0.5)
        plt.show()


def baseline_correction(data_dir: str, method: str, is_plot: bool):
    """
    :param data_dir:
    :param method:
        "AirPLS":自己编写的airpls
        "arPLS":rampy中的基线校正方法
        "poly": polynomial fitting, with splinesmooth the degree of the polynomial.
        "unispline": spline with the UnivariateSpline function of Scipy, splinesmooth is
                     the spline smoothing factor (assume equal weight in the present case);
        "gcvspline": spline with the gcvspl.f algorythm, really robust.
                     Spectra must have x, y, ese in it, and splinesmooth is the smoothing factor;
                     For gcvspline, if ese are not provided we assume ese = sqrt(y).
                     Requires the installation of gcvspline with a "pip install gcvspline" call prior to use;
        "exp": exponential background;
        "log": logarythmic background;
        "rubberband": rubberband baseline fitting;
        "als": (automatic) baseline least square fitting following Eilers and Boelens 2005;
        "arPLS": (automatic) Baseline correction using asymmetrically reweighted penalized least squares smoothing. Baek et al. 2015, Analyst 140: 250-257;
        'drPLS': (automatic) Baseline correction method based on doubly reweighted penalized least squares. Xu et al., Applied Optics 58(14):3913-3920.
    :param is_plot: 是否画图，True画
    :return:
    """

    if isinstance(data_dir, str):
        sample = read_raman_sample(data_dir)

    else:
        assert isinstance(data_dir, RamanSpectral)
        sample = data_dir

    # if is_plot:
    # plot_one_sample(sample, 'Origin')
    # 基线校正
    if method == "AirPLS":
        airpls = AirPLS(np.array(sample.get_intensity()))
        x = airpls.airPLS()
        if is_plot:
            plot_one_sample((sample.get_bands(), list(x)), 'AirPLS')
    elif method == "emd":
        x = sample.get_intensity()
        emd = pyhht.emd.EMD(np.array(x))
        imfs = emd.decompose()
        base = imfs[-1, :]
        res = np.array(x) - base
        if is_plot:
            pyhht.plot_imfs(x, imfs, sample.get_bands())
        return res
    else:
        x = np.array(sample.get_bands())
        y = np.array(sample.get_intensity())
        bir = np.array([[0, 3000]])

        y_corrected, background = rp.baseline(x, y, bir, method, lam=10 ** 10)
        y_corrected = list(y_corrected.reshape(-1))
        background = list(background.reshape(-1))
        if is_plot:
            plot_one_sample((sample.get_bands(), y_corrected), "Corrected" + ' ' + method)
            # plot_one_sample((sample.get_bands(), background), "Backgroud")

        return y_corrected


if __name__ == '__main__':
    # plot_split_data(9)
    # sample = read_raman_sample('./split_data/train/real/t-2_0.txt')
    baseline_correction('./split_data/train/real/t-2_0.txt', method='arPLS', is_plot=True)
    baseline_correction('./split_data/train/fake/f-2_0.txt', method='arPLS', is_plot=True)

    baseline_correction('./split_data/train/real/t-2_0.txt', method='drPLS', is_plot=True)
    baseline_correction('./split_data/train/fake/f-2_0.txt', method='drPLS', is_plot=True)

    baseline_correction('./split_data/train/real/t-2_0.txt', method='als', is_plot=True)
    a=baseline_correction('./split_data/train/fake/f-2_0.txt', method='als', is_plot=True)

    # baseline_correction('./split_data/train/real/t-2_0.txt', method='rubberband', is_plot=True)
    # baseline_correction('./split_data/train/fake/f-2_0.txt', method='rubberband', is_plot=True)
