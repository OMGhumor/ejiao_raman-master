import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from dataset import *

COLORS = list(mcolors.TABLEAU_COLORS.keys())  # 颜色变化


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

    def plot(self):
        fig, ax = plt.subplots()
        ax.set_xlim(200, 3000)
        ax.set_xlabel('Bands')
        ax.set_ylabel('Intensity')
        ax.set_title(self.__sample_name)
        ax.plot(self.__bands, self.__intensity, color='red', linewidth=0.5)
        plt.show()


def read_raman_sample(path_dir: str):
    '''

    :param path_dir:
    :return:
    '''
    with open(path_dir, 'r', encoding='utf-8') as fr:
        txt = fr.readlines()

    doc = [i.strip().split('\t') for i in txt]
    bands, intensity = [float(i[0]) for i in doc], [float(i[1]) for i in doc]
    raman_sample = RamanSpectral(bands, intensity, path_dir.split('/')[-1])

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


if __name__ == '__main__':
    plot_split_data(9)
