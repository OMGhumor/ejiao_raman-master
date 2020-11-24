import os
import random
import shutil
import logging



def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def rmdir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)


def split_dataset(random_seed=1):
    '''

    :param random_seed:
    :return:
    '''
    BASEDIR = os.path.dirname(__file__)
    random.seed(random_seed)
    data_dir = os.path.join(BASEDIR, 'data')
    split_dir = os.path.join(BASEDIR, 'split_data')
    train_dir = os.path.join(split_dir, 'train')
    valid_dir = os.path.join(split_dir, 'valid')
    test_dir = os.path.join(split_dir, 'test')

    rmdir(train_dir)
    rmdir(valid_dir)
    rmdir(test_dir)

    train_pct = 0.8
    # valid_pct = 0.2
    # test_pct = 0.2

    for root, dirs, files in os.walk(data_dir):
        for sub_file in dirs:
            sample_txt_name = os.listdir(os.path.join(root, sub_file))
            random.shuffle(sample_txt_name)
            sample_count = len(sample_txt_name)

            train_point = int(sample_count * train_pct)
            # valid_point = int(sample_count * (train_pct + valid_pct))

            for i in range(sample_count):
                if i < train_point:
                    out_dir = os.path.join(train_dir, sub_file)
                # elif i < valid_point:
                #     out_dir = os.path.join(valid_dir, sub_file)
                else:
                    out_dir = os.path.join(test_dir, sub_file)

                makedir(out_dir)

                target_path = os.path.join(out_dir, sample_txt_name[i])
                src_path = os.path.join(data_dir, sub_file, sample_txt_name[i])

                shutil.copy(src_path, target_path)

            # print('Class:{}, train:{}, valid:{}, test:{}'.format(sub_file, train_point, valid_point - train_point,
            #                                                      sample_count - valid_point))
            #

if __name__ == '__main__':
    BASEDIR = os.path.dirname(__file__)
    random.seed(2)
    data_dir = os.path.join(BASEDIR, 'data')
    split_dir = os.path.join(BASEDIR, 'split_data')
    train_dir = os.path.join(split_dir, 'train')
    valid_dir = os.path.join(split_dir, 'valid')
    test_dir = os.path.join(split_dir, 'test')

    rmdir(train_dir)
    rmdir(valid_dir)
    rmdir(test_dir)

    train_pct = 0.6
    valid_pct = 0.2
    test_pct = 0.2

    for root, dirs, files in os.walk(data_dir):
        for sub_file in dirs:
            sample_txt_name = os.listdir(os.path.join(root, sub_file))
            random.shuffle(sample_txt_name)
            sample_count = len(sample_txt_name)

            train_point = int(sample_count * train_pct)
            valid_point = int(sample_count * (train_pct + valid_pct))

            for i in range(sample_count):
                if i < train_point:
                    out_dir = os.path.join(train_dir, sub_file)
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, sub_file)
                else:
                    out_dir = os.path.join(test_dir, sub_file)

                makedir(out_dir)

                target_path = os.path.join(out_dir, sample_txt_name[i])
                src_path = os.path.join(data_dir, sub_file, sample_txt_name[i])

                shutil.copy(src_path, target_path)

            print('Class:{}, train:{}, valid:{}, test:{}'.format(sub_file, train_point, valid_point - train_point,
                                                                 sample_count - valid_point))