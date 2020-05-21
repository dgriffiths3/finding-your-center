import numpy as np
import glob
import shutil
import os


label_lut = {
    'ceiling': 0, 'floor': 1, 'wall': 2, 'beam': 3, 'column': 4,
    'door': 5, 'window': 6, 'table': 7, 'chair': 8, 'sofa': 9,
    'bookcase': 10, 'board': 11, 'clutter':12, 'stairs':13
}


def process_files(config):

    if not os.path.isdir(config['out_dir']): os.mkdir(config['out_dir'])

    for area_num in range(1, 7):
        sub_folders = glob.glob(os.path.join(config['in_dir'], 'Area_{}'.format(area_num), '*'))

        first_file = True

        for room_folder in sub_folders:

            room_name = room_folder.split('/')[-1]

            first_file = True

            if os.path.isdir(room_folder):

                files = glob.glob(os.path.join(room_folder, 'Annotations', '*.txt'))

                for file in files:

                    label = label_lut[file.split('/')[-1].split('.')[0].split('_')[0]]

                    if first_file == True:
                        first_file = False
                        data = np.loadtxt(file, delimiter=' ')
                        data = np.insert(data, data.shape[-1], label, axis=1)
                    else:
                        file_data = np.loadtxt(file, delimiter=' ')
                        file_data = np.insert(file_data, data.shape[-1]-1, label, axis=1)
                        data = np.vstack((data, file_data))
                np.random.shuffle(data)
                np.save(
                    os.path.join(config['out_dir'], 'Area_'+str(area_num)+'_'+room_name+'.npy'),
                    data,
                )
                print('[info] processed room: {:}'.format(room_folder))


if __name__ == "__main__":

    config = {
        'in_dir' : './data/Stanford3dDataset',
        'out_dir' : './data/datasets/Stanford3dDataset/processed',
    }

    process_files(config)
