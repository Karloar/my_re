import pickle as pkl
import os
import platform


cwd = os.getcwd()
model_dir = '/Users/karloar/Documents/other/ltp_data_v3.4.0'
if platform.system() == 'Windows':
    model_dir = r'E:\ltp_data'


wiki_vec_file = os.path.join(model_dir, 'wiki_vec_bin_model.pkl')

if __name__ == '__main__':
    wiki_vec = pkl.load(open(wiki_vec_file, 'rb'))
    for x in wiki_vec:
        print(x[0])