from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='mimiccxr',
                       base_path='/crimea/mimic-cxr/',
                       reports_per_image=5,
                       min_word_freq=5,
                       output_folder='/crimea/liuguanx/mimic-output/',
                       max_len=100)
