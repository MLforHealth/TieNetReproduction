from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='mimiccxr',
                       mimic_csv_path='./mimic-cxr-map.csv/',
                       image_folder='./images/',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='./',
                       max_len=50)
