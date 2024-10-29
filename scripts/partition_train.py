from train import train_func


def main():
    
    LIST = 1
    NODE = f'f3_folds_tesst_13{LIST}]'
    REPORT_NAME = f'{NODE}_run'

    report_path = 'reports_2/'
    
    EPOCAS = 200
    BATCH_SIZE = 8
    CAP = 1.0
    
    # The import name is not gonna be used, 
    # so can be any valid file
    REPETITION = 'V2'
    IMPORT_NAME = f'{REPETITION}_E300_B32_S256_f3'
    SUPERVISED = True
    FREEZE = False
    DOWNSTREAM_DATA = 'f3'
    MODE = 'supervised'
    
    path = '../../shared_data/seismic_vinicius/'
    
    
    list_of_paths_1 = [
        'f3_fold_0',
        'f3_fold_0_cropped',
        'f3_fold_0_il',
        'f3_fold_0_xl',
        'f3_random_3',
    ]
    
    list_of_paths_2 = [
        'f3_fold_4',
        'f3_fold_4_cropped',
        'f3_fold_4_il',
        'f3_fold_4_xl',
        'f3_blocks',
    ]
    
    list_of_lists = [list_of_paths_1, list_of_paths_2]
    
    with open(report_path + f'{REPORT_NAME}.txt', 'w') as f:
        f.write('Report of the training\n')
        f.write('---------------------------------------\n')
        f.write(f'Paths being used: {list_of_lists[LIST]}\n')
        f.write(f'Node: {NODE}\n')
        f.write('---------------------------------------\n')
    
    
    for folder in list_of_lists[LIST]:
        
        ROOT_DIR = path + folder
        SAVE_NAME = folder
        
        with open(report_path + f'{REPORT_NAME}.txt', 'a') as f:
            f.write(f'------------------ Training on {path + folder} ------------------\n')
            f.write(f'------------------ Saving: {SAVE_NAME} ------------------\n')
        
        
        train_func(
            epocas=EPOCAS,
            batch_size=BATCH_SIZE,
            cap=CAP,
            import_name=IMPORT_NAME,
            save_name=SAVE_NAME,
            supervised=SUPERVISED,
            freeze=FREEZE,
            downstream_data=DOWNSTREAM_DATA,
            mode=MODE,
            repetition=REPETITION,
            root_dir=ROOT_DIR
        )
    
    
if __name__ == "__main__":
    main()

