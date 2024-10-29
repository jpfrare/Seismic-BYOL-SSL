from evaluate import eval_func
import numpy as np





def main():
    
    report_name = 'folds_eval_25_10'
    
    data = 'f3'
    main_path = '../../shared_data/seismic_vinicius/'
    
    # list_of_partitions = ['f3_fold_0', 'f3_fold_1', 'f3_fold_2', 'f3_fold_3', 'f3_fold_4', 
    #                     'f3_random_0', 'f3_random_1', 'f3_random_2', 'f3_random_3', 'f3_random_4', 
    #                     'f3_uniform_split', 'f3_highest_mse']
    # list_of_models = ['split_0', 'split_1', 'split_2', 'split_3', 'split_4', 
    #                   'random_0', 'random_1', 'random_2', 'random_3', 'random_4',
    #                   'uniform', 'high_mse']
    
    repetition = 'V_0.04'
    
    list_of_partitions = [
        'f3_fold_0',
        'f3_fold_0_cropped',
        'f3_fold_0_il',
        'f3_fold_0_xl',
        'f3_random_3',
        'f3_fold_4',
        'f3_fold_4_cropped',
        'f3_fold_4_il',
        'f3_fold_4_xl',
        'f3_blocks'
    ]
    
    list_of_models = [
        'f3_fold_0',
        'f3_fold_0_cropped',
        'f3_fold_0_il',
        'f3_fold_0_xl',
        'f3_random_3',
        'f3_fold_4',
        'f3_fold_4_cropped',
        'f3_fold_4_il',
        'f3_fold_4_xl',
        'f3_blocks'
    ]


    # list_of_partitions = ['dataset_random/', 'dataset_uniform/', 'dataset_window/', 'highest_mse_pair/']
    # list_of_models = ['random', 'uniform', 'window', 'high_mse_pair']
    
    # list_of_models = ['window_1', 'mse_pair_1']
    # list_of_partitions = ['dataset_window/', 'highest_mse_pair/']
    
    with open(report_name + '.txt', 'w') as f:
        f.write('---------------------------------------\n')
        f.write(f'Partitions: {list_of_partitions}\n')
        f.write(f'Models: {list_of_models}\n')
        f.write('---------------------------------------\n')
    
    for i in range(len(list_of_partitions)):
        partition = list_of_partitions[i]
        model = list_of_models[i]
        
        root_dir = main_path + partition
        
        with open(report_name + '.txt', 'a') as f:
            f.write(f'------------------ {partition} ------------------\n')
        
        iou, f1 = eval_func(import_name=model,
                  mode='supervised',
                  dataset=data,
                  repetition=repetition,
                  root_dir=root_dir,
                  )

        with open(report_name + '.txt', 'a') as f:
            f.write(30*'--' + '\n')
            f.write(model + '\n')
            f.write(f'iou = {iou[2]:.4f}\n')
            f.write(f'f1 = {f1[2]:.4f}\n')            
    
    
    



if __name__ == "__main__":
    main()