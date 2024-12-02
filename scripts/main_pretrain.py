from pretrain import pretrain_func



"""
Com essa função é possível treinar modelos em sequência dentro de um loop.
A ideia é usar listas e dicionários para armazenar os parâmetros de cada modelo.

A função de treino pode receber uma lista de parâmetros que representa os modelos
a serem treinados. Todos devem ter um respectivo nome para serem salvos, batch_size,
cap e flag do treinamento supervisionado    
"""



def main():
    
    NODE = 'gpu[2]'
    REPORT_NAME = f'pretrain_{NODE}_run'

    report_path = 'reports_2/'

    EPOCAS = 300
    BATCH_SIZE = 32
    INPUT_SIZE = 256
    
    list_of_repets = ['V01', 'V2', 'V3', 'V4', 'V5'] 
    # list_of_repets = ['V6', 'V7', 'V8', 'V9', 'V10']
    
    # list_of_datas = ['both']
    # path = '../../asml/datasets/tiff_data/both/images'
    
    list_of_datas = ['both_N']
    path = '../../asml/datasets/tiff_data/both_N/images'
    
    # list_of_datas = ['f3']
    # path = '../../asml/datasets/tiff_data/f3_segmentation/images'
    
    # list_of_datas = ['seam_ai']
    # path = '../../asml/datasets/tiff_data/seam_ai/images'
    
    # list_of_datas = ['f3_norm']
    # path = '../../asml/datasets/tiff_data/f3_segmentation_N/images'
    
    # list_of_datas = ['seam_ai_norm']
    # path = '../../asml/datasets/tiff_data/seam_ai_N/images'
    
    with open(report_path + f'{REPORT_NAME}.txt', 'w') as f:
        f.write('Report of the pretraining\n')
        f.write('---------------------------------------\n')
        f.write(f'Datas being used: {list_of_datas}\n')
        f.write(f'Node: {NODE}\n')
        f.write(f'Repets: {list_of_repets}\n')
        f.write('---------------------------------------\n')
    
    for repetition in list_of_repets:
        for data in list_of_datas:
            
            print(30*'*-')
            print(f'Running with data {data}. ')
            print(30*'*-')
            
            with open(report_path + f'{REPORT_NAME}.txt', 'a') as f:
                f.write(30*'*-' + '\n')
                f.write(f'------------------ Pretraining on {data} ------------------\n')
                f.write(f'Repetition: {repetition}')
            
            save_name = f'{repetition}_E{EPOCAS}_B{BATCH_SIZE}_S{INPUT_SIZE}_{data}'
            
            pretrain_func(epocas=EPOCAS,
                        batch_size=BATCH_SIZE,
                        input_size=INPUT_SIZE,
                        repetition=repetition,
                        save_name=save_name,
                        data=data,
                        path=path
                        )

            with open(report_path + f'{REPORT_NAME}.txt', 'a') as f:
                f.write(f'------------------ Pretrain finished ------------------\n')
                

if __name__ == "__main__":
    main()