from train import train_func



"""
Com essa função é possível treinar modelos em sequência dentro de um loop.
A ideia é usar listas e dicionários para armazenar os parâmetros de cada modelo.

A função de treino pode receber uma lista de parâmetros que representa os modelos
a serem treinados. Todos devem ter um respectivo nome para serem salvos, batch_size,
cap e flag do treinamento supervisionado    
"""

def main():

    report_path = 'reports/'

    EPOCAS = 75
    BATCH_SIZE = 8
    FREEZE = False
    
    NODE = 'fuck'
    REPORT_NAME = f'{NODE}_run'
    
    # list_of_pretrains = ['sup', 'seg']
    # list_of_pretrains = ['seam_ai', 'seam_ai_norm']
    list_of_pretrains = ['f3_norm']
    # list_of_pretrains = ['both', 'both_N']
    # list_of_pretrains = ['COCO']
    # list_of_pretrains = ['sup', 'seg']

    list_of_datas = ['f3', 'seam_ai', 'f3_N', 'seam_ai_N']    
    
    list_of_repets = ['V9']
    
    list_of_seeds = list(range(48, 50))
    
    # list_of_repets = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10']
    # list_of_seeds = list(range(40, 60))   
    
    # list_of_caps = [0.01, 0.1, 0.5]
    list_of_caps = [1.0]
    # list_of_caps = [0.01, 0.1, 0.5, 1.0]
    
    with open(report_path + f'{REPORT_NAME}.txt', 'w') as f:
        f.write('Report of the training\n')
        f.write('---------------------------------------\n')
        f.write(f'Datas being used: {list_of_datas}\n')
        f.write(f'Pretrains models: {list_of_pretrains}\n')
        f.write(f'Caps: {list_of_caps}\n')
        f.write(f'Node: {NODE}\n')
        f.write('---------------------------------------\n')
    
    
    for num, repetition in enumerate(list_of_repets):
        for pretrain in list_of_pretrains:
            for data in list_of_datas:
                for cap in list_of_caps:
                    
                    if data == 'f3':
                        root_dir = '../../asml/datasets/tiff_data/f3_segmentation'
                    elif data == 'seam_ai':
                        root_dir = '../../asml/datasets/tiff_data/seam_ai'
                    elif data == 'f3_N':
                        root_dir = '../../asml/datasets/tiff_data/f3_segmentation_N'
                    elif data == 'seam_ai_N':
                        root_dir = '../../asml/datasets/tiff_data/seam_ai_N'
                    else:
                        raise ValueError('Data not found. Must be one of "f3" or "seam_ai"')
                    
                    with open(report_path + f'{REPORT_NAME}.txt', 'a') as f:
                        f.write(f'------------------ Pre on {pretrain} and train on {data} ------------------\n')
                        print(30*'*-')
                        print(f'Running with data {data} and model pretrained in {pretrain} with cap {cap*100:.0f}%. ')
                        print(30*'*-')
                    
                    if pretrain  in ['f3', 'seam_ai', 'both', 'f3_norm', 'seam_ai_norm', 'both_N']:
                        mode = 'byol'
                        supervised = False
                        import_name = f'{repetition}_E300_B32_S256_{pretrain}'
                        save_name = f'{repetition}_pre_{pretrain}_train_{data}_cap_{cap*100:.0f}%'
                    
                    elif pretrain == 'seg':
                        mode = 'seg'
                        supervised = False
                        if data == 'f3':
                            import_name = f'{repetition}_sup_seam_ai_cap_100%'
                        elif data == 'seam_ai':
                            import_name = f'{repetition}_sup_f3_cap_100%'
                        elif data == 'f3_N':
                            import_name = f'{repetition}_sup_seam_ai_N_cap_100%'
                        elif data == 'seam_ai_N':
                            import_name = f'{repetition}_sup_f3_N_cap_100%'
                        save_name = f'{repetition}_pre_{pretrain}_train_{data}_cap_{cap*100:.0f}%'
                        
                    elif pretrain == 'COCO':
                        mode = 'coco'
                        supervised = False
                        # Não importa, não será usado. Deve ser um import válido
                        import_name = f'{repetition}_E300_B32_S256_f3'         
                        save_name = f'{repetition}_pre_COCO_train_{data}_cap_{cap*100:.0f}%'
                    
                    elif pretrain == 'IMAGENET':
                        mode = 'imagenet'
                        supervised = False
                        # Não importa, não será usado. Deve ser um import válido
                        import_name = f'{repetition}_E300_B32_S256_f3'
                        save_name = f'{repetition}_pre_IMAGENET_train_{data}_cap_{cap*100:.0f}%'
                    
                    elif pretrain == 'sup':
                        mode = 'supervised'
                        supervised = True
                        # Não importa, não será usado. Deve ser um import válido
                        import_name = f'{repetition}_E300_B32_S256_f3'
                        save_name = f'{repetition}_sup_{data}_cap_{cap*100:.0f}%'

                    with open(report_path + f'{REPORT_NAME}.txt', 'a') as f:
                            f.write(f'Running with data {data} and model pretrained in {pretrain} with cap {cap*100:.0f}%. ')
                            f.write(f'Import name: {import_name}\n')
                            f.write(f'Save name: {save_name}\n')
                            f.write(f'Mode: {mode}; Cap: {cap*100:.0f}%; Repetitions: {repetition}; Seed: {list_of_seeds[num]}\n')
                        
                    train_func(
                        epocas=EPOCAS,
                        batch_size=BATCH_SIZE,
                        cap=cap,
                        import_name=import_name,
                        save_name=save_name,
                        supervised=supervised,
                        freeze=FREEZE,
                        downstream_data=data,
                        mode=mode,
                        repetition=repetition,
                        seed=list_of_seeds[num],
                        root_dir=root_dir,
                        # aux=list_of_repets_aux[num]
                    )
                        
                    with open(report_path + f'{REPORT_NAME}.txt', 'a') as f:
                        f.write('---------------------------------------\n')
                        f.write('Treinamento finalizado\n')
                        f.write('---------------------------------------\n')
            
                            

if __name__ == "__main__":
    main()