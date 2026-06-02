import pandas as pd
import os
import matplotlib.pyplot as plt
import math
from pathlib import Path
import yaml
import numpy as np

class TrainMetrics():
    raw_data: dict[str, tuple[Path,Path,Path]] #dicioário: nome_modelo -> lista com os caminhos do csv de cada repetição
    save_root: Path                            #onde os dados serão salvos
    model_csvs: dict

    def __init__(self, raw_data, save_root):
        self.raw_data = raw_data
        self.save_root = Path(save_root)
        self.model_csvs = self._get_model_csvs()

    
    def _get_model_csvs(self):
        model_csv = {}
        for model_name in self.raw_data.keys():
            data = []
            for repetition in range(3):
                csv_path = self.raw_data[model_name][repetition]
                df_repetition = pd.read_csv(csv_path)
                df_repetition = df_repetition.groupby('step').first().reset_index()
                '''importante ver que o agrupamento por step e junto com o .first() faz com que, ele junte as informações da loss de treino
                com a loss de validação para o mesmo step de validação (o que é importante e precisamos) '''
                df_repetition = df_repetition.dropna(subset=['train_loss_epoch', 'val_acc1', 'val_acc5', 'val_loss'])
                df_repetition['key'] = repetition
                data.append(df_repetition)
            
            df_model = pd.concat(data)
            model_to_curves = df_model.groupby('step').agg(
                mean_val_loss= ('val_loss', 'mean'),
                std_val_loss= ('val_loss', 'std'),
                mean_train_loss = ('train_loss_epoch', 'mean'),
                std_train_loss = ('train_loss_epoch', 'std'),
                mean_acc1= ('val_acc1', 'mean'),
                std_acc1= ('val_acc1', 'std'),
                mean_acc5= ('val_acc5', 'mean'),
                std_acc5= ('val_acc5', 'std') 
            ).reset_index()

            model_csv[model_name] = model_to_curves
        
        return model_csv
    
    def group_plot(self, y_axis: str, x_axis: str, mul_factor: int, title: str, xlabel_title: str, ylabel_title: str, yscale: str = None):
        '''plot de apenas um gráfico da única variável escolhida para todos os modelos (média sombreado com desvio padrão)
        mul_factor é o fator multiplicativo para a variável y'''

        plt.figure(figsize=(16,6))

        for model_name, dataframe in self.model_csvs.items():
            mean = dataframe[f'mean_{y_axis}']*mul_factor
            std = dataframe[f'std_{y_axis}']*mul_factor

            plt.plot(
                dataframe[x_axis],
                mean,
                label= f'{model_name}',
                linewidth= 2
            )

            plt.fill_between(
                dataframe[x_axis],
                mean - std,
                mean + std,
                alpha=0.15, zorder= 2
            )
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(xlabel_title, fontsize= 12)
        plt.ylabel(ylabel_title, fontsize= 12)

        if yscale is not None:
            plt.yscale(yscale)

        plt.grid(True, linestyle= '--', alpha= 0.6)
        plt.legend(fontsize=10)

        save_path = self.save_root/f'{title}.png'
        plt.savefig(save_path, dpi= 300, bbox_inches='tight')
        plt.close()
    
    def individual_plot(self, y_axis: list[str], x_axis: str, mul_factor: int, title: str, ncols: int = 4, yscale: str = None):
        '''plota vários gráficos (cada um referente a um modelo) de um conjunto de métricas escolhido (média sombreado com desvio padrão)'''

        num_models = len(self.model_csvs)
        nrows = math.ceil(num_models/ncols) #dado um número de colunas, consegue calcular o número de linhas

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8)) #fig -> contém a painel que contém os demais gráficos, axs é a lista de mini gráficos

        #mesmo que haja apenas um modelo, torna ax iterável
        if num_models > 1:
            axs = axs.flatten() #pega uma matriz e achata ela, concatena as linhas em colunas
        else:
            axs = [axs]

        for i, (model_name, dataframe) in enumerate(self.model_csvs.items()): 
            ax = axs[i]

            for metric in y_axis: #itera pelas métricas pedidas e as plota
                mean = dataframe[f'mean_{metric}']*mul_factor
                std = dataframe[f'std_{metric}']*mul_factor

                ax.plot(
                    dataframe[x_axis],
                    mean,
                    linewidth= 2,
                    label= metric
                )

                ax.fill_between(
                    dataframe[x_axis],
                    mean - std,
                    mean + std,
                    alpha= 0.15,
                    zorder= 2
                )

            ax.set_title(model_name, fontsize=12, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(fontsize= 9)

            if yscale is not None:
                ax.set_yscale(yscale)

        #apaga os quadradinhos não preenchidos
        for i in range(num_models, len(axs)):
            fig.delaxes(axs[i])
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        fig.tight_layout()

        save_path = self.save_root / f'{title}_Clean_Grid.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def show_last_step_metrics(self, filename):
        data = []
        for model_name, dataframe in self.model_csvs.items():
            max_step = dataframe['step'].max()                                  #pegando o último passo
            line_dataframe = dataframe[dataframe['step'] == max_step].copy()    #selecionando as tuplas que contém apenas o último passo (apenas uma linha)
            
            #criando dicinário 
            reformed_data = {
                'Model Name': model_name,
                'Acuraccy Top1': f"{line_dataframe['mean_acc1'].item()*100:.2f}% ± {line_dataframe['std_acc1'].item()*100:.2f}%",
                'Acuraccy Top5': f"{line_dataframe['mean_acc5'].item()*100:.2f}% ± {line_dataframe['std_acc5'].item()*100:.2f}%",
                'Train Loss': f"{line_dataframe['mean_train_loss'].item():.4f} ± {line_dataframe['std_train_loss'].item():.4f}",
                'Val Loss': f"{line_dataframe['mean_val_loss'].item():.4f} ± {line_dataframe['std_val_loss'].item():.4f}"
            }

            data.append(reformed_data)
        
        path = self.save_root/filename
        df_final = pd.DataFrame(data)

        with open(path, 'w') as f:
            f.write('Last Step Metrics: \n')
            f.write(df_final.to_string(index=False))
            

    
class FinetuningMetrics():
    finetune_dataset: str
    raw_data: dict[str, list[Path, Path, Path]]
    save_root: Path

    def __init__(self, raw_data, finetune_dataset, save_root):
        self.finetune_dataset = finetune_dataset
        self.raw_data = raw_data
        self.save_root = Path(save_root)
    
    def _get_model_data(self):
        model_data = {}
        for model_name in self.raw_data.keys():
            miou_rep = []
            for repetition in range(3):
                path = self.raw_data[model_name][repetition]
                with open(path, 'r') as data:
                    metrics = yaml.safe_load(data)
                    miou = metrics['classification']['mIoU']
                    miou_rep.append(miou)
            
            mean = np.mean(miou_rep)
            std = np.std(miou_rep)

            model_data[model_name] = f'{mean:.2f} ± {std:.2f}'
        
        return model_data
    
    def save_miou_table(self):
        '''Recupera os dados processados de mIoU, organiza em uma tabela
        e salva em um arquivo de texto alinhado na pasta root.
        '''
        import pandas as pd

        model_data_dict = self._get_model_data()

        table_rows = list(model_data_dict.items())

        df_miou = pd.DataFrame(table_rows, columns=['Model Name', 'mIoU'])

        save_path = self.save_root / f'{self.finetune_dataset} finetuned Models mIoU'

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(df_miou.to_string(index=False))

if __name__ == '__main__':

    root = Path('/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/logs+checkpoints')
    t_root = root / 'Train'
    
    train_raw_data = {
        'Full Dataset': [],
        '600 images per class': [],
        '100 images per class': []
    }

    f_root = root / 'Finetune'

    f3_raw_data = {
        'Full Dataset': [],
        '600 images per class': [],
        '100 images per class': []
    }

    parihaka_raw_data = {
        'Full Dataset': [],
        '600 images per class': [],
        '100 images per class': []
    }

    for i in ['0', '1', '2']:
        # CORREÇÃO: Mudado de train_root para t_root
        train_raw_data['Full Dataset'].append(t_root / i / 'full' / 'logs' / 'full' / 'imagenet' / 'metrics.csv')
        train_raw_data['600 images per class'].append(t_root / i / 'default' / 'num_classes_1000_per_class_600' / 'logs' / 'metrics.csv')
        train_raw_data['100 images per class'].append(t_root / i / 'default' / 'num_classes_1000_per_class_100' / 'logs' / 'metrics.csv')

        rep_root = f_root / i
       
        # CORREÇÃO: Capturando o arquivo correto de dentro do .glob() usando next()
        f3_raw_data['Full Dataset'].append(next((rep_root / 'full' / 'finetune_f3_N' / 'logs').glob("metrics*.yaml")))
        f3_raw_data['600 images per class'].append(next((rep_root / 'default' / 'num_classes_1000_per_class_600' / 'finetune_f3_N' / 'logs').glob('metrics*.yaml')))
        f3_raw_data['100 images per class'].append(next((rep_root / 'default' / 'num_classes_1000_per_class_100' / 'finetune_f3_N' / 'logs').glob('metrics*.yaml')))
                
        parihaka_raw_data['Full Dataset'].append(next((rep_root / 'full' / 'finetune_seam_ai_N' / 'logs').glob("metrics*.yaml")))
        parihaka_raw_data['600 images per class'].append(next((rep_root / 'default' / 'num_classes_1000_per_class_600' / 'finetune_seam_ai_N' / 'logs').glob('metrics*.yaml')))
        parihaka_raw_data['100 images per class'].append(next((rep_root / 'default' / 'num_classes_1000_per_class_100' / 'finetune_seam_ai_N' / 'logs').glob('metrics*.yaml')))
    
    # Executando a análise de treino
    train_metrics = TrainMetrics(train_raw_data, './data')
    train_metrics.group_plot('acc1', 'step', 100, 'Models Acuraccy top 1', 'steps', 'Acc1 (%)', None)
    
    # CORREÇÃO: Removido ': int' e corrigido 'steps' para 'step' (conforme seu agrupamento)
    train_metrics.individual_plot(['val_loss', 'train_loss'], 'step', 1, 'Val and Train loss x Steps', ncols=3, yscale='log')
    train_metrics.show_last_step_metrics('last_step_metrics.txt')

    # CORREÇÃO: Nome da classe ajustado para FinetuningMetrics e pasta para './data'
    parihaka_metrics = FinetuningMetrics(parihaka_raw_data, 'Seam_Ai', './data')
    parihaka_metrics.save_miou_table()

    # CORREÇÃO: f3_ra_data corrigido para f3_raw_data, nome da classe ajustado e pasta para './data'
    f3_metrics = FinetuningMetrics(f3_raw_data, 'F3', './data')
    f3_metrics.save_miou_table()



    