import pandas as pd
import os
import matplotlib.pyplot as plt
import math
from pathlib import Path

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
    
