import pandas as pd
import os
import matplotlib.pyplot as plt
import math
from pathlib import Path
import yaml
import numpy as np
import copy

class TrainMetrics():
    raw_data: dict[str, tuple[Path,Path,Path]] #dicioário: nome_modelo -> lista com os caminhos do csv de cada repetição
    save_root: Path                            #onde os dados serão salvos
    model_csvs: dict

    def __init__(self, raw_data, save_root):
        self.raw_data = raw_data
        self.save_root = Path(save_root)
        self._get_model_csvs()
        self.save_root.mkdir(parents= True, exist_ok= True)

    
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
        
        self.model_csvs = model_csv
    
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
            valid_rows = dataframe.dropna(
            subset=[
                'mean_acc1',
                'std_acc1',
                'mean_acc5',
                'std_acc5',
                'mean_val_loss',
                'std_val_loss'
                ]
            )

            line_dataframe = valid_rows.iloc[-1]
            
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
    raw_data: dict[str, list[Path, Path, Path]]
    save_root: Path
    model_data: dict
    model_csvs: dict
    

    def __init__(self, raw_data, save_root, infos):
        self.raw_data = raw_data
        self.save_root = Path(save_root)
        self.save_root.mkdir(parents= True, exist_ok= True)
        self.model_data = copy.deepcopy(infos)

        self._get_model_csvs()
        self._get_model_data()
    
    def _get_model_data(self):
        '''organiza o self.model data da seguinte forma:
            self.model_data['Nome Modelo'] = {'mean' (média do mIoU), 'std' (desvio padrão do mIoU), 
            'n_images' (número de imagens de pré treino), 'n_classes' (número de classes de pré treino), 'color' (cor de plot),
            'label' (nome para ser plotado)}'''

        for model_name in self.raw_data.keys():
            miou_rep = []
            for repetition in range(3):
                path = next(self.raw_data[model_name][repetition].glob("metrics*.yaml"))
                with open(path, 'r') as data:
                    metrics = yaml.safe_load(data)
                    miou = metrics['classification']['mIoU'][0]
                    miou_rep.append(miou)
            
            mean = np.mean(miou_rep)
            std = np.std(miou_rep)

            self.model_data[model_name]['mean'] = mean
            self.model_data[model_name]['std'] = std
        
    
    def _get_model_csvs(self):
        '''organiza o dataframe de modo que self.model_csvs esteja organizado por época em val e train loss dos modelos, média e desvio padrão'''
        model_csv = {}
        for model_name in self.raw_data.keys():
            data = []
            for repetition in range(3):
                csv_path = self.raw_data[model_name][repetition] / 'metrics.csv'
                df_repetition = pd.read_csv(csv_path)
                df_repetition = df_repetition.groupby('epoch').first().reset_index()
                # groupby + first() junta as métricas de treino e validação
                # correspondentes ao mesmo epoch, eliminando as linhas em que
                # apenas train_loss ou apenas val_loss aparecem.
                df_repetition = df_repetition.dropna(subset=['train_loss', 'val_loss'])
                df_repetition['key'] = repetition
                data.append(df_repetition)
            df_model = pd.concat(data)
            model_to_curves = df_model.groupby('epoch').agg(
                mean_val_loss= ('val_loss', 'mean'),
                std_val_loss= ('val_loss', 'std'),
                mean_train_loss = ('train_loss', 'mean'),
                std_train_loss = ('train_loss', 'std')
            ).reset_index()

            model_csv[model_name] = model_to_curves
        
        self.model_csvs = model_csv
    
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

        self._save_plot(fig, f'{title}_Clean_Grid.png')

    def plot_miou_vs_pretrained_images(self, filename):
        """Scatter plot de mIoU em função do número de imagens de pré-treinamento."""

        fig, ax = plt.subplots(figsize=(10, 6))

        # Recupera os dados do Scratch
        scratch = self.model_data["Scratch"]

        # Linha horizontal do Scratch
        ax.axhline(
            y=scratch["mean"],
            color="black",
            linestyle="--",
            linewidth=2,
            label="Scratch baseline",
            zorder=1,
        )

        # Faixa correspondente ao desvio padrão
        ax.axhspan(
            scratch["mean"] - scratch["std"],
            scratch["mean"] + scratch["std"],
            color="black",
            alpha=0.2,
            zorder=0,
        )

        for model_name, info in self.model_data.items():
                if model_name != "Scratch":
                    ax.errorbar(
                        info["n_images"],
                        info["mean"],
                        yerr=info["std"],
                        fmt="o",
                        color=info["color"],
                        markersize=8,
                        capsize=4,
                        elinewidth=1.5,
                        zorder=3,
                        label=model_name,      # <-- legenda usa o nome completo
                    )

        ax.set_xlabel("Number of pretraining images", fontsize=12)
        ax.set_ylabel("Mean mIoU", fontsize=12)
        ax.set_title(filename, fontsize=14)

        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xscale("log")

        # legenda fora do gráfico
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
            frameon=False,
            title="Models"
        )

        self._save_plot(fig, filename)
    
    def plot_miou_vs_classes(self, filename):
        """Scatter plot de mIoU em função do número de classes de pré-treinamento."""

        fig, ax = plt.subplots(figsize=(10, 6))

        # Recupera os dados do Scratch
        scratch = self.model_data["Scratch"]

        # Linha horizontal do Scratch
        ax.axhline(
            y=scratch["mean"],
            color="black",
            linestyle="--",
            linewidth=2,
            label="Scratch baseline",
            zorder=1,
        )

        # Faixa correspondente ao desvio padrão
        ax.axhspan(
            scratch["mean"] - scratch["std"],
            scratch["mean"] + scratch["std"],
            color="black",
            alpha=0.2,
            zorder=0,
        )

        for model_name, info in self.model_data.items():
                if model_name != "Scratch":
                    ax.errorbar(
                        info["n_classes"],
                        info["mean"],
                        yerr=info["std"],
                        fmt="o",
                        color=info["color"],
                        markersize=8,
                        capsize=4,
                        elinewidth=1.5,
                        zorder=3,
                        label=model_name,      # <-- legenda usa o nome completo
                    )

        ax.set_xlabel("Number of pretrained classes", fontsize=12)
        ax.set_ylabel("Mean mIoU", fontsize=12)
        ax.set_title(filename, fontsize=14)

        ax.grid(True, linestyle="--", alpha=0.5)

        # legenda fora do gráfico
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
            frameon=False,
            title="Models"
        )

        self._save_plot(fig, filename)
        
    def save_miou_table(self, filename):
        '''Recupera os dados processados de mIoU, organiza em uma tabela
        e salva em um arquivo de texto alinhado na pasta root.
        '''
        import pandas as pd

        table_rows = []
        for model_name, info in self.model_data.items():
            table_rows.append(
                (
                    model_name,
                    f"{info['mean']:.2f} ± {info['std']:.2f}"
                )
            )

        df_miou = pd.DataFrame(table_rows, columns=['Model Name', 'mIoU'])

        save_path = self.save_root / filename

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(df_miou.to_string(index=False))
    
    def _save_plot(self, fig, filename):
        fig.tight_layout()
        fig.savefig(
            self.save_root / filename,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
    

#--------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    MODEL_INFO = {
        "Full Dataset": {
            "n_images": 1_281_000,
            "n_classes": 1040,
            "color": "orange",
            "label": "FD",
        },

        "600 images per class": {
            "n_images": 600_000,
            "n_classes": 1020,
            "color": "navy",
            "label": "600PC",
        },

        "100 images per class": {
            "n_images": 100_000,
            "n_classes": 1000,
            "color": "blue",
            "label": "100PC",
        },

        "10 images per class": {
            "n_images": 10_000,
            "n_classes": 980,
            "color": "slateblue",
            "label": "10PC",
        },

        "500 Classes": {
            "n_images": 640_500,
            "n_classes": 500,
            "color": "darkred",
            "label": "500C",
        },

        "79 Classes": {
            "n_images": 101_199,
            "n_classes": 79,
            "color": "red",
            "label": "79C",
        },

        "9 Classes": {
            "n_images": 11_529,
            "n_classes": 9,
            "color": "tomato",
            "label": "9C",
        },

        "Level 9": {
            "n_images": 1_281_000,
            "n_classes": 477,
            "color": "green",
            "label": "L9(477C)",
        },

        "Level 6": {
            "n_images": 1_281_000,
            "n_classes": 80,
            "color": "lime",
            "label": "L6(80C)",
        },

        "Level 3": {
            "n_images": 1_281_000,
            "n_classes": 9,
            "color": "palegreen",
            "label": "L3(9C)",
        },

        "Scratch": {
            "n_images": 1,
            "n_classes": 0,
            "color": "black",
            "label": "Scratch",
        },
    }


    root = Path('/petrobr/parceirosbr/spfm/joao.frare/logs+checkpoints_imagenet')
    t_root = root / 'Train'
    f_root = root / 'Finetune'
    
    ipc_experiments = {
        'Full Dataset': [],
        '600 images per class': [],
        '100 images per class': [],
        '10 images per class': []
    }

    class_experiments = {
        '500 Classes': [],
        '79 Classes': [],
        '9 Classes': []
    }

    taxonomic_experiments = {
        '477 Classes': [],
        '80 Classes': [],
        '9 Classes': []
    }

    parihaka_full_finetuning = {
        'Full Dataset': [],
        '600 images per class': [],
        '100 images per class': [],
        '10 images per class': [],
        '500 Classes': [],
        '79 Classes': [],
        '9 Classes': [],
        'Level 9': [],
        'Level 6': [],
        'Level 3': [],
        'Scratch' : []
    }

    parihaka_linear_redout = {
        'Full Dataset': [],
        '600 images per class': [],
        '100 images per class': [],
        '10 images per class': [],
        '500 Classes': [],
        '79 Classes': [],
        '9 Classes': [],
        'Level 9': [],
        'Level 6': [],
        'Level 3': [],
        'Scratch': []
    }

    f3_full_finetuning = {
        'Full Dataset': [],
        '600 images per class': [],
        '100 images per class': [],
        '10 images per class': [],
        '500 Classes': [],
        '79 Classes': [],
        '9 Classes': [],
        'Level 9': [],
        'Level 6': [],
        'Level 3': [],
        'Scratch': []
    }

    f3_linear_redout = {
        'Full Dataset': [],
        '600 images per class': [],
        '100 images per class': [],
        '10 images per class': [],
        '500 Classes': [],
        '79 Classes': [],
        '9 Classes': [],
        'Level 9': [],
        'Level 6': [],
        'Level 3': [],
        'Scratch': []
    }

    parihaka_custom_linear = {
        'Full Dataset': [],
        'Scratch': []
    }

    parihaka_custom_deeplab = {
        'Full Dataset': [],
        'Scratch': []
    }

    f3_custom_linear = {
        'Full Dataset': [],
        'Scratch': []
    }

    f3_custom_deeplab = {
        'Full Dataset': [],
        'Scratch': []
    }

    for i in ['0', '1', '2']:
        # CORREÇÃO: Mudado de train_root para t_root
        ipc_experiments['Full Dataset'].append(t_root / i / 'full' / 'logs' / 'full' / 'imagenet' / 'metrics.csv')
        for ipc in ['600', '100', '10']:
            ipc_experiments[f'{ipc} images per class'].append(t_root / i / 'default' / f'num_classes_1000_per_class_{ipc}' / 'logs' / 'metrics.csv')

        for c in ['500', '79', '9']:
            class_experiments[f'{c} Classes'].append(t_root / i / 'default' / f'num_classes_{c}_per_class_1300' / 'logs' / 'metrics.csv')
        
        taxonomic_experiments['477 Classes'].append(t_root / i / 'taxonomic' / 'top_down_level_9' / 'logs' / 'metrics.csv')
        taxonomic_experiments['80 Classes'].append(t_root / i / 'taxonomic' / 'top_down_level_6' / 'logs' / 'metrics.csv')
        taxonomic_experiments['9 Classes'].append(t_root / i / 'taxonomic' / 'top_down_level_3' / 'logs' / 'metrics.csv')

        for ipc in ['600', '100', '10']:
            parihaka_full_finetuning[f'{ipc} images per class'].append(Path(f_root / i / 'default' / f'num_classes_1000_per_class_{ipc}' / 'finetune_seam_ai_N' / 'full_finetuning_deeplab' / 'logs'))

            parihaka_linear_redout[f'{ipc} images per class'].append(Path(f_root / i / 'default' /  f'num_classes_1000_per_class_{ipc}' / 'finetune_seam_ai_N' / 'full_freeze_linear' / 'logs'))

            f3_full_finetuning[f'{ipc} images per class'].append(Path(f_root / i / 'default' / f'num_classes_1000_per_class_{ipc}' /'finetune_f3_N' / 'full_finetuning_deeplab' / 'logs'))

            f3_linear_redout[f'{ipc} images per class'].append(Path(f_root / i / 'default' /  f'num_classes_1000_per_class_{ipc}' / 'finetune_f3_N' / 'full_freeze_linear' / 'logs'))

        for c in ['500', '79', '9']:
            parihaka_full_finetuning[f'{c} Classes'].append(Path(f_root / i / 'default' / f'num_classes_{c}_per_class_1300' / 'finetune_seam_ai_N' / 'full_finetuning_deeplab' / 'logs'))

            parihaka_linear_redout[f'{c} Classes'].append(Path(f_root / i / 'default' /  f'num_classes_{c}_per_class_1300' / 'finetune_seam_ai_N' / 'full_freeze_linear' / 'logs'))
            
            f3_full_finetuning[f'{c} Classes'].append(Path(f_root / i / 'default' / f'num_classes_{c}_per_class_1300' /'finetune_f3_N' / 'full_finetuning_deeplab' / 'logs'))

            f3_linear_redout[f'{c} Classes'].append(Path(f_root / i / 'default' /  f'num_classes_{c}_per_class_1300' / 'finetune_f3_N' / 'full_freeze_linear' / 'logs'))
            
        
        for level in ['9', '6', '3']:
            parihaka_full_finetuning[f'Level {level}'].append(Path(f_root / i / 'taxonomic' / f'top_down_level_{level}' / 'finetune_seam_ai_N' / 'full_finetuning_deeplab' / 'logs'))
            
            parihaka_linear_redout[f'Level {level}'].append(Path(f_root / i / 'taxonomic' /  f'top_down_level_{level}' / 'finetune_seam_ai_N' / 'full_freeze_linear' / 'logs'))
            
            f3_full_finetuning[f'Level {level}'].append(Path(f_root / i / 'taxonomic' / f'top_down_level_{level}' /'finetune_f3_N' / 'full_finetuning_deeplab' / 'logs'))
            
            f3_linear_redout[f'Level {level}'].append(Path(f_root / i / 'taxonomic' /  f'top_down_level_{level}' / 'finetune_f3_N' / 'full_freeze_linear' / 'logs'))
            
        
        parihaka_full_finetuning[f'Full Dataset'].append(Path(f_root / i / 'full' / 'finetune_seam_ai_N' / 'full_finetuning_deeplab' / 'logs'))

        parihaka_linear_redout[f'Full Dataset'].append(Path(f_root / i / 'full' / 'finetune_seam_ai_N' / 'full_freeze_linear' / 'logs'))


        f3_full_finetuning[f'Full Dataset'].append(Path(f_root / i / 'full' / 'finetune_f3_N' / 'full_finetuning_deeplab' / 'logs'))

        f3_linear_redout[f'Full Dataset'].append(Path(f_root / i / 'full' / 'finetune_f3_N' / 'full_freeze_linear' / 'logs'))


        parihaka_full_finetuning[f'Scratch'].append(Path(f_root / i / 'scratch' / 'finetune_seam_ai_N' / 'full_finetuning_deeplab' / 'logs'))

        parihaka_linear_redout[f'Scratch'].append(Path(f_root / i / 'scratch' / 'finetune_seam_ai_N' / 'full_freeze_linear' / 'logs'))


        f3_full_finetuning[f'Scratch'].append(Path(f_root / i / 'scratch' / 'finetune_f3_N' / 'full_finetuning_deeplab' / 'logs'))

        f3_linear_redout[f'Scratch'].append(Path(f_root / i / 'scratch' / 'finetune_f3_N' / 'full_freeze_linear' / 'logs'))


    
    #--------------------------------------------------------------pre-train------------------------------------------------------------------------------
    ipc_metrics = TrainMetrics(ipc_experiments, Path('./data')/'pretrain')
    ipc_metrics.group_plot('acc1', 'step', 100, 'Models Acuraccy top 1', 'steps', 'Acc1 (%)', None)
    ipc_metrics.individual_plot(['val_loss', 'train_loss'], 'step', 1, 'per class: Val and Train loss x Steps', ncols= 2, yscale= 'log')
    ipc_metrics.show_last_step_metrics('per_class_last_step_metrics.txt')
    
    class_experiments_metrics = TrainMetrics(class_experiments, Path('./data')/'pretrain')
    class_experiments_metrics.individual_plot(['val_loss', 'train_loss'], 'step', 1, 'random classes: Val and Train loss x Steps', ncols= 2, yscale= 'log')
    class_experiments_metrics.show_last_step_metrics('random_classes_last_step_metrics.txt')

    taxonomic_experiments_metrics = TrainMetrics(taxonomic_experiments, Path('./data')/'pretrain')
    taxonomic_experiments_metrics.individual_plot(['val_loss', 'train_loss'], 'step', 1, 'taxonomic classes: Val and Train loss x Steps', ncols= 2, yscale= 'log')
    taxonomic_experiments_metrics.show_last_step_metrics('taxonomic_classes_last_step_metrics.txt')


    #---------------------------------------------------------finetuning-----------------------------------------------------------------------------------

    parihaka_full_finetuning_metrics = FinetuningMetrics(parihaka_full_finetuning, Path('./data')/'finetune', MODEL_INFO)
    parihaka_full_finetuning_metrics.save_miou_table('Models on Full Finetuning - Parihaka')
    parihaka_full_finetuning_metrics.individual_plot(['val_loss', 'train_loss'], 'epoch', 1, 'Full Finetuning - Parihaka', ncols= 3)
    parihaka_full_finetuning_metrics.plot_miou_vs_pretrained_images('Full Finetuning - Parihaka mIoU x Number of Images')
    parihaka_full_finetuning_metrics.plot_miou_vs_classes('Full Finetuning - Parihaka mIoU x Number of Classes')

    parihaka_linear_redout_metrics = FinetuningMetrics(parihaka_linear_redout, Path('./data')/'finetune', MODEL_INFO)
    parihaka_linear_redout_metrics.save_miou_table('Models on Linear Redout - Parihaka')
    parihaka_linear_redout_metrics.individual_plot(['val_loss', 'train_loss'], 'epoch', 1, 'Linear Redout - Parihaka', ncols= 3)
    parihaka_linear_redout_metrics.plot_miou_vs_pretrained_images('Linear Redout - Parihaka mIoU x Number of Images')
    parihaka_linear_redout_metrics.plot_miou_vs_classes('Linear Redout - Parihaka mIoU x Number of Classes')

    f3_full_finetuning_metrics = FinetuningMetrics(f3_full_finetuning, Path('./data')/'finetune', MODEL_INFO)
    f3_full_finetuning_metrics.save_miou_table('Models on Full Finetuning - F3')
    f3_full_finetuning_metrics.individual_plot(['val_loss', 'train_loss'], 'epoch', 1, 'Full Finetuning - F3', ncols= 3)
    f3_full_finetuning_metrics.plot_miou_vs_pretrained_images('Full Finetuning - F3 mIoU x Number of Images')
    f3_full_finetuning_metrics.plot_miou_vs_classes('Full Finetuning - F3 mIoU x Number of Classes')

    f3_linear_redout_metrics = FinetuningMetrics(f3_linear_redout, Path('./data')/'finetune', MODEL_INFO)
    f3_linear_redout_metrics.save_miou_table('Models on Linear Redout - F3')
    f3_linear_redout_metrics.individual_plot(['val_loss', 'train_loss'], 'epoch', 1, 'Linear Redout - F3', ncols= 3)
    f3_linear_redout_metrics.plot_miou_vs_pretrained_images('Linear Redout - F3 mIoU x Number of Images')
    f3_linear_redout_metrics.plot_miou_vs_classes('Linear Redout - F3 mIoU x Number of Classes')




    