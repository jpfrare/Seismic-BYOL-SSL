import pandas as pd
import os
import matplotlib.pyplot as plt
import math
from pathlib import Path
import yaml
import numpy as np
import copy
from modelInfo import ModelInfo


class TrainMetrics():
    models: list[ModelInfo]   

    def __init__(self, save_root, models):
        self.raw_data = raw_data
        self.save_root = Path(save_root)
        self.save_root.mkdir(parents= True, exist_ok= True)
    
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
    finetune_dataset: str
    backbone_freeze: str
    

    def __init__(self, raw_data, save_root, infos, finetune_dataset, backbone_freeze):
        self.raw_data = raw_data
        self.save_root = Path(save_root)
        self.save_root.mkdir(parents= True, exist_ok= True)
        self.model_data = copy.deepcopy(infos)
        self.finetune_dataset = finetune_dataset
        self.backbone_freeze = backbone_freeze
    
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
                        info["n_images"] + info["img_offset"],
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
                        info["n_classes"] + info["class_offset"],
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
    

    def plot_heatmap(self, filename):

        rows = sorted(
            {info["n_classes"] for info in self.model_data.values()},
            reverse=True,
        )

        cols = sorted(
            {info["n_images"] for info in self.model_data.values()},
        )

        heatmap = pd.DataFrame(
            np.nan,
            index=rows,
            columns=cols,
        )

        for info in self.model_data.values():
            heatmap.loc[
                info["n_classes"],
                info["n_images"],
            ] = info["mean"]

        fig, ax = plt.subplots(figsize=(8,6))

        im = ax.imshow(
            heatmap.values,
            cmap="viridis",
            aspect="auto",
        )

        # eixo x
        xlabels = []
        for n in cols:
            if n >= 1_000_000:
                xlabels.append(f"{n/1e6:.2f}M")
            elif n >= 1000:
                xlabels.append(f"{n/1000:.0f}k")
            else:
                xlabels.append(str(n))

        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels(xlabels)

        # eixo y
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels(rows)

        ax.set_xlabel("Number of pretraining images")
        ax.set_ylabel("Number of pretraining classes")
        ax.set_title("Mean mIoU")

        # escreve o valor em cada célula
        for i in range(len(rows)):
            for j in range(len(cols)):
                value = heatmap.values[i, j]

                if not np.isnan(value):
                    ax.text(
                        j,
                        i,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )

        fig.colorbar(im, ax=ax, label="mIoU")

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
    

class Metrics():
    models: list[ModelInfo]

    def __init__(self, models: list[ModelInfo]):
        self.models = models
    
    def _save_plot(self, fig, filename: str, save_path: Path):
        save_path.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(
            save_path / filename,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
    
    def individual_plot(
        self,
        title: str,
        save_path: Path,
        y_axis: list[str],
        x_axis: str,
        mul_factor: float = 1,
        ncols: int = 4,
        yscale: str | None = None,
        finetune: bool = False,
        dataset: str | None = None,
        protocol: str | None = None
        ):
        '''plota vários gráficos (cada um referente a um modelo) de um conjunto de métricas escolhido (média sombreado com desvio padrão)'''

        num_models = len(self.models)
        nrows = math.ceil(num_models/ncols) #dado um número de colunas, consegue calcular o número de linhas

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8), sharey= True) #fig -> contém a painel que contém os demais gráficos, axs é a lista de mini gráficos

        axs = axs.flatten() if num_models > 1 else [axs]

        for i in range(num_models):
            ax = axs[i]
            dataframe = self.models[i].get_pretrain_dataframe() if not finetune else self.models[i].get_finetune_dataframe(dataset, protocol)
            if dataframe.empty:
                continue

            for metric in y_axis:
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
            ax.set_title(self.models[i].model_name, fontsize=12, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(fontsize= 9)

            if yscale is not None:
                ax.set_yscale(yscale)
        
        #apaga os quadradinhos não preenchidos
        for i in range(num_models, len(axs)):
            fig.delaxes(axs[i])
        
        fig.suptitle(title, fontsize=16, fontweight='bold')

        self._save_plot(fig, f'{title}_Clean_Grid.png', save_path)
    
    def group_plot(
        self,
        title: str,
        save_path: Path,
        y_axis: str,
        x_axis: str,
        mul_factor: float = 1,
        yscale: str | None =  None,
        finetune: bool = False,
        dataset: str | None = None,
        protocol: str | None = None
        ):
        

