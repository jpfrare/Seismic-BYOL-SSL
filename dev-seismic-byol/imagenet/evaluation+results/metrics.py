import pandas as pd
import os
import matplotlib.pyplot as plt
import math
from pathlib import Path
import yaml
import numpy as np
import copy
from modelInfo import ModelInfo

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

        self._save_plot(fig, f'{title}.png', save_path)
    
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

        '''plot de apenas um gráfico da única variável escolhida para todos os modelos (média sombreado com desvio padrão)
        mul_factor é o fator multiplicativo para a variável y'''

        plt.figure(figsize=(16,6))

        for model in self.models:
            dataframe = model.get_pretrain_dataframe() if not finetune else model.get_finetune_dataframe(dataset, protocol)
            if dataframe.empty:
                continue

            mean = dataframe[f'mean_{y_axis}']*mul_factor
            std = dataframe[f'std_{y_axis}']*mul_factor

            plt.plot(
                dataframe[x_axis],
                mean,
                linewidth= 2,
                label= model.model_name
            )

            plt.fill_between(
                dataframe[x_axis],
                mean - std,
                mean + std,
                alpha = 0.15,
                zorder = 2
            )
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(x_axis, fontsize= 12)
        plt.ylabel(y_axis, fontsize= 12)

        if yscale is not None:
            plt.yscale(yscale)
        
        save_path = save_path / f'{title}.png'
        plt.savefig(save_path, dpi= 300, bbox_inches='tight')
        plt.close()
    
    def plot_heatmap(
    self,
    title: str,
    save_path: Path,
    dataset: str,
    protocol: str,
    ):

    rows = sorted(
        {model.pretrained_classes for model in self.models},
        reverse=True,
    )

    cols = sorted(
        {model.pretrained_images for model in self.models},
    )

    heatmap = pd.DataFrame(
        np.nan,
        index=rows,
        columns=cols,
    )

    for model in self.models:

        # ignora modelos que não pertencem à malha
        if model.scratch or model.torchPretrained:
            continue

        mean, std = model.get_finetune_miou(dataset, protocol)

        heatmap.loc[
            model.pretrained_classes,
            model.pretrained_images,
        ] = mean

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        heatmap.values,
        cmap="OrRd",
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
    ax.set_title(title)

    # escreve o valor nas células
    for i in range(len(rows)):
        for j in range(len(cols)):
            value = heatmap.iloc[i, j]

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

    fig.colorbar(im, ax=ax, label="Mean mIoU")

    self._save_plot(
        fig,
        f"{title}.png",
        save_path,
    )

