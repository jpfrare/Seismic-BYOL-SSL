import pandas as pd
import os
import matplotlib.pyplot as plt
import math
from pathlib import Path
import yaml
import numpy as np
import copy
from modelInfo import ModelInfo
import seaborn as sns

class Metrics():
    models: list[ModelInfo]

    def __init__(self, models: list[ModelInfo]):
        self.models = models
    
    def __add__(self, other):
        return Metrics(self.models + other.models)
    
    def add_model(self, model: ModelInfo):
        self.models.append(model)
    
    def _save_plot(self, fig, filename: str, save_path: Path):
        save_path.mkdir(parents=True, exist_ok=True)
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
        protocol: str | None = None):

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

            if yscale is not None:
                ax.set_yscale(yscale)
        
        handles, labels = axs[0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.98),
            frameon=False,
            fontsize=12
        )

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        #apaga os quadradinhos não preenchidos
        for i in range(num_models, len(axs)):
            fig.delaxes(axs[i])

        self._save_plot(fig, f'{title}.png', save_path)
    
    def group_plot(
        self,
        title: str,
        save_path: Path,
        x_axis: str,
        x_axis_label: str,
        y_axis: str,
        y_axis_label: str,
        min_y: float | None = None,
        max_y: float | None = None,
        y_step: float | None = None,
        mul_factor: float = 1,
        yscale: str | None =  None,
        finetune: bool = False,
        dataset: str | None = None,
        protocol: str | None = None):

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
        
            plt.xlabel(x_axis_label, fontsize= 12)
            plt.ylabel(y_axis_label, fontsize= 12)

        if yscale is not None:
            plt.yscale(yscale)
            
        if min_y is not None and max_y is not None:
            plt.ylim(min_y, max_y)

        if y_step is not None:
            if min_y is None or max_y is None:
                ymin, ymax = plt.ylim()
                ticks = np.arange(ymin, ymax + y_step, y_step)
            else:
                ticks = np.arange(min_y, max_y + y_step, y_step)
            
            plt.yticks(ticks)
        
        plt.legend(
            loc='best',      # ou 'upper right', 'lower left', etc.
            fontsize=10,
            frameon=True,
            ncol=2           # opcional, divide em 2 colunas se houver muitos modelos
        )
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / f'{title}.png'
        plt.savefig(save_path, dpi= 300, bbox_inches='tight')
        plt.close()
    
    def plot_heatmap(
    self,
    title: str,
    save_path: Path,
    dataset: str | None = None,
    protocol: str | None = None,
    pretrain: bool = False,
    cmap: str = "rocket",
    ):

        if pretrain:
            models_to_plot = [model for model in self.models if not (model.scratch or model.torchPretrained)]
        else:
            models_to_plot = [model for model in self.models if not model.torchPretrained]

        rows = sorted(
            {model.pretrained_classes for model in models_to_plot},
            reverse=True,
        )

        cols = sorted(
            {model.pretrained_images for model in models_to_plot},
        )


        df_mean = pd.DataFrame(
            np.nan,
            index=rows,
            columns=cols,
        )

        df_std = pd.DataFrame(
            np.nan,
            index=rows,
            columns=cols,
        )

        min_metric = np.inf if not pretrain else None
        max_metric = -np.inf if not pretrain else None

        #preenchendo os dataframes e definindo o mínimo e máximo do mIoU para a escala entre os protocolos
        for model in models_to_plot:
            # ignora modelos que não pertencem à malha
            if pretrain:
                mean, std = model.get_pretrain_top1acc()
                mean *= 100
                std *= 100

            else:

                for p in model.protocols:

                    miou, std_miou = model.get_finetune_miou(dataset, p)
                    miou *= 100
                    std_miou *= 100

                    if p == protocol:
                        mean = miou
                        std = std_miou

                    max_metric = max(max_metric, miou)
                    min_metric = min(min_metric, miou)

            df_mean.loc[
                model.pretrained_classes,
                model.pretrained_images,
            ] = mean

            df_std.loc[
                model.pretrained_classes,
                model.pretrained_images,
            ] = std

        fig, ax = plt.subplots(figsize=(8, 6))

        df_annot = df_mean.copy().astype(object)

        for i in range(len(rows)):
            for j in range(len(cols)):
                mean = df_mean.iloc[i,j]
                std = df_std.iloc[i,j]

                if not np.isnan(mean):
                    df_annot.iloc[i,j] = f'{mean:.2f}%\n±{std:.2f}%'
                else:
                    df_annot.iloc[i,j] = ''
        
        def format_number(n):
            if n >= 1_000_000:
                return f"{n / 1_000_000:.2f}M"
            elif n >= 1_000:
                return f"{n / 1_000:.0f}k"
            return str(n)
        
        x_labels = [format_number(n) for n in cols]

        sns.heatmap(
            data= df_mean,
            vmin = min_metric,
            vmax= max_metric,
            cmap= cmap,
            annot= df_annot,
            xticklabels= x_labels,
            ax= ax,
            linecolor= 'white',
            cbar_kws = {
                'label': 'Top-1 Accuracy (%)' if pretrain else 'mIoU (%)'
            },
            fmt= ''
        )

        ax.set(xlabel= 'Number of Pretrained Images', ylabel= 'Number of Pretrained Classes')
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()

        
        fig.tight_layout()
        self._save_plot(
            fig,
            f"{title}.png",
            save_path,
        )
    
    def save_miou_table(self, filename, save_path, dataset, protocol):
        '''Recupera os dados processados de mIoU, organiza em uma tabela
        e salva em um arquivo de texto alinhado na pasta root.
        '''

        table_rows = []
        for model in self.models:
            mean, std = model.get_finetune_miou(dataset, protocol)
            table_rows.append(
                (
                    model.model_name,
                    f"{mean:.2f} ± {std:.2f}"
                )
            )

        df_miou = pd.DataFrame(table_rows, columns=['Model Name', 'mIoU'])

        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / filename

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(df_miou.to_string(index=False))
        

    def plot_taxonomic_x_default(
    self,
    title: str,
    save_path: Path,
    dataset: str | None = None,
    protocol: str | None = None,
    pretrain: bool = False,
    ):

        classes = sorted({model.pretrained_classes for model in self.models})

        default_mean, default_std = [], []
        tax_mean, tax_std = [], []
        labels = []

        for c in classes:

            default_model = next(
                (
                    m for m in self.models
                    if m.reduction_mode == "default"
                    and m.pretrained_classes == c
                ),
                None,
            )

            tax_model = next(
                (
                    m for m in self.models
                    if m.reduction_mode == "taxonomic"
                    and m.pretrained_classes == c
                ),
                None,
            )

            if default_model is None or tax_model is None:
                continue

            mean, std = (
                default_model.get_pretrain_top1acc()
                if pretrain
                else default_model.get_finetune_miou(dataset, protocol)
            )

            default_mean.append(mean * 100)
            default_std.append(std * 100)

            mean, std = (
                tax_model.get_pretrain_top1acc()
                if pretrain
                else tax_model.get_finetune_miou(dataset, protocol)
            )

            tax_mean.append(mean * 100)
            tax_std.append(std * 100)

            labels.append(str(c))

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(9,5))

        error_style = dict(
            lw=1,
            capsize=4,
            capthick=1,
            ecolor="black"
        )

        bars_default = ax.bar(
            x - width/2,
            default_mean,
            width,
            yerr=default_std,
            color="#4C72B0",
            label="Default",
            error_kw=error_style,
        )

        bars_tax = ax.bar(
            x + width/2,
            tax_mean,
            width,
            yerr=tax_std,
            color="#DD8452",
            label="Taxonomic",
            error_kw=error_style,
        )

        ax.set_ylim(0, 100)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)

        ax.set_xlabel(
            "Number of Pretraining Classes",
            fontsize=14,
        )

        ax.set_ylabel(
            "Top-1 Accuracy (%)" if pretrain else "mIoU (%)",
            fontsize=14,
        )

        ax.tick_params(axis="y", labelsize=12)

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.25,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(
            frameon=False,
            fontsize=12,
            loc="best",
        )

        plt.tight_layout()
        self._save_plot(
            fig,
            f"{title}.png",
            save_path,
        )


