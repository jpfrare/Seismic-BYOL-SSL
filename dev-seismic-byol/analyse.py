import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os



def group_metrics_by_cap(csv_path, train_data_filter, metric="acc"):
    df = pd.read_csv(csv_path)

    # Filtrar por train_data

    # Converter cap para número (ex: "50%" -> 50)
    df["cap"] = df["cap"].str.rstrip('%').astype(float)

    # Agrupar por pretrain_data e cap
    grouped = df.groupby(["pretrain_data", "cap", "train_data"])[metric].agg(['mean', 'std']).reset_index()
    grouped.to_csv("new_metrics.csv", index=False)
    
    grouped = grouped[grouped["train_data"] == train_data_filter].copy()    
    return grouped


def plot_metric_by_cap(grouped_df, metric="acc"):
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # Plot para cada linha de pretrain_data
    for pretrain_data in grouped_df["pretrain_data"].unique():
        data = grouped_df[grouped_df["pretrain_data"] == pretrain_data]
        plt.errorbar(
            data["cap"], data["mean"], yerr=data["std"], label=pretrain_data,
            capsize=4, marker='o', linestyle='-', linewidth=2
        )

    plt.title(f"{metric} por cap (média ± std)")
    plt.xlabel("Cap (%)")
    plt.ylabel(metric)
    plt.legend(title="Pré-treino")
    plt.tight_layout()
    plt.show()


def barplot_from_df(
    df,
    x="cap",
    y="mean",
    err="std",
    hue="pretrain_data",
    title=None,
    save_path=None,
    pretrains=None,
    caps=None,
    train_data=None
):

    import matplotlib.pyplot as plt
    import seaborn as sns

    df = df.copy()
    df[err] = df[err].fillna(0)

    if pretrains is not None:
        df = df[df[hue].isin(pretrains)]
    if caps is not None:
        df = df[df[x].isin(caps)]

    if df.empty:
        print("Nenhum dado após aplicar filtros de pretrains e caps.")
        return

    df = df.sort_values(by=[x, hue])
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    ax = sns.barplot(data=df, x=x, y=y, hue=hue, errorbar=None)

    # Adiciona barras de erro corretamente por grupo
    i = 0
    for container in ax.containers:
        for bar in container:
            if i >= len(df):
                break
            height = bar.get_height()
            x_pos = bar.get_x() + bar.get_width() / 2
            std_val = df.iloc[i][err]
            ax.errorbar(x_pos, height, yerr=std_val, color='black', capsize=5, fmt='none')
            i += 1

    plt.xlabel("Cap (%)")
    plt.ylabel(y)
    plot_title = title or f"{y} por {x} agrupado por {hue}"
    if train_data:
        plot_title += f": {train_data}"
    plt.title(plot_title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Gráfico salvo em: {save_path}")
    else:
        plt.show()


def plot_from_df(df, x="cap", y="mean", err="std", hue="pretrain_data", title=None, save_path=None, log_scale=False, pretrains=None, caps=None, train_data=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = df.copy()
    df[err] = df[err].fillna(0)

    if pretrains is not None:
        df = df[df[hue].isin(pretrains)]
    if caps is not None:
        df = df[df[x].isin(caps)]

    df = df.sort_values(by=[hue, x])

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    for pretrain_data, group in df.groupby(hue):
        plt.errorbar(
            group[x], group[y], yerr=group[err],
            label=pretrain_data, capsize=4,
            marker='o', linestyle='-', linewidth=2
        )

    plt.xlabel("Cap (%)")
    plt.ylabel(y)
    plot_title = title or f"{y} vs {x}"
    if train_data:
        plot_title += f": {train_data}"
    plt.title(plot_title)
    if log_scale:
        plt.xscale("log")
    plt.legend(title=hue)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Gráfico salvo em: {save_path}")
    else:
        plt.show()
        
        
finetune_list = ['f3', 'f3_N', 'seam_ai', 'seam_ai_N']

all_cap_list = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 13.0, 22.0, 36.0, 60.0, 100.0]
small_caps = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]

metric = "mIoU"

seismic_pretrain = ["f3_N", "f3", "seam_ai_N", "seam_ai", "both_N", "both", "a700"]

all_pretrain_list = ["f3_N", "f3", "seam_ai_N", "seam_ai", "both_N", "both", "a700", "coco", "imagenet", "sup"]

csv_path = "eval_metrics.csv"

old = False

if old:
    os.makedirs("outputs/old", exist_ok=True)
    all_cap_list = [1.0, 10.0, 50.0, 100.0]
    small_caps = [1.0, 10.0]

else:
    os.makedirs("outputs/new", exist_ok=True)


for train_data in finetune_list:

    if old:    
        metrics = pd.read_csv("old_metrics.csv")
        metrics = metrics[metrics["train_data"] == train_data].copy()
    else:
        metrics = group_metrics_by_cap(csv_path=csv_path, train_data_filter=train_data, metric=metric)

    # print(metrics)

    save_path = f'outputs/old/{train_data}' if old else f'outputs/new/{train_data}'

    # plot_from_df(metrics, 
    #             save_path=f"{save_path}_lines.png", 
    #             log_scale=True,
    #             pretrains=seismic_pretrain,
    #             caps=cap_list,
    #             train_data=train_data)

    # barplot_from_df(metrics,
    #             save_path=f"{save_path}_bars.png", 
    #             pretrains=seismic_pretrain,
    #             caps=cap_list,
    #             train_data=train_data)
    
    plot_from_df(metrics, 
                save_path=f"{save_path}_lines.png", 
                log_scale=True,
                pretrains=all_pretrain_list,
                caps=all_cap_list,
                train_data=train_data)

    barplot_from_df(metrics,
                save_path=f"{save_path}_bars.png", 
                pretrains=all_pretrain_list,
                caps=all_cap_list,
                train_data=train_data)
    
    barplot_from_df(metrics,
                save_path=f"{save_path}_first_bars.png", 
                pretrains=all_pretrain_list,
                caps=small_caps,
                train_data=train_data)
    

    plot_from_df(metrics, 
                save_path=f"{save_path}_filtered_lines.png", 
                log_scale=True,
                pretrains=[train_data, 'sup', 'coco', 'imagenet', 'a700', 'both_N' if '_N' in train_data else 'both'],
                caps=all_cap_list,
                train_data=train_data)

