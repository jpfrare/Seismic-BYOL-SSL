import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt


root = "/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/checkpoints/logs_vinicius/pretrain"

folder_name = 'data' #pasta que armazena os dados
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

cores = {10: 'red', 100: 'orange', 600: 'green', 1300: 'blue'} #cores do gráfico
plt.figure(figsize=(12, 7)) #criação do gráfico das curvas sobrepostas
ax = plt.gca()

data = []

for per_class in [10, 100, 600, 1300]:
    r = []
    for repetition in range(3):
        model = f'V{repetition}_pretrain_imagenet_{per_class}_per_class/imagenet'
        path = f'{root}/{repetition}/{model}/{model}/metrics.csv'

        df = pd.read_csv(path)                          #lendo csv com os dados
        df = df[['epoch', 'train_acc', 'train_loss', 'val_acc1', 'val_acc5', 'val_loss']]

        df = df.groupby('epoch').mean().reset_index()   #agrupa por épocas -> onde todos os atributos estão presentes
        df['val_gap'] = df['train_loss'] - df['val_loss']
        df['acc_gap'] = df['train_acc'] - df['val_acc1']
        df['key'] = repetition
        r.append(df)

    df_total = pd.concat(r)

    #montando a parte da tabela
    df_table = df_total.groupby('key').agg(
       mean_val_gap= ('val_gap', 'mean'),
       mean_acc_gap= ('acc_gap', 'mean'),
       min_val_loss= ('val_loss', 'min'),
       max_acc5= ('val_acc5', 'max'),
       max_acc1= ('val_acc1', 'max')
    ).reset_index()
    
    df_table['name'] = f'{per_class}_images_per_class'
    df_table = df_table.groupby('name').agg(
        val_gap= ('mean_val_gap', 'mean'),
        std_val_gap= ('mean_val_gap', 'std'),

        val_acc_gap= ('mean_acc_gap', 'mean'),
        std_val_acc_gap= ('mean_acc_gap', 'std'),

        min_val_loss= ('min_val_loss', 'mean'),
        std_min_val_loss= ('min_val_loss', 'std'),

        max_acc5= ('max_acc5', 'mean'),
        std_max_acc5= ('max_acc5', 'std'),

        max_acc1= ('max_acc1', 'mean'),
        std_max_acc1= ('max_acc1', 'std')
    )

    df_table['Val Gap'] = df_table.apply(
        lambda x: f"{x['val_gap']:.2f} ± {x['std_val_gap']:.2f}", axis= 1)

    df_table['Min Val Loss'] = df_table.apply(
        lambda x: f"{x['min_val_loss']:.2f} ± {x['std_min_val_loss']:.2f}", axis= 1)

    df_table['Val Acc Gap'] = df_table.apply(
        lambda x: f"{x['val_acc_gap']:.2f} ± {x['std_val_acc_gap']:.2f}", axis= 1)

    df_table['Top5'] = df_table.apply(
        lambda x: f"{x['max_acc5']:.2f} ± {x['std_max_acc5']:.2f}", axis= 1)

    df_table['Top1'] = df_table.apply(
        lambda x: f"{x['max_acc1']:.2f} ± {x['std_max_acc1']:.2f}", axis= 1)

    data.append(df_table[['Val Gap', 'Min Val Loss', 'Val Acc Gap', 'Top5', 'Top1']])

    
    #montagem da curva
    val_loss_axis = df_total.groupby('epoch').agg(
        mean_val_loss= ('val_loss', 'mean'),
        std_val_loss= ('val_loss', 'std')
    ).reset_index()

    epochs = val_loss_axis['epoch']
    mean = val_loss_axis['mean_val_loss']
    std = val_loss_axis['std_val_loss']

    plt.plot(epochs, mean, label=f'{per_class} per class', color=cores[per_class], linewidth=2)    #linha principal
    plt.fill_between(epochs, mean - std, mean + std, color=cores[per_class], alpha=0.2)


#montagem do gráfico
plt.title("Comparison of Validation Loss: ImageNet Subsets", fontsize=14, fontweight='bold')
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(title="Datasets", loc='upper right')

plt.savefig("data/combined_loss_curves.png", bbox_inches='tight', dpi=300)
plt.close()

#montagem da tabela
final_df = pd.concat(data)

df_imagem = final_df.round(4)
fig, ax = plt.subplots(figsize=(12, 6)) # Ajuste o tamanho conforme necessário
ax.axis('off') # Esconde os eixos do gráfico

tabela = ax.table(cellText=df_imagem.values, 
                 colLabels=df_imagem.columns, 
                 rowLabels=df_imagem.index,
                 loc='center', 
                 cellLoc='center')

tabela.auto_set_font_size(False)
tabela.set_fontsize(10)
tabela.scale(1.2, 1.5) 

plt.title("Imagenet Results per Image per Class", fontsize=14)
plt.savefig("data/train_data.png", bbox_inches='tight', dpi=300)
final_df.to_csv("data/results_per_image_per_class.csv")

print('sucesso total')


