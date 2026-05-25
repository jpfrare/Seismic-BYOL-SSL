import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt

root = "/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/logs+checkpoints/Train"

data = []

for repetition in range(3):
    path = f'{root}/{repetition}/full/logs/full/imagenet/metrics.csv'
    df = pd.read_csv(path)

    df = df.groupby('step').first().reset_index()
    df['key'] = repetition
    data.append(df)

df_full = pd.concat(data)
df_to_curves = df_full.groupby('step').agg(
    mean_val_loss= ('val_loss', 'mean'),
    std_val_loss= ('val_loss', 'std'),
    mean_train_loss_epoch = ('train_loss_epoch', 'mean'),
    std_train_loss_epoch = ('train_loss_epoch', 'std'),
    mean_train_loss_step = ('train_loss_step', 'mean'),
    std_train_loss_step = ('train_loss_step', 'std'),
    mean_acc_1= ('val_acc1', 'mean'),
    std_acc_1= ('val_acc1', 'std'),
    mean_acc5= ('val_acc5', 'mean'),
    std_acc5= ('val_acc5', 'std') 
).reset_index()

df_steps = df_to_curves.dropna(subset=['mean_train_loss_step'])
df_epochs = df_to_curves.dropna(subset=['mean_train_loss_epoch', 'mean_val_loss'])

# 2. Criar a figura com 2 subplots lado a lado
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# =====================================================================
# GRÁFICO 1: Histórico de Loss (Ordem de camadas ajustada + Escala LOG)
# =====================================================================

# 1. Nuvem de treino por STEP ao fundo (zorder=1, alpha bem baixo para não poluir)
ax1.plot(
    df_steps['step'], df_steps['mean_train_loss_step'], 
    color='tab:blue', alpha=0.60, linewidth=1.5, label='Train Loss (Batches)', zorder=1
)

# 2. Linha de validação no meio (zorder=2)
ax1.plot(
    df_epochs['step'], df_epochs['mean_val_loss'], 
    color='tab:red', marker='o', markersize=4, linewidth=2, label='Val Loss', zorder=2
)
ax1.fill_between(
    df_epochs['step'], 
    df_epochs['mean_val_loss'] - df_epochs['std_val_loss'], 
    df_epochs['mean_val_loss'] + df_epochs['std_val_loss'], 
    alpha=0.15, color='tab:red', zorder=2
)

# 3. Linha de treino por ÉPOCA EM DESTAQUE NO TOPO (zorder=3, linewidth mais grossa)
ax1.plot(
    df_epochs['step'], df_epochs['mean_train_loss_epoch'], 
    color='tab:orange', linewidth=3, label='Train Loss (Época)', zorder=3
)
ax1.fill_between(
    df_epochs['step'], 
    df_epochs['mean_train_loss_epoch'] - df_epochs['std_train_loss_epoch'], 
    df_epochs['mean_train_loss_epoch'] + df_epochs['std_train_loss_epoch'], 
    alpha=0.25, color='tab:blue', zorder=3
)

# Ativação da escala LOG no eixo Y (Isso resolve o esmagamento)
ax1.set_yscale('log')

ax1.set_title('Curvas de Convergência (Loss - Escala Log)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Steps', fontsize=11)
ax1.set_ylabel('Loss (Log)', fontsize=11)
ax1.grid(True, which="both", linestyle='--', alpha=0.5)
ax1.legend(loc='upper right')

# =====================================================================
# GRÁFICO 2: Evolução das Acurácias (Top-1 e Top-5 por Época)
# =====================================================================

# Curva Top-1
ax2.plot(
    df_epochs['step'], df_epochs['mean_acc_1'] * 100, 
    color='tab:green', marker='s', markersize=4, linewidth=2, label='Val Acc Top-1'
)
ax2.fill_between(
    df_epochs['step'], 
    (df_epochs['mean_acc_1'] - df_epochs['std_acc_1']) * 100, 
    (df_epochs['mean_acc_1'] + df_epochs['std_acc_1']) * 100, 
    alpha=0.15, color='tab:green'
)

# Declarando a série limpa antes do plot para evitar erro de sintaxe
std_acc5_clean = df_to_curves.loc[df_epochs.index, 'std_acc5']

# Curva Top-5
ax2.plot(
    df_epochs['step'], df_epochs['mean_acc5'] * 100, 
    color='tab:orange', marker='^', markersize=4, linewidth=2, label='Val Acc Top-5'
)
ax2.fill_between(
    df_epochs['step'], 
    (df_epochs['mean_acc5'] - std_acc5_clean) * 100, 
    (df_epochs['mean_acc5'] + std_acc5_clean) * 100, 
    alpha=0.15, color='tab:orange'
)

ax2.set_title('Métricas de Validação (Acurácia)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Steps', fontsize=11)
ax2.set_ylabel('Acurácia (%)', fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='lower right')

# =====================================================================
# SALVAMENTO DO ARQUIVO (A linha que estava faltando!)
# =====================================================================
plt.tight_layout()
os.makedirs('data', exist_ok=True)
plt.savefig('data/analise_compartimentada_imagenet.png', dpi=300)
plt.close()

# =====================================================================
# CÁLCULO DOS RESULTADOS ESTATÍSTICOS FINAIS
# =====================================================================
ultimo_step = df_epochs['step'].max()

df_final = df_full[df_full['step'] == ultimo_step]

final_stats = df_final.agg(
    mean_top1=('val_acc1', 'mean'),
    std_top1=('val_acc1', 'std'),
    mean_top5=('val_acc5', 'mean'),
    std_top5=('val_acc5', 'std')
)

print("="*45)
print(f"RESULTADOS FINAIS DO EXPERIMENTO (STEP {ultimo_step})")
print("="*45)
print(f"Acurácia Top-1 Média: {final_stats.loc['mean_top1', 'val_acc1']*100:.2f}% ± {final_stats.loc['std_top1', 'val_acc1']*100:.2f}%")
print(f"Acurácia Top-5 Média: {final_stats.loc['mean_top5', 'val_acc5']*100:.2f}% ± {final_stats.loc['std_top5', 'val_acc5']*100:.2f}%")
print("="*45)