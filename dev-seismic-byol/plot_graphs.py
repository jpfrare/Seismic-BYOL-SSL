import pandas as pd
import numpy as np

# Caminho para o arquivo CSV
csv_file = 'final_results_modules.csv'

# Leitura do arquivo CSV em um DataFrame
df = pd.read_csv(csv_file)


# Convertendo os valores True e False na coluna 'modules' para 1 e 0
df['modules'] = df['modules'].apply(lambda x: ''.join(['1' if val else '0' for val in eval(x)]))

# Exibindo o DataFrame
# print(df.columns)
# print(sorted(df['modules'].unique()))

# Duplicar o último ponto ('11111') com nome diferente
last_mask = df['modules'] == '11111'
last_rows = df[last_mask].copy()
last_rows['modules'] = '11111*'   # novo nome

# Concatenar no DataFrame original
df = pd.concat([df, last_rows], ignore_index=True)


# Agrupando os valores por 'pretrain_data', 'finetune_data' e 'cap'
grouped = df.groupby(['pretrain_data', 'finetune_data', 'cap', 'modules'])

# Calculando a média e o desvio padrão apenas para a coluna 'mIoU'
result = grouped['mIoU'].agg(['mean', 'std'])

# Exibindo o resultado
# print(result)

print(result)

import matplotlib.pyplot as plt

# Ordem específica para os módulos
# module_order = ['00000', '00001', '00011', '00111', '01111', '11111', '11110', '11100', '11000', '10000', '0000']

module_order = ['11111', '01111', '00111', '00011', '00001', '00000', '10000', '11000', '11100', '11110', '11111*']

# Iterando sobre cada valor único de 'cap'
# Imprimindo todos os valores únicos de 'pretrain_data'
print("Unique pretrain_data values:", df['pretrain_data'].unique())

for cap in df['cap'].unique():
    # Filtrando os dados para o valor atual de 'cap'
    cap_data = result.loc[result.index.get_level_values('cap') == cap]
    
    plt.figure(figsize=(10, 6))
    
    # Extraindo os valores de 'modules' na ordem especificada para todos os pretrain_data
    for pretrain_data in df['pretrain_data'].unique():
        pretrain_data_group = cap_data.loc[cap_data.index.get_level_values('pretrain_data') == pretrain_data]
        
        means = []
        stds = []
        for module in module_order:
            try:
                mean = pretrain_data_group.loc[pretrain_data_group.index.get_level_values('modules') == module]['mean'].values[0]
                std = pretrain_data_group.loc[pretrain_data_group.index.get_level_values('modules') == module]['std'].values[0]
            except IndexError:
                mean = np.nan
                std = np.nan
            means.append(mean)
            stds.append(std)
        
        # Adicionando os dados ao gráfico
        plt.errorbar(module_order, means, yerr=stds, fmt='o-', capsize=5, label=f'Pretrain: {pretrain_data}')
        # plt.errorbar(module_order, means, fmt='o-', capsize=5, label=f'Pretrain: {pretrain_data}')
        # Plota a sequência conectada (sem o último 11111)

    
    # Configurando o gráfico
    plt.xticks(rotation=45)
    plt.xlabel('Modules')
    plt.ylabel('Mean IoU')
    plt.title(f'Cap: {cap}')
    plt.legend()
    plt.tight_layout()
    output_filename = f'modules_output/cap_{cap}_combined_pretrain_data.png'
    plt.savefig(output_filename)
    plt.close()
