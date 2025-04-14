import os
import shutil


# Função para copiar e renomear os arquivos
def copy_files(src_path, dest_path, prefix):
    for root, _, files in os.walk(src_path):
        for file_name in files:
            # Construa o novo nome com o prefixo
            new_name = f"{prefix}_{file_name}"
            print(new_name)
            # Defina o caminho de origem e destino
            src_file = os.path.join(root, file_name)
            dest_file = os.path.join(dest_path, new_name)
            # Copie o arquivo para o destino com o novo nome
            shutil.copy2(src_file, dest_file)

def main():
    # Defina os caminhos dos datasets
    dataset1_path = '../asml/datasets/tiff_data/f3_segmentation_N'
    dataset2_path = '../asml/datasets/tiff_data/seam_ai_N'
    combined_dataset_path = '../asml/datasets/tiff_data/both_N'

    # Nomes dos datasets para usar como prefixo
    dataset1_prefix = 'f3'
    dataset2_prefix = 'seam_ai'

    # Estrutura de pastas
    subdirs = ['train', 'val', 'teste']

    # Crie as pastas combinadas se não existirem
    for folder in ['images', 'annotations']:
        for subdir in subdirs:
            os.makedirs(os.path.join(combined_dataset_path, folder, subdir), exist_ok=True)

    # Copie os arquivos do dataset1
    for subdir in subdirs:
        copy_files(
            os.path.join(dataset1_path, 'images', subdir),
            os.path.join(combined_dataset_path, 'images', subdir),
            dataset1_prefix
        )
        copy_files(
            os.path.join(dataset1_path, 'annotations', subdir),
            os.path.join(combined_dataset_path, 'annotations', subdir),
            dataset1_prefix
        )

    # Copie os arquivos do dataset2
    for subdir in subdirs:
        copy_files(
            os.path.join(dataset2_path, 'images', subdir),
            os.path.join(combined_dataset_path, 'images', subdir),
            dataset2_prefix
        )
        copy_files(
            os.path.join(dataset2_path, 'annotations', subdir),
            os.path.join(combined_dataset_path, 'annotations', subdir),
            dataset2_prefix
        )

    print("Datasets combinados com sucesso!")


if __name__ == "__main__":
    main()