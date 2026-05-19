import sys
from pathlib import Path



# Importa as suas funções e os seus Readers reais do seu projeto
# Ajuste o caminho do import caso a estrutura de pastas seja diferente
from base.utils import reduce_taxonomic_diversity
from base.ImagenetReader import ImagenetReader, ImagenetValReader  # substitua pelo nome do seu arquivo de readers

if __name__ == "__main__":
    # --- CONFIGURAÇÃO DOS CAMINHOS REAIS DO CLUSTER ---
    
    meta_mat = "/petrobr/parceirosbr/home/joao.frare/workspace/spfm/sharedata/datasets/ImageNet_2012/extra_files/ILSVRC2012_devkit_t12/data/meta.mat"
    gt_txt = "/petrobr/parceirosbr/home/joao.frare/workspace/spfm/sharedata/datasets/ImageNet_2012/extra_files/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
    
    # Caminhos das pastas de imagens (coloque o caminho correto do seu ambiente)
    train_img_root = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/train"
    val_img_root = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/val"
    
    # O arquivo de entradas estruturadas do treino (.npy)
    train_entries_npy = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/extras_v3/entries-TRAIN.npy"

    print("=" * 70)
    print("   INICIANDO INSTANCIAÇÃO DOS READERS VIA IMPORT")
    print("=" * 70)

    # 1. Instanciação dos objetos originais do seu código
    try:
        print("⏳ Carregando ImagenetValReader...")
        val_reader = ImagenetValReader(root=val_img_root, gt_path=gt_txt, meta_path=meta_mat)
        
        print("⏳ Carregando ImagenetReader...")
        train_reader = ImagenetReader(root=train_img_root, entries_path=train_entries_npy)
    except Exception as e:
        print(f"❌ Erro ao instanciar os Readers originais.\nDetalhe: {e}")
        sys.exit(1)

    # 2. Configuração do fatiamento taxonômico para o teste
    # Sinta-se livre para mudar aqui e testar o comportamento de diferentes níveis!
    TOP_DOWN = False
    LEVEL = 2

    print(f"\n⚡ Disparando .to_coarse_classes(top_down={TOP_DOWN}, level={LEVEL}) nos Readers...")
    
    # Executa os métodos internos que modificam o self.targets de cada classe
    try:
        num_classes_train = train_reader.to_coarse_classes(top_down=TOP_DOWN, level=LEVEL, mat_path= meta_mat)
        num_classes_val = val_reader.to_coarse_classes(top_down=TOP_DOWN, level=LEVEL, mat_path= meta_mat)
    except TypeError as te:
        print(f"❌ Erro de assinatura: verifique se a função reduce_taxonomic_diversity")
        print(f"recebe o 'mat_path' ou se ele está fixo no escopo. Detalhe: {te}")
        sys.exit(1)

    print("\n" + "-" * 60)
    print("📊 MÓDULO DE VERIFICAÇÃO DE ALINHAMENTO INTERNO:")
    print("-" * 60)
    print(f"Classes finais computadas no Treino:     {num_classes_train}")
    print(f"Classes finais computadas na Validação:  {num_classes_val}")

    # ---------------------------------------------------------------------
    # CRITÉRIO DE VALIDAÇÃO 1: Consistência na quantidade de saídas do modelo
    # ---------------------------------------------------------------------
    if num_classes_train != num_classes_val:
        print("\n❌ ERRO DE DESALINHAMENTO CRÍTICO: Treino e Validação geraram shapes de saída diferentes!")
        print("Causa provável: Seus dicionários de mapeamento interno ('wind_to_coarse')")
        print("geraram IDs numéricos diferentes porque a ordenação de WNIDs divergiu.")
    else:
        print("\n✅ Sucesso: Ambas as estruturas concordam no número de neurônios de saída.")

    # ---------------------------------------------------------------------
    # CRITÉRIO DE VALIDAÇÃO 2: Teste Cruzado de Labels de Validação vs Treino
    # ---------------------------------------------------------------------
    print("\n🔍 Analisando chaves para garantir que a mesma pasta (WNID) possui a mesma label...")
    
    # Pegamos o mapa gerado dentro do objeto do treino
    train_map = train_reader.wind_to_coarse
    
    # Na sua classe ImagenetValReader, você não salva o 'wnid_to_coarse' como atributo do self,
    # mas conseguimos extrair o comportamento reconstruindo o mapeamento a partir do dicionário original do Reader:
    val_all_wnids = val_reader.all_wnids
    
    # Reconstrói temporariamente o mapeamento final que a validação aplicou
    # (Usando a mesma lógica da linha: old_label_to_new_label = {...} do seu ValReader)
    # Primeiro precisamos gerar o dicionário wnid -> coarse_label rodando a redução isolada para conferir
    wnid_to_coarse_val_check, _ = reduce_taxonomic_diversity(val_all_wnids, TOP_DOWN, LEVEL)

    mismatches = 0
    checked_keys = 0

    for wnid, t_label in train_map.items():
        if wnid in wnid_to_coarse_val_check:
            v_label = wnid_to_coarse_val_check[wnid]
            checked_keys += 1
            if t_label != v_label:
                mismatches += 1
                if mismatches <= 5:
                    print(f"   ⚠️ Conflito no WNID {wnid}: Treino associou à classe {t_label}, mas Validação associou à classe {v_label}")

    if mismatches > 0:
        print(f"\n❌ CONFLITO TAXONÔMICO DETECTADO: {mismatches} classes estão desalinhadas!")
        print("Isso fará o modelo falhar na validação, pois os labels não apontam para o mesmo conceito.")
    else:
        print(f"\n✅ Sucesso Total: Todos os {checked_keys} WNIDs ativos possuem mapeamento idêntico entre os Readers.")

    print("\n" + "=" * 70)
    print("   FIM DO TESTE DE INTEGRAÇÃO")
    print("=" * 70)