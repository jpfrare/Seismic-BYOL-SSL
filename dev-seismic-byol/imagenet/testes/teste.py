import sys
from pathlib import Path
import scipy.io as sio

# Garante que o Python enxergue a raiz do projeto para imports locais
sys.path.append(str(Path(__file__).resolve().parent))

# Importa as suas funções e os seus Readers reais do seu projeto
from base.utils import reduce_taxonomic_diversity, build_heritage_path
from base.ImagenetReader import ImagenetReader, ImagenetValReader  

if __name__ == "__main__":
    # --- CONFIGURAÇÃO DOS CAMINHOS REAIS DO CLUSTER ---
    meta_mat = "/petrobr/parceirosbr/home/joao.frare/workspace/spfm/sharedata/datasets/ImageNet_2012/extra_files/ILSVRC2012_devkit_t12/data/meta.mat"
    gt_txt = "/petrobr/parceirosbr/home/joao.frare/workspace/spfm/sharedata/datasets/ImageNet_2012/extra_files/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
    
    train_img_root = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/train"
    val_img_root = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/val"
    train_entries_npy = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/extras_v3/entries-TRAIN.npy"

    print("=" * 70)
    print("   INICIANDO INSTANCIAÇÃO DOS READERS VIA IMPORT")
    print("=" * 70)

    # 1. Instanciação dos objetos leitores
    try:
        print("⏳ Carregando ImagenetValReader...")
        val_reader = ImagenetValReader(root=val_img_root, gt_path=gt_txt, mat_path=meta_mat)
        
        print("⏳ Carregando ImagenetReader...")
        train_reader = ImagenetReader(root=train_img_root, entries_path=train_entries_npy)
    except Exception as e:
        print(f"❌ Erro ao instanciar os Readers originais.\nDetalhe: {e}")
        sys.exit(1)

    # 2. Configuração do fatiamento taxonômico para o teste
    # Modifique aqui (LEVEL = 3, 4, 7...) para avaliar o comportamento do colapso
    TOP_DOWN = False
    LEVEL = 3

    print(f"\n⚡ Disparando .to_coarse_classes(top_down={TOP_DOWN}, level={LEVEL}) nos Readers...")
    
    try:
        num_classes_train = train_reader.to_coarse_classes(top_down=TOP_DOWN, level=LEVEL, mat_path=meta_mat)
        num_classes_val = val_reader.to_coarse_classes(top_down=TOP_DOWN, level=LEVEL, mat_path=meta_mat)
    except TypeError as te:
        print(f"❌ Erro de assinatura: verifique os parâmetros da função reduce_taxonomic_diversity. Detalhe: {te}")
        sys.exit(1)

    print("\n" + "-" * 60)
    print("📊 MÓDULO DE VERIFICAÇÃO DE ALINHAMENTO INTERNO:")
    print("-" * 60)
    print(f"Classes finais computadas no Treino:     {num_classes_train}")
    print(f"Classes finais computadas na Validação:  {num_classes_val}")

    # CRITÉRIO 1: Consistência na quantidade de saídas
    if num_classes_train != num_classes_val:
        print("\n❌ ERRO DE DESALINHAMENTO CRÍTICO: Treino e Validação geraram shapes de saída diferentes!")
        sys.exit(1)
    else:
        print("\n✅ Sucesso: Ambas as estruturas concordam no número de neurônios de saída.")

    # CRITÉRIO 2: Teste Cruzado de Labels (Mapeamento Idêntico)
    print("\n🔍 Analisando chaves para garantir que a mesma pasta (WNID) possui a mesma label...")
    train_map = train_reader.wind_to_coarse
    val_all_wnids = val_reader.all_wnids
    
    wnid_to_coarse_val_check, _ = reduce_taxonomic_diversity(val_all_wnids, TOP_DOWN, LEVEL, meta_mat)

    mismatches = 0
    checked_keys = 0

    for wnid, t_label in train_map.items():
        if wnid in wnid_to_coarse_val_check:
            v_label = wnid_to_coarse_val_check[wnid]
            checked_keys += 1
            if t_label != v_label:
                mismatches += 1
                if mismatches <= 5:
                    print(f"   ⚠️ Conflito no WNID {wnid}: Treino -> classe {t_label}, Validação -> classe {v_label}")

    if mismatches > 0:
        print(f"\n❌ CONFLITO TAXONÔMICO DETECTADO: {mismatches} classes estão desalinhadas!")
        sys.exit(1)
    else:
        print(f"\n✅ Sucesso Total: Todos os {checked_keys} WNIDs ativos possuem mapeamento idêntico entre os Readers.")

    # ---------------------------------------------------------------------
    # 🚨 MÓDULO INTEGRADO: DETECTOR DE OVERLAP BASEADO EM DISTÂNCIA DA RAIZ
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("   ANÁLISE TOPOLÓGICA DE OVERLAP (MÉTRICAS DO SUPERVISIONADO)")
    print("=" * 70)

    # Carrega a árvore de caminhos reais calculada pela utils
    heritage_path = build_heritage_path(meta_mat)

    # Agrupa os WNIDs originais por Coarse Label gerada
    coarse_groups = {}
    for wnid, coarse_label in train_map.items():
        coarse_groups.setdefault(coarse_label, []).append(wnid)

    # Estatísticas básicas de volume
    sizes = [len(v) for v in coarse_groups.values()]
    max_size = max(sizes)
    avg_size = sum(sizes) / len(sizes)
    
    print(f"• Distribuição: Em média, cada superclasse englobou {avg_size:.2f} classes finas.")
    print(f"• Tamanho do maior grupo formado: {max_size} classes finas.")

    worst_overlap_depth = 999
    problematic_groups_count = 0

    # Varre cada grupo para calcular o Menor Ancestral Comum (LCA) e sua distância até a raiz
    for coarse_id, wnids_do_grupo in coarse_groups.items():
        if len(wnids_do_grupo) <= 1:
            continue
        
        # Encontra a intersecção de toda a árvore de ancestrais deste grupo específico
        ancestrais_comuns = set(heritage_path[wnids_do_grupo[0]])
        for w in wnids_do_grupo[1:]:
            ancestrais_comuns = ancestrais_comuns.intersection(set(heritage_path[w]))
        
        # Reconstrói a ordem correta do topo para a base (da raiz até o LCA)
        exemplo_wnid = wnids_do_grupo[0]
        caminho_total = heritage_path[exemplo_wnid]
        ancestrais_ordenados = [anc for anc in caminho_total if anc in ancestrais_comuns]
        
        # Mede a profundidade (tamanho do caminho da raiz até o LCA encontrado)
        dist_to_root = len(ancestrais_ordenados)

        if dist_to_root < worst_overlap_depth:
            worst_overlap_depth = dist_to_root

        # Se dist_to_root <= 3, colapsou em nós perigosamente genéricos (ex: entity, organism, artifact)
        if dist_to_root <= 3:
            problematic_groups_count += 1
            if problematic_groups_count <= 3:  # Limita logs no terminal
                print(f"  ⚠️ Superclasse {coarse_id} colapsou muito alto! Distância até a raiz: {dist_to_root} nós.")
                print(f"     Contém {len(wnids_do_grupo)} classes finas agrupadas no mesmo neurônio.")

    print(f"\n📈 MÉTRICAS TOPOLÓGICAS CALCULADAS:")
    print(f"  - Distância do maior colapso até a raiz absoluta: {worst_overlap_depth} níveis.")
    print(f"  - Quantidade de superclasses em zonas de overlap crítico: {problematic_groups_count}")

    print("\n📋 DIAGNÓSTICO DO EXPERIMENTO REAL:")
    print("-" * 45)
    
    # Avaliação de risco baseada na topologia real do WordNet
    if worst_overlap_depth <= 2:
        print("❌ Veredicto: OVERLAP CRÍTICO DETECTADO!")
        print("   As classes colapsaram nas raízes máximas conceituais ('entity' ou 'physical object').")
        print("   O gradiente vai sofrer severamente no supervisionado puro. REDUZA O LEVEL!")
    elif worst_overlap_depth == 3:
        print("⚠️  Veredicto: ALERTA DE OVERLAP MODERADO.")
        print("   Agrupamentos massivos no nível de 'Artifact' ou 'Organism'.")
        print("   Aceitável para análises macro, mas as fronteiras visuais estão muito borradas.")
    else:
        print("✅ Veredicto: CRITÉRIO DE DISTÂNCIA E OVERLAP APROVADO.")
        print("   Os agrupamentos pararam em nós profundos e especializados.")
        print("   Fronteiras semânticas e geométricas preservadas com segurança para o Cross-Entropy.")

    print("=" * 70)
    print("   FIM DO TESTE DE INTEGRAÇÃO")
    print("=" * 70)