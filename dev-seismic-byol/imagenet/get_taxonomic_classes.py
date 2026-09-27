from collections import defaultdict

from scipy.io import loadmat

from base.TaxonomicHandler import TaxonomicHandler


MAT_ROOT = (
    "/petrobr/parceirosbr/home/joao.frare/workspace/spfm/"
    "sharedata/datasets/ImageNet_2012/extra_files/"
    "ILSVRC2012_devkit_t12/data/meta.mat"
)


# ============================================================
# Load
# ============================================================

handler = TaxonomicHandler(MAT_ROOT)

synsets = loadmat(MAT_ROOT)["synsets"]

# As 1000 classes originais do ImageNet.
# Pelo README:
#   IDs 1-1000 = low-level synsets
imagenet_wnids = [
    str(synsets[i][0][1][0])
    for i in range(1000)
]

print("=" * 80)
print("BASIC INFORMATION")
print("=" * 80)

print(f"Total synsets in meta.mat: {len(synsets)}")
print(f"ImageNet low-level synsets: {len(imagenet_wnids)}")

print()


# ============================================================
# 1. Teste da hierarquia
# ============================================================

print("=" * 80)
print("TESTING HERITAGE PATH")
print("=" * 80)

heritage_path = handler.build_heritage_path()

print(f"Number of low-level paths: {len(heritage_path)}")

# Quantos synsets diferentes aparecem como ancestrais?
all_ancestors = set()

for path in heritage_path.values():
    # O primeiro elemento é a própria classe.
    all_ancestors.update(path[1:])

print(f"Number of unique ancestors: {len(all_ancestors)}")

print()


# ============================================================
# 2. Verificar se os 1000 low-level synsets são realmente
#    folhas entre si
# ============================================================

print("=" * 80)
print("TESTING LOW-LEVEL SYNSET OVERLAP")
print("=" * 80)

imagenet_set = set(imagenet_wnids)

low_level_overlap = []

for wnid in imagenet_wnids:

    ancestors = heritage_path[wnid]

    # Ignoramos o próprio WNID.
    ancestor_overlap = imagenet_set.intersection(ancestors[1:])

    if ancestor_overlap:
        low_level_overlap.append(
            (wnid, ancestor_overlap)
        )


if not low_level_overlap:
    print(
        "OK: nenhum low-level synset é ancestral de outro "
        "low-level synset."
    )
else:
    print(
        "WARNING: foram encontrados low-level synsets "
        "que são ancestrais de outros:"
    )

    for wnid, overlaps in low_level_overlap:
        print(
            wnid,
            "->",
            handler.wnid_to_description[wnid]
        )

        for ancestor in overlaps:
            print(
                "    ancestor:",
                ancestor,
                "->",
                handler.wnid_to_description[ancestor]
            )

print()


# ============================================================
# 3. Testar cada nível de redução
# ============================================================

for top_down in [False, True]:

    print("=" * 80)

    direction = "TOP-DOWN" if top_down else "BOTTOM-UP"

    print(f"REDUCTION: {direction}")

    print("=" * 80)

    for level in [9, 7, 6, 3]:

        (
            wnid_to_class,
            num_classes
        ) = handler.reduce_taxonomic_diversity(
            imagenet_wnids,
            top_down,
            level
        )

        chosen = list(handler.chosen_wnids)
        chosen_set = set(chosen)

        print()
        print("-" * 80)
        print(f"LEVEL = {level}")
        print("-" * 80)

        print(f"Number of chosen classes: {num_classes}")

        # ----------------------------------------------------
        # Mostrar classes escolhidas
        # ----------------------------------------------------

        print("\nChosen synsets:")

        for class_id, wnid in enumerate(chosen):

            description = handler.wnid_to_description[wnid]

            print(
                f"  {class_id:3d} | "
                f"{wnid:10s} | "
                f"{description}"
            )

        # ----------------------------------------------------
        # Verificar se existem chosen synsets que são
        # ancestrais de outros chosen synsets
        # ----------------------------------------------------

        print("\nChecking overlap between chosen synsets...")

        chosen_overlaps = []

        for wnid in chosen:

            ancestors = heritage_path.get(wnid, [])

            overlap = chosen_set.intersection(
                ancestors[1:]
            )

            if overlap:

                chosen_overlaps.append(
                    (wnid, overlap)
                )

        if not chosen_overlaps:

            print(
                "  OK: no chosen synset is an ancestor "
                "of another chosen synset."
            )

        else:

            print(
                "  WARNING: chosen synsets contain "
                "ancestor/descendant relationships!"
            )

            for wnid, overlaps in chosen_overlaps:

                print(
                    f"\n  {wnid} -> "
                    f"{handler.wnid_to_description[wnid]}"
                )

                for ancestor in overlaps:

                    print(
                        f"      ancestor: {ancestor} -> "
                        f"{handler.wnid_to_description[ancestor]}"
                    )

        # ----------------------------------------------------
        # Verificar quantas classes ImageNet caíram em cada
        # classe taxonômica
        # ----------------------------------------------------

        grouped_classes = defaultdict(list)

        for original_wnid, class_id in wnid_to_class.items():

            grouped_classes[class_id].append(
                original_wnid
            )

        print("\nNumber of original ImageNet classes per chosen class:")

        for class_id, members in sorted(
            grouped_classes.items()
        ):

            chosen_wnid = chosen[class_id]

            print(
                f"  class {class_id:3d} | "
                f"{chosen_wnid:10s} | "
                f"{handler.wnid_to_description[chosen_wnid]} "
                f"| {len(members)} original classes"
            )

        # ----------------------------------------------------
        # Verificar se todos os 1000 exemplos foram atribuídos
        # ----------------------------------------------------

        total_assigned = sum(
            len(members)
            for members in grouped_classes.values()
        )

        print(
            f"\nTotal original classes assigned: "
            f"{total_assigned}"
        )

        if total_assigned == 1000:
            print("  OK: all 1000 ImageNet classes were assigned.")
        else:
            print(
                "  WARNING: not all ImageNet classes "
                "were assigned!"
            )

print()
print("=" * 80)
print("DONE")
print("=" * 80)