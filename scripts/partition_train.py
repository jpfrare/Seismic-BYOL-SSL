from train import train_func


def main():

    LIST = 1
    NODE = f"f3_folds_tesst_13{LIST}]"
    REPORT_NAME = f"{NODE}_run"

    report_path = "reports_2/"

    EPOCAS = 200
    BATCH_SIZE = 8
    CAP = 1.0

    # The import name is not gonna be used,
    # so can be any valid file
    REPETITION = "V2"
    IMPORT_NAME = f"{REPETITION}_E300_B32_S256_f3"
    SUPERVISED = True
    FREEZE = False
    DOWNSTREAM_DATA = "f3"
    MODE = "supervised"

    list_of_paths = [
        # f3 ilan
        "../data/f3",
        # f3 original
        "../../shared_data/seismic_vinicius/f3",
        # f3 com val do fold 0 com contaminação (sem crop)
        "../../shared_data/seismic_vinicius/f3_fold_0_final_un",
        # f3 com val do fold 0 sem contaminação (cropped)
        "../../shared_data/seismic_vinicius/f3_fold_0_final_crop",
    ]

    list_of_names = [
        "final_f3_ilan",
        "final_f3_original",
        "final_fold_0_un",
        "final_fold_0_crop",
    ]

    with open(report_path + f"{REPORT_NAME}.txt", "w") as f:
        f.write("Report of the training\n")
        f.write("---------------------------------------\n")
        f.write(f"Paths being used: {list_of_paths}\n")
        f.write(f"Node: {NODE}\n")
        f.write("---------------------------------------\n")

    for idx, path in enumerate(list_of_paths):

        SAVE_NAME = list_of_names[idx]
        ROOT_DIR = path

        with open(report_path + f"{REPORT_NAME}.txt", "a") as f:
            f.write(f"------------------ Training on {path} ------------------\n")
            f.write(f"------------------ Saving: {SAVE_NAME} ------------------\n")

        train_func(
            epocas=EPOCAS,
            batch_size=BATCH_SIZE,
            cap=CAP,
            import_name=IMPORT_NAME,
            save_name=SAVE_NAME,
            supervised=SUPERVISED,
            freeze=FREEZE,
            downstream_data=DOWNSTREAM_DATA,
            mode=MODE,
            repetition=REPETITION,
            root_dir=ROOT_DIR,
        )


if __name__ == "__main__":
    main()
