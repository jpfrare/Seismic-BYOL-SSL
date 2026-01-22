from evaluate import eval_func
import numpy as np


def main():

    report_name = "final_folds_eval_04"

    data = "f3"

    list_of_paths = [
        # f3 ilan
        "../data/f3",
        "../data/f3",
        "../data/f3",
        # f3 original
        "../../shared_data/seismic_vinicius/f3",
        "../../shared_data/seismic_vinicius/f3",
        "../../shared_data/seismic_vinicius/f3",
        # f3 com val do fold 0 com contaminação (sem crop)
        "../../shared_data/seismic_vinicius/f3_fold_0_final_un",
        "../../shared_data/seismic_vinicius/f3_fold_0_final_un",
        "../../shared_data/seismic_vinicius/f3_fold_0_final_un",
        # f3 com val do fold 0 sem contaminação (cropped)
        "../../shared_data/seismic_vinicius/f3_fold_0_final_crop",
        "../../shared_data/seismic_vinicius/f3_fold_0_final_crop",
        "../../shared_data/seismic_vinicius/f3_fold_0_final_crop",
    ]

    list_of_models = [
        "final_f3_ilan",
        "final_f3_ilan-v1",
        "final_f3_ilan-v2",
        "final_f3_original",
        "final_f3_original-v1",
        "final_f3_original-v2",
        "final_fold_0_crop",
        "final_fold_0_crop-v1",
        "final_fold_0_crop-v2",
        "final_fold_0_un",
        "final_fold_0_un-v1",
        "final_fold_0_un-v2",
    ]

    repetition = "V_0.04"

    with open(report_name + ".txt", "w") as f:
        f.write("---------------------------------------\n")
        f.write(f"Models: {list_of_models}\n")
        f.write("---------------------------------------\n")

    for idx, model in enumerate(list_of_models):

        path = list_of_paths[idx]

        print(f"---------- Path: {path} ----------")

        with open(report_name + ".txt", "a") as f:
            f.write(f"------------------ {model} ------------------\n")

        iou, f1 = eval_func(
            import_name=model,
            mode="supervised",
            dataset=data,
            repetition=repetition,
            root_dir=path,
        )

        with open(report_name + ".txt", "a") as f:
            f.write(30 * "--" + "\n")
            f.write(model + "\n")
            f.write(f"iou = {iou[2]:.4f}\n")
            f.write(f"f1 = {f1[2]:.4f}\n")


if __name__ == "__main__":
    main()
