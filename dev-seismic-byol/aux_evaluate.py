import os
import re

def extract_epoch_number(filename):
    match = re.match(r"epoch=(\d+)", filename)
    return int(match.group(1)) if match else -1


def find_ckpt_files(base_dir="ckpt/train", target_repetition=None):
    results = []

    repetitions = [target_repetition] if target_repetition != None else os.listdir(base_dir)

    for repetition_dir in repetitions:
        rep_path = os.path.join(base_dir, str(repetition_dir))
        if not os.path.isdir(rep_path):
            continue

        for model_dir in os.listdir(rep_path):
            model_path = os.path.join(rep_path, model_dir)
            if not os.path.isdir(model_path):
                continue

            # extrai metadados do nome do modelo
            match = re.match(r"V(\d+)_pre_(.+?)_train_(.+?)_cap_(.+)", model_dir)
            if not match:
                continue
            repetition, pretrain_data, train_data, cap = match.groups()

            for train_data_dir in os.listdir(model_path):
                ckpt_path = os.path.join(model_path, train_data_dir)
                if not os.path.isdir(ckpt_path):
                    continue

                # encontra o arquivo com maior número de epoch
                ckpt_files = [f for f in os.listdir(ckpt_path) if f.startswith("epoch=")]
                if ckpt_files:
                    ckpt_files.sort(key=extract_epoch_number, reverse=True)
                    results.append({
                        "repetition": repetition,
                        "pretrain_data": pretrain_data,
                        "train_data": train_data,
                        "cap": cap,
                        "ckpt_file": os.path.join(ckpt_path, ckpt_files[0])
                    })

    return results


def main():


    results = find_ckpt_files(target_repetition=0j)
    
    # for r in results[:3]:
    #     print(r)
        
    print(len(results))
    
    results = find_ckpt_files(target_repetition=2)
    
    # for r in results[:3]:
    #     print(r)
        
    print(len(results))
    
    results = find_ckpt_files(target_repetition=3)
    
    # for r in results[:3]:
    #     print(r)
        
    print(len(results))
    
if __name__ == "__main__":
    main()