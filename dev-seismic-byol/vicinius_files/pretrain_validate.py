import torch
from pprint import pprint

ckpt_path = "/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/checkpoints/ckpt_vinicius/pretrain/0/V0_pretrain_both_N_In256_B32_E85000_lr1e-05/both_N/last.ckpt"

# Carrega checkpoint
ckpt = torch.load(ckpt_path, map_location="cpu")

print("\n=== KEYS DO CHECKPOINT ===")
print(ckpt.keys())

# --------------------------------------------------
# 1. Hyperparameters (se existir)
# --------------------------------------------------
print("\n=== HYPERPARAMETERS ===")
if "hyper_parameters" in ckpt:
    pprint(ckpt["hyper_parameters"])
    
    if "learning_rate" in ckpt["hyper_parameters"]:
        print("\n[OK] Learning rate (hyper_parameters):",
              ckpt["hyper_parameters"]["learning_rate"])
    else:
        print("\n[!] learning_rate não encontrado em hyper_parameters")
else:
    print("[!] Nenhum hyper_parameters encontrado")

# --------------------------------------------------
# 2. Optimizer states (LR REAL usado)
# --------------------------------------------------
print("\n=== OPTIMIZER STATES ===")
if "optimizer_states" in ckpt and len(ckpt["optimizer_states"]) > 0:
    opt_state = ckpt["optimizer_states"][0]

    if "param_groups" in opt_state:
        print("\nLearning rates por param_group:")
        for i, group in enumerate(opt_state["param_groups"]):
            lr = group.get("lr", None)
            print(f"Group {i}: lr = {lr}")
    else:
        print("[!] param_groups não encontrado dentro do optimizer")
else:
    print("[!] Nenhum optimizer_states encontrado")

# --------------------------------------------------
# 3. Scheduler (se existir)
# --------------------------------------------------
print("\n=== LR SCHEDULER STATES ===")
if "lr_schedulers" in ckpt:
    pprint(ckpt["lr_schedulers"])
else:
    print("[!] Nenhum scheduler encontrado")

# --------------------------------------------------
# 4. Outras infos úteis
# --------------------------------------------------
print("\n=== INFO EXTRA ===")
for key in ["epoch", "global_step"]:
    if key in ckpt:
        print(f"{key}: {ckpt[key]}")