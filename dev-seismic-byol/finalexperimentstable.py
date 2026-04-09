import yaml
import pandas as pd
from pathlib import Path
import re

from pathlib import Path
import yaml

ROOT = Path("/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/checkpoints/logs_vinicius/train_patch")

repetitions = [0,1,2]
pretrains = ["imagenet", "coco", "seam_ai_N", "f3_N", "both_N", "scratch"]
finetunes = ["seam_ai_N", "f3_N"]

realNames = {
    "seam_ai_N": "Parihaka",
    "f3_N": "F3",
    "both_N": "F3+Par",
    "imagenet": "Imagenet",
    "coco": "CoCo",
    "scratch": "Scratch"
}

rows = []

for r in repetitions:
    for p in pretrains:
        for f in finetunes:

            directorypath = ROOT / f"{r}" / f"V{r}_pre_{p}_train_{f}_cap_100%_{{}}"/f"{f}"

            for filepath in directorypath.glob("metrics_*.yaml"):

                with open(filepath) as file:
                    data = yaml.safe_load(file)

                miou = data["classification"]["mIoU"][0]

                rows.append({
                    "pretrain": realNames[p],
                    "finetune": realNames[f],
                    "miou": miou
                })


# 🔹 dataframe
df = pd.DataFrame(rows)

# 🔹 média e desvio
agg = df.groupby(["finetune", "pretrain"])["miou"].agg(["mean", "std"]).reset_index()

# 🔹 formatar
agg["result"] = agg.apply(
    lambda x: f"{x['mean']:.2f} ± {x['std']:.2f}", axis=1
)

# 🔹 tabela final
table = agg.pivot(index="finetune", columns="pretrain", values="result")

print("\n TABELA FINAL:\n")
print(table) 
o path está certo