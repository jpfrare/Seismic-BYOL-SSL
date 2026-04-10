import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/checkpoints/logs_vinicius/pretrain")

realNames = {
    "seam_ai_N": "Parihaka",
    "both_N": "F3+Par",
    "f3_N": "F3"
}

plt.figure(figsize=(10,6))

for pretrain in ["seam_ai_N", "both_N", "f3_N"]:
    
    all_losses = []

    for repetition in range(3):
        p = f"V{repetition}_pretrain_{pretrain}_In256_B32_E85000_lr1e-05"
        file = ROOT / f"{repetition}" / p / pretrain / p / pretrain / "metrics.csv"
        print(file)
        print(file.exists())

        df = pd.read_csv(file)
        all_losses.append(df["train_loss"].values)

    # transforma em array: (n_repetições, n_steps)
    losses = pd.DataFrame(all_losses)

    mean = losses.mean(axis=0)
    std = losses.std(axis=0)
    steps = df["step"]  # qualquer um serve, já que são iguais

    name = realNames[pretrain]

    plt.plot(steps, mean, label=name)
    plt.fill_between(steps, mean - std, mean + std, alpha=0.2)

plt.xscale("log")
plt.xlim(0, 85000)

plt.xlabel("Step")
plt.ylabel("Train Loss")
plt.title("Pretraining Loss (mean ± std, 3 runs)")
plt.legend()
plt.grid()
plt.tight_layout()

output_path = Path("/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/results/validate_experiments/loss_curves.png")

output_path.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output_path, dpi=300)

print(f"✅ Salvo em: {output_path}")