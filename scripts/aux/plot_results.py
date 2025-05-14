import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(
    data_file, output_file, metric="iou", selected_backbones=None, selected_data=None
):
    """
    Generates a bar plot of the evaluation metrics grouped by `cap` with mean and standard deviation.

    Parameters:
        data_file (str): Path to the CSV file containing evaluation results.
        output_file (str): Path to save the generated plot as a PNG file.
        metric (str): The evaluation metric to display on the y-axis ("iou" or "f1").
        selected_backbones (list): List of backbones to display. If None, all backbones are shown.
        selected_data (str): The data type to filter by (e.g., "f3", "seam_ai"). If None, all data is included.
    """
    # Load the data
    df = pd.read_csv(data_file)

    # Filter by selected data and backbones
    if selected_data:
        df = df[df["downstream"] == selected_data]
    if selected_backbones:
        df = df[df["backbone"].isin(selected_backbones)]

    # Group by cap, backbone, and downstream, calculating mean and std for the selected metric
    grouped = (
        df.groupby(["cap", "backbone", "downstream"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Fill NaN values with 0 to avoid plotting issues
    grouped = grouped.fillna(0)

    # Get the unique values of cap, backbones, and downstream in the filtered dataset
    # caps = sorted(grouped["cap"].unique())
    # backbones = sorted(grouped["backbone"].unique())

    caps = grouped["cap"].unique()
    backbones = grouped["backbone"].unique()

    # Create the bar plot
    bar_width = 0.08  # Width of individual bars
    x_positions = np.arange(len(caps))  # Positions for the caps on the x-axis

    plt.figure(figsize=(10, 6))

    for i, backbone in enumerate(backbones):
        # Filter rows for the current backbone
        backbone_data = grouped[grouped["backbone"] == backbone]

        # Align x positions for this backbone
        backbone_x = x_positions + i * bar_width

        # Plot bars with error bars for std
        plt.bar(
            backbone_x,
            backbone_data["mean"],
            yerr=backbone_data["std"],
            width=bar_width,
            label=backbone,
            capsize=5,
        )

    # Customize plot
    plt.xticks(
        x_positions + (len(backbones) - 1) * bar_width / 2, [f"{cap}" for cap in caps]
    )
    plt.xlabel("Fraction of Data (cap)", fontsize=12)
    plt.ylabel(f"Mean {metric.upper()} with Std Dev", fontsize=12)
    plt.title(
        f"{metric.upper()} vs Fraction of Data for Selected Backbones on {selected_data}",
        fontsize=14,
    )
    plt.legend(title="Backbone", fontsize=10, loc="best")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved to {output_file}")


# Example usage:
plot_metrics(
    data_file="evaluation_results.csv",  # CSV file with evaluation results
    output_file="outputs/downstream_f3.png",  # Output file for the plot
    metric="iou",  # Metric to plot: "iou" or "f1"
    selected_backbones=[
        "f3",
        "f3_norm",
        "seam_ai",
        "seam_ai_norm",
        "both",
        "both_N",
    ],  # Backbones to display
    selected_data="f3",  # Data to filter by (set to None for all data)
)
plot_metrics(
    data_file="evaluation_results.csv",  # CSV file with evaluation results
    output_file="outputs/downstream_f3_N.png",  # Output file for the plot
    metric="iou",  # Metric to plot: "iou" or "f1"
    selected_backbones=[
        "f3",
        "f3_norm",
        "seam_ai",
        "seam_ai_norm",
        "both",
        "both_N",
    ],  # Backbones to display
    selected_data="f3_N",  # Data to filter by (set to None for all data)
)
plot_metrics(
    data_file="evaluation_results.csv",  # CSV file with evaluation results
    output_file="outputs/downstream_seam_ai.png",  # Output file for the plot
    metric="iou",  # Metric to plot: "iou" or "f1"
    selected_backbones=[
        "f3",
        "f3_norm",
        "seam_ai",
        "seam_ai_norm",
        "both",
        "both_N",
    ],  # Backbones to display
    selected_data="seam_ai",  # Data to filter by (set to None for all data)
)
plot_metrics(
    data_file="evaluation_results.csv",  # CSV file with evaluation results
    output_file="outputs/downstream_seam_ai_N.png",  # Output file for the plot
    metric="iou",  # Metric to plot: "iou" or "f1"
    selected_backbones=[
        "f3",
        "f3_norm",
        "seam_ai",
        "seam_ai_norm",
        "both",
        "both_N",
    ],  # Backbones to display
    selected_data="seam_ai_N",  # Data to filter by (set to None for all data)
)
plot_metrics(
    data_file="evaluation_results.csv",  # CSV file with evaluation results
    output_file="outputs/downstream_f3_full.png",  # Output file for the plot
    metric="iou",  # Metric to plot: "iou" or "f1"
    selected_backbones=[
        "f3",
        "f3_norm",
        "seam_ai",
        "seam_ai_norm",
        "both",
        "both_N",
        "COCO",
        "imagenet",
        "seg",
        "sup",
    ],  # Backbones to display
    selected_data="f3",  # Data to filter by (set to None for all data)
)
plot_metrics(
    data_file="evaluation_results.csv",  # CSV file with evaluation results
    output_file="outputs/downstream_f3_N_full.png",  # Output file for the plot
    metric="iou",  # Metric to plot: "iou" or "f1"
    selected_backbones=[
        "f3",
        "f3_norm",
        "seam_ai",
        "seam_ai_norm",
        "both",
        "both_N",
        "COCO",
        "imagenet",
        "seg",
        "sup",
    ],  # Backbones to display
    selected_data="f3_N",  # Data to filter by (set to None for all data)
)
plot_metrics(
    data_file="evaluation_results.csv",  # CSV file with evaluation results
    output_file="outputs/downstream_seam_ai_full.png",  # Output file for the plot
    metric="iou",  # Metric to plot: "iou" or "f1"
    selected_backbones=[
        "f3",
        "f3_norm",
        "seam_ai",
        "seam_ai_norm",
        "both",
        "both_N",
        "COCO",
        "imagenet",
        "seg",
        "sup",
    ],  # Backbones to display
    selected_data="seam_ai",  # Data to filter by (set to None for all data)
)
plot_metrics(
    data_file="evaluation_results.csv",  # CSV file with evaluation results
    output_file="outputs/downstream_seam_ai_N_full.png",  # Output file for the plot
    metric="iou",  # Metric to plot: "iou" or "f1"
    selected_backbones=[
        "f3",
        "f3_norm",
        "seam_ai",
        "seam_ai_norm",
        "both",
        "both_N",
        "COCO",
        "imagenet",
        "seg",
        "sup",
    ],  # Backbones to display
    selected_data="seam_ai_N",  # Data to filter by (set to None for all data)
)
