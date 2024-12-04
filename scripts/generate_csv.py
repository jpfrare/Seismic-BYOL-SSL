import os
import pandas as pd
import hashlib
from evaluate import eval_func

def generate_csv_from_models(models_folder, output_csv="evaluation_results.csv"):
    """
    Generates or updates a CSV file containing evaluation results for each model, with IoU and F1 as evaluation metrics.
    If a model with the same hash (`id`) is re-evaluated, the previous row is overwritten.
    
    Parameters:
        models_folder (str): Path to the folder containing the models and their evaluation results.
        output_csv (str): Path to save or update the CSV file.
    """
    # Initialize an empty list to hold new rows for the CSV
    rows = []

    # Load existing CSV if it exists
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
    else:
        df = pd.DataFrame()

    # Create a dictionary for quick look-up of existing hashes (ids)
    existing_hashes = {} if df.empty else dict(zip(df['id'], df.index))

    # Walk through the models folder and its subfolders
    # for repetition in [f'V{i}' for i in range(2, 9)]:  # Example with repetitions V1 and V2
    for repetition in ['V9', 'V10']:
        repetition_folder = f'{models_folder}/{repetition}'

        if not os.path.exists(repetition_folder):
            continue
        
        for pretrain in ["f3", "f3_norm", "seam_ai", "seam_ai_norm", "both", "both_N", "COCO", "IMAGENET", "sup", "seg"]:
            for data in ["f3", "f3_N", "seam_ai", "seam_ai_N"]:
                for cap in [0.01, 0.1, 0.5, 1]:

                    # Determine model name based on pretrain and other parameters
                    if pretrain in ['f3', 'seam_ai', 'both', 'f3_norm', 'seam_ai_norm', 'both_N']:
                        mode = 'byol'
                        model_name = f'{repetition}_pre_{pretrain}_train_{data}_cap_{cap*100:.0f}%'
                    elif pretrain == 'seg':
                        mode = 'seg'
                        model_name = f'{repetition}_pre_{pretrain}_train_{data}_cap_{cap*100:.0f}%'
                    elif pretrain == 'COCO':
                        mode = 'coco'
                        model_name = f'{repetition}_pre_COCO_train_{data}_cap_{cap*100:.0f}%'
                    elif pretrain == 'IMAGENET':
                        mode = 'imagenet'
                        model_name = f'{repetition}_pre_IMAGENET_train_{data}_cap_{cap*100:.0f}%'
                    elif pretrain == 'sup':
                        mode = 'supervised'
                        model_name = f'{repetition}_sup_{data}_cap_{cap*100:.0f}%'
                        
                    file_path = f'{repetition_folder}/{model_name}'
                    
                    # Map data to the correct root directory
                    if data == 'f3':
                        root_dir = '../../asml/datasets/tiff_data/f3_segmentation'
                    elif data == 'seam_ai':
                        root_dir = '../../asml/datasets/tiff_data/seam_ai'
                    elif data == 'f3_N':
                        root_dir = '../../asml/datasets/tiff_data/f3_segmentation_N'
                    elif data == 'seam_ai_N':
                        root_dir = '../../asml/datasets/tiff_data/seam_ai_N'
                    else:
                        raise ValueError('Data not found. Must be one of "f3" or "seam_ai".')

                    # Generate a hash ID for the model
                    file_id = hashlib.md5(model_name.encode()).hexdigest()

                    # Call eval_func to get IoU and F1 values
                    output = eval_func(
                        import_name=model_name,
                        mode=mode,
                        repetition=repetition,
                        root_dir=root_dir
                    )

                    print(f"Evaluating {model_name} -> Output: {output}")

                    # Extract IoU and F1 values
                    iou_value = output['iou']
                    f1_value = output['f1']

                    if iou_value is not None and f1_value is not None:
                        # Create a new row for this model
                        row = {
                            "id": file_id,
                            "model_name": model_name,
                            "path": file_path,
                            "repetition": repetition,
                            "backbone": pretrain,
                            "downstream": data,
                            "cap": cap,
                            "iou": iou_value,
                            "f1": f1_value
                        }

                        # Overwrite if the hash exists (update the existing row), otherwise append as a new row
                        if file_id in existing_hashes:
                            # Update the row in the DataFrame
                            df.loc[existing_hashes[file_id]] = row
                        else:
                            # Add the new row to the list of rows to append
                            rows.append(row)

    # Convert the new rows to a DataFrame and append them to the existing DataFrame
    if rows:
        new_df = pd.DataFrame(rows)
        df = pd.concat([df, new_df], ignore_index=True)

    # Save the updated DataFrame to CSV
    df.to_csv(output_csv, index=False)
    print(f"CSV file updated and saved as {output_csv}")

# Example of usage:
generate_csv_from_models(
    models_folder="../saves/models",  # The path to your models folder
    output_csv="evaluation_results.csv"  # Save or update the results as a CSV file
)
