import os
import pandas as pd
import hashlib
from evaluate import eval_func

def generate_csv_from_models(models_folder, output_csv="evaluation_results.csv"):
    """
    Generates a CSV file containing evaluation results for each model, with IoU and F1 as evaluation metrics.
    
    Parameters:
        models_folder (str): Path to the folder containing the models and their evaluation results.
        output_csv (str): Path to save the generated CSV file.
    """
    # Initialize an empty list to hold the rows of the CSV
    rows = []

    # Walk through the models folder and its subfolders
    for repetition in [f'V{i}' for i in range(1,2)]:  # For repetitions V1 to V10
        repetition_folder = f'{models_folder}/{repetition}'

        print(repetition)
        print(models_folder)
        print(repetition_folder)
        
        for pretrain in ["f3", "f3_norm", "seam_ai", "seam_ai_norm", "both", "both_N", "COCO", "IMAGENET", "sup", "seg"]:
            for data in ["f3", "f3_N", "seam_ai", "seam_ai_N"]:
                for cap in [0.01, 0.1, 0.5, 1]:

                    if pretrain  in ['f3', 'seam_ai', 'both', 'f3_norm', 'seam_ai_norm', 'both_N']:
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
                    
                    
                    if data == 'f3':
                        root_dir = '../../asml/datasets/tiff_data/f3_segmentation'
                        dataset = 'f3'
                    elif data == 'seam_ai':
                        root_dir = '../../asml/datasets/tiff_data/seam_ai'
                        dataset = 'f3'
                    elif data == 'f3_N':
                        root_dir = '../../asml/datasets/tiff_data/f3_segmentation_N'
                        dataset = 'seam_ai'
                    elif data == 'seam_ai_N':
                        root_dir = '../../asml/datasets/tiff_data/seam_ai_N'
                        dataset = 'seam_ai'
                    else:
                        raise ValueError('Data not found. Must be one of "f3" or "seam_ai"')
                                        
                    # print(file_path)
                        
                    output = eval_func(
                        import_name = model_name,
                        mode = mode,
                        repetition = repetition,
                        root_dir = root_dir
                    )
                    
                    print(output)
                    
                    iou_value = output['iou']
                    f1_value = output['f1']
                        
                    if iou_value is not None and f1_value is not None:
                        # Generate an ID from the file name using hash
                        file_id = hashlib.md5(model_name.encode()).hexdigest()
                        
                        # Append row with the required data
                        rows.append({
                            "id": file_id,
                            "model_name": model_name,
                            "path": file_path,
                            "repetition": repetition,
                            "backbone": pretrain,
                            "downstream": data,
                            "cap": cap,
                            "iou": iou_value,
                            "f1": f1_value
                        })

    # Convert the rows to a pandas DataFrame
    df = pd.DataFrame(rows)
    
    # Save the DataFrame to CSV
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved as {output_csv}")

# Example of usage:
generate_csv_from_models(
    models_folder="../saves/models",  # The path to your models folder
    output_csv="evaluation_results.csv"  # Save the results as a CSV file
)
