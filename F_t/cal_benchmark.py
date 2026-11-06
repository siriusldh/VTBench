import os
import json

# Define metrics to calculate averages
metrics = ['tf_similarity', 'precision', 'recall', 'edit_dist']

# Define root directory
root_dir = os.path.join(os.getenv('VTBENCH_DATA_ROOT', './data'), 'ocr_results')

# Iterate over each folder
for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    if os.path.isdir(folder_path):
        folder_metrics_sum = {metric: 0.0 for metric in metrics}
        file_count = 0

        # Iterate over text files in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith('.txt'):
                with open(file_path, 'r') as file:
                    file_metrics_sum = {metric: 0.0 for metric in metrics}
                    line_count = 0

                    # Read each line
                    for line in file:
                        # Parse the first JSON object in the line
                        first_part = line.split('|')[0].strip()

                        # Check if first_part is '0'
                        if first_part == '0':
                            # If '0', set all metrics to 0
                            data = {metric: 0.0 for metric in metrics}
                        else:
                            data = json.loads(first_part.replace("'", "\""))

                        # Accumulate metric values
                        for metric in metrics:
                            file_metrics_sum[metric] += data.get(metric, 0.0)
                        line_count += 1

                    # Calculate average for the file
                    if line_count > 0:
                        for metric in metrics:
                            folder_metrics_sum[metric] += file_metrics_sum[metric] / line_count
                        file_count += 1

        # Calculate average for the folder
        if file_count > 0:
            print(f"Folder: {folder}","file_count:",file_count)
            total = 0
            for metric in metrics:
                average = folder_metrics_sum[metric] / file_count
                total += average
                print(f"  {metric} average: {average}")
            print(f"Folder: {folder}","metric:",total/len(metrics))
