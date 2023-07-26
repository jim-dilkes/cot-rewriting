# code: gpt4
import os
import json
import pandas as pd

def get_all_json_files(dir_name):
    # Traverse through all subdirectories and get all details.json files
    json_files = []
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if file == "details.json":
                json_files.append(os.path.join(root, file))
    return json_files

def compile_data(json_files):
    data = []
    for file in json_files:
        with open(file, 'r') as f:
            contents = json.load(f)
            # Ensure contents is a list for uniform handling
            if not isinstance(contents, list):
                contents = [contents]
            for content in contents:
                # Flatten cost into main dict
                cost_data = content.pop('Cost')
                for cost_type, cost_values in cost_data.items():
                    # Check if the cost_values is a dict before trying to extract items
                    if isinstance(cost_values, dict):
                        for cost_measure, value in cost_values.items():
                            content[f'Cost_{cost_type}_{cost_measure}'] = value
                    else:
                        # if cost_values is not a dict, it's directly stored
                        content[f'Cost_{cost_type}'] = cost_values
                data.append(content)
    return data

def main():
    json_files = get_all_json_files('.')
    data = compile_data(json_files)
    df = pd.DataFrame(data)

    # replace new line characters with \n in string type columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: x.replace('\n', '\\n') if isinstance(x, str) else x)

    # Save dataframe to csv
    df.to_csv('all_results.csv', index=False)

if __name__ == "__main__":
    main()
