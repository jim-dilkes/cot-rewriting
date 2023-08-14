# code: adapted from gpt4
import os
import json
import pandas as pd
import argparse

def get_all_json_files(dir_name):
    # Traverse through all subdirectories and get all details.json files
    json_files = []
    for root, dirs, files in os.walk(dir_name):
        json_files.extend(
            os.path.join(root, file)
            for file in files
            if file == "details.json"
        )
    return json_files


def compile_data(json_files):
    old_models_keys = ['Prompt','Prompt system message','Answer system message']
    data = []
    for file in json_files:
        with open(file, 'r') as f:
            contents = json.load(f)
            # Ensure contents is a list for uniform handling
            if not isinstance(contents, list):
                contents = [contents]
            for content in contents:
                # Handle model details from old details.json files
                old_models_items = [content.pop(key) for key in old_models_keys if key in content]
                content['Models'] = old_models_items or None
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=".", help="")
    args = parser.parse_args()
    
    json_files = get_all_json_files(args.results_dir)
    data = compile_data(json_files)
    df = pd.DataFrame(data)

    # replace new line characters with \n in string type columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: x.replace('\n', '\\n') if isinstance(x, str) else x)

    # Save dataframe to csv
    df.to_csv(os.path.join(args.results_dir,"all_results.csv"), index=False)

if __name__ == "__main__":
    main()
