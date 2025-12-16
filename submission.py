import os
import json

input_folder = "mvp_sspro_qwen3vl_32b"  # <--- Modify your folder path
output_json = "en_results_qwen3vl32b_mvp.json"
def process_json_files(input_folder, output_json):
    # List to hold the processed data
    models_data = {}

    # Iterate over all files in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)

                # Extract necessary details
                model_name = filename.rsplit('.', 1)[0]  # Remove the extension
                leaderboard_detailed_style = data["metrics"]["leaderboard_detailed_style"]

                # Collect the application results in the required format
                application_results = {}
                for app, values in leaderboard_detailed_style.items():
                    app_name = app.split("app:", 1)[-1]
                    application_results[app_name] = {
                        "icon": values["icon_acc"],
                        "text": values["text_acc"],
                        "avg": values["action_acc"]
                    }

                leaderboard_simiple = data["metrics"]["leaderboard_simple_style"]

                # Read data directly from the leaderboard
                group_results = {}
                group_names = ["Dev", "Creative", "CAD", "Scientific", "Office", "OS"]

                # Loop through each group and extract the relevant information
                for group in group_names:
                    group_results[group] = {
                        "icon": leaderboard_simiple[f"group:{group}"]["icon_acc"],
                        "text": leaderboard_simiple[f"group:{group}"]["text_acc"],
                        "avg": leaderboard_simiple[f"group:{group}"]["action_acc"],
                    }

                # Add the results to the dictionary
                models_data[model_name] = {
                    "link": "",
                    "description": "",
                    "results": {
                        "group": group_results,
                        "application": application_results,
                        "overall": {
                            "icon": data["metrics"]["overall"]["icon_acc"],
                            "text": data["metrics"]["overall"]["text_acc"],
                            "avg": data["metrics"]["overall"]["action_acc"]
                        }
                    },
                    
                }

    # Save the processed data to a new JSON file
    with open(output_json, 'w') as output_file:
        json.dump(models_data, output_file, indent=4)

    print(f"Data has been saved to {output_json}")


# Process the JSON files
process_json_files(input_folder, output_json)