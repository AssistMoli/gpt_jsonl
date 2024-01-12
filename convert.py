# https://medium.com/@garethcull/fine-tuning-ai-models-a-practical-guide-for-beginners-dc313b2e0f76


def format_for_fine_tuning(data):

    """

    Background: This function helps format data into a list of dicts into the required shape for fine tuning

    Params:
    data (list): list of dicts

    Returns:
    training_data_list (list): a list in the proper format for converting to jsonl

    """

    training_data_list = []

    for x in data:

        updated_data = {
            "messages": [
                {
                    "role": "system",
                    "content": x['system_message']
                },
                {
                    "role": "user",
                    "content": x['user_content']
                },
                {
                    "role": "assistant",
                    "content": x['assistant_content']
                }
            ]
        }

        training_data_list.append(updated_data)

    print(training_data_list)

    return training_data_list

def convert_to_jsonl_and_save(data_list, filename, encoding):

    """
    Background:
    This function converts the data_list provided into a jsonl file

    Params:
    data_list (list): a list of a dict ready to convert to jsonl
    filename (str): the name of the filename we want to convert

    """

    with open(filename, 'w', encoding=encoding) as file:

        for data_dict in data_list:

            # Convert each dictionary to a JSON string
            json_str = json.dumps(data_dict, ensure_ascii=False)

            json_str = (json_str + '\n')

            # Write the JSON string followed by a newline character to the file
            file.write((json_str))

    print(f"Data has been written to {filename}")


# Import modules
import json
import pandas as pd

title = 'sample'

import chardet
with open(f'{title}.csv', 'rb') as file:
    result = chardet.detect(file.read())
encoding = result['encoding']
print(f"The detected encoding is: {encoding}")

# Open csv
csv_training_data = pd.read_csv(f'{title}.csv', encoding=encoding)

# Convert to JSON String
csv_training_str = csv_training_data.to_json(orient='records')

# parse JSON string and convert it into a list of python dictionaries
training_dict = json.loads(csv_training_str)

# format dict into shape for fine tuning
fine_tune_formatted_data = format_for_fine_tuning(training_dict)

# Save as jsonl
convert_to_jsonl_and_save(fine_tune_formatted_data, f'{title}.jsonl', encoding)















