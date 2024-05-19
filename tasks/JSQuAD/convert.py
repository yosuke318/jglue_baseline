import json

def convert_json_format(data):
    converted_data = []
    for item in data:
        for paragraph in item["paragraphs"]:
            for qa in paragraph["qas"]:
                converted_item = {
                    "context": paragraph["context"],
                    "question": qa["question"],
                    "answers": {
                        "text": [qa["answers"][0]["text"]],
                        "answer_start": [qa["answers"][0]["answer_start"]]
                    }
                }
                converted_data.append(converted_item)
    return converted_data


def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def save_json_file(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    data = load_json_file('./training.json')  # todo 訓練用と検証用書き換える
    converted_data = convert_json_format(data['train'])
    print(converted_data)
    save_json_file(converted_data, "./new_training.json")