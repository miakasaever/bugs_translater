import os
import json
import csv
from openai import OpenAI
import time
from collections import defaultdict


def tokenize_sentence(sentence):
    """
    壮语句子分词
    """
    for punc in "，。、；！？「」『』【】（）《》“”…,.;?!":
        sentence = sentence.replace(punc, " ")
    words = sentence.split()
    return [word.strip() for word in words if word.strip()]


def create_acquire_json(acqure_json_path):
    dict = defaultdict()
    keys = ["model", "temperature", "max_tokens"]
    for key in keys:
        dict[key] = input(f"acquire need{key}")
    try:
        with open(acqure_json_path, 'w', encoding='utf-8') as f:
            json.dump(dict, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(e)


def acquire(prompt, acquire_json, max_retries=3):
    """
    acquire LLMS
    """
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=acquire_json['model'],
                messages=[
                    {"role": "system", "content": "你是一位精通壮语-汉语翻译的专家"},
                    {"role": "user", "content": f"{prompt}"},
                ],
                temperature=acquire_json['temperature'],
                max_tokens=acquire_json['max_tokens']
            )
            result = response.choices[0].message.content.strip()

            if "\n" in result:
                result = result.split("\n")[0].strip()
            return result
        except Exception as e:
            time.sleep(2)

    return "Fail to translate"


def modify_csv(file_path, row_index, new_value):
    """
    修改CSV文件
    """
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            data = list(reader)
        if row_index < len(data):
            data[row_index][1] = new_value
        else:
            return
        with open(file_path, mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)

    except Exception as e:
        print("Error")


def save_translation_results(results, output_path):
    """
    保存翻译结果到JSON文件
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Error")
