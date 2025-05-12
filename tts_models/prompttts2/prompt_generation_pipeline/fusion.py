# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import math
import random
import pandas as pd
from itertools import product
from tqdm import tqdm

from prompt_pl import prompt_word_path, use_placeholder


def gen_num_categories(count_categories: list[int]) -> list[list[int]]:
    '''Generate entire classification combinations.'''
    return list(product(*[range(n) for n in count_categories]))


def fusion(
    categories: dict[str, set[str]], 
    csv_lambda: dict[str, int], 
    fu_data_path: str
) -> None:
    '''
    Read the prompt words and generate the fusion dataset.
    Args:
        categories: The categories.
        csv_lambda: The lambda function of csv file.
        fu_data_path: The fusion dataset path.
    Returns:
        Labeled CSV file.
    '''
    # Initialize the prompt words
    root_path = os.getcwd()  # return the path from root folder of system
    prompt_word_rpath = os.path.join(root_path, prompt_word_path)

    # Browse the categories dictionary
    for category in categories.keys():
        for sub_category in categories[category].keys():
            while True:
                try:
                    with open(os.path.join(prompt_word_rpath, sub_category + '.txt'), 'r') as f:
                        lines = f.readlines()
                    if len(lines) == 0:
                        continue
                    categories[category][sub_category].update(
                        filter(None, (line.strip().lower() for line in lines))
                    )
                    break
                except:
                    continue
    
    # Process the prompt words
    normal_categories = [list(categories[key])[0] for key in categories.keys()]
    count_categories = [len(categories[key].keys()) for key in categories.keys()]
    num_categories = gen_num_categories(count_categories)

    # Create columns for output csv file
    columns = list(categories.keys())
    columns.extend(['vietnamese', 'prompt_class_num'])  # additional columns
    answer_list = list()

    for num_category in tqdm(num_categories):
        for key in csv_lambda.keys():
            sample_num = csv_lambda[key]
            aug_csv = pd.read_csv(key)
            # str concatenation from values of 'num_category'
            prompt_class_num = '-'.join([str(num_category_i) for num_category_i in num_category])

            if 'pl' == key.split('.')[0].split('_')[-1]:  # check 'pl' file
                english_list = list()
                while len(english_list) < sample_num:
                    prompt_class_num_place = '-'.join(
                        ['U' if num_category_i == 0 and random.random() < use_placeholder else 'P' for num_category_i in num_category]
                    )  # P for placeholder, U for unplaceholder
                    if prompt_class_num_place not in aug_csv['prompt_class_num_place'].unique():
                        continue
                    # Get sentence from csv file having the same prompt_class_num_place valu
                    sentence = aug_csv[aug_csv['prompt_class_num_place'] == prompt_class_num_place]['vietnamese'].sample(1).item()
                    for i, place_i in enumerate(prompt_class_num_place.split('-')):
                        if place_i == 'P':
                            category = list(categories.keys())[i]
                            sub_category = list(categories[category].keys())[num_category[i]]
                            word = random.choice(list(categories[category][sub_category]))
                            # Replace the placeholder with the randomly words from categories 
                            sentence = sentence.replace(f'[{category}]', word)
                    english_list.append(sentence)
            else:
                english_list = list(
                    aug_csv[aug_csv['prompt_class_num'] == prompt_class_num]['english'].sample(sample_num)
                )  # get sentence from 'vietnamese' column in csv file, having the same prompt_class_num value
            
            for english in english_list:
                answer = num_category[:]        # copy num_category list
                answer.append(english)          # append vietnamese sentence to answer list
                answer.append(prompt_class_num) # append the sentence and its category to answer list
                answer_list.append(answer)      # append answer list to answer_list list

    pd.DataFrame(answer_list, columns=columns).to_csv(fu_data_path, index=None)
    