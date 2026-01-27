import json
import os

from PIL import Image

from loguru import logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

IMAGES_PATH = "/data-net/storage/users/evivoli/CoMix/data/datasets/Manga109/images"

def mangavqa_doc_to_visual(doc):
    book = doc["title"]
    image_num = doc["image_number"]
    
    image = Image.open(os.path.join(IMAGES_PATH, book, f"{image_num:03d}.jpg"))

    return [image.convert("RGB")]

def mangavqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"

def mangavqa_test_process_results(doc, results):
    pred = results[0]
    questionId = doc["questionId"]
    
    results_dict = {
        "anls": {"questionId": int(questionId), "answer": pred}, 
        "f1": {"questionId": int(questionId), "answer": pred},
        "exact_match": {"questionId": int(questionId), "answer": pred},
        "submission": {"questionId": int(questionId), "answer": pred},
        }
    
    return results_dict

def mangavqa_test_aggregate_results(results, args):
    # save results as json
    path = generate_submission_file("docvqa_test_for_submission.json", args)
    with open(path, "w") as f:
        json.dump(results, f)
    logger.info(f"Results saved to {path}")


