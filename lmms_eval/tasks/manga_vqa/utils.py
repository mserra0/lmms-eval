import json
import os
import hashlib

from PIL import Image

from loguru import logger

from lmms_eval.api.metrics import levenshtein_distance

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

IMAGES_PATH = "/data-net/storage/users/evivoli/CoMix/data/datasets/Manga109/images"

def create_deterministic_id(doc):
    book = doc["title"]
    image_num = doc["image_number"]
    
    question= doc['question']

    question_hash = hashlib.sha256(question.encode('utf-8')).hexdigest()[:8]
    
    doc_id = f"{book}_{image_num:03d}_{question_hash}"
    
    return doc_id

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
    
    questionId = create_deterministic_id(doc)
    
    # Process prediction
    pred = pred.strip()

    # Get ground truths
    ground_truths = [doc["answer"]] if isinstance(doc["answer"], str) else doc["answer"]

    # Calculate ANLS
    anls_values = []
    for answer in ground_truths:
        gt_answer = " ".join(answer.strip().lower().split())
        det_answer = " ".join(pred.strip().lower().split())
        
        dist = levenshtein_distance(gt_answer, det_answer)
        length = max(len(answer.upper()), len(pred.upper()))
        anls_values.append(0.0 if length == 0 else float(dist) / float(length))
    
    anls_score = 1 - min(anls_values) if anls_values else 0
    if anls_score < 0.5:
        anls_score = 0

    # Calculate F1
    import string
    def normalize_answer(s):
        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            return "".join(ch for ch in text if ch not in set(string.punctuation))

        def lower(text):
            return text.lower()

        return white_space_fix(remove_punc(lower(s)))
    
    f1_scores = []
    for answer in ground_truths:
        pred_tokens = normalize_answer(pred).split()
        gt_tokens = normalize_answer(answer).split()
        
        common = set(pred_tokens) & set(gt_tokens)
        num_same = len(common)
        
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            f1_scores.append(int(pred_tokens == gt_tokens))
            continue
            
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        f1_scores.append(f1)
    
    f1_score = max(f1_scores) if f1_scores else 0

    # Calculate Exact Match
    em_scores = []
    for answer in ground_truths:
        em_scores.append(1.0 if normalize_answer(pred) == normalize_answer(answer) else 0.0)
    exact_match_score = max(em_scores) if em_scores else 0

    results_dict = {
        "anls": anls_score, 
        "f1": f1_score,
        "exact_match": exact_match_score,
        "submission": {"questionId": questionId, "answer": pred},
        }
    
    return results_dict

def mangavqa_test_aggregate_results(results, args):
    # save results as json
    path = generate_submission_file("manga_vqa_test_submission.json", args)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    logger.info(f"Results saved to {path}")


