import re
from math import isclose
from collections import defaultdict
from rapidfuzz import fuzz
from difflib import SequenceMatcher


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


# def anls_compute(groundtruth, prediction, threshold=0.5):
#     dist = levenshtein_distance(groundtruth, prediction)
#     length = max(len(groundtruth.upper()), len(prediction.upper()))
#     value = 0.0 if length == 0 else float(dist) / float(length)
#     anls = 1.0 - value
#     if anls<=threshold:
#         anls = 0.0
#     return anls

def anls_compute(ground_truth_answers,predicted_answer, threshold=0.5):
    """
    Compute ANLS by taking the best match from multiple ground truth answers.
    
    :param predicted_answer: The answer predicted by the model.
    :param ground_truth_answers: A list of valid ground truth answers.
    :param threshold: The minimum similarity required for a valid match.
    :return: ANLS score (between 0 and 1).
    """
    max_similarity = 0

    for gt_answer in ground_truth_answers:
        similarity = fuzz.ratio(predicted_answer.lower(), gt_answer.lower()) / 100  # Normalize to [0,1]
        max_similarity = max(max_similarity, similarity)

    return max(0, 1 - max(0, 1 - max_similarity)) if max_similarity >= threshold else 0

def is_float_equal(reference, prediction, include_percentage: bool = False, is_close: float = False) -> bool:
    def get_precision(gt_ans: float) -> int:
        precision = 3
        if '.' in str(gt_ans):
            precision = len(str(gt_ans).split('.')[-1])
        return precision

    reference = float(str(reference).strip().rstrip("%").strip())
    try:
        prediction = float(str(prediction).strip().rstrip("%").strip())
    except:
        return False

    if include_percentage:
        gt_result = [reference / 100, reference, reference * 100]
    else:
        gt_result = [reference]
    for item in gt_result:
        try:
            if is_close:
                if isclose(item, prediction, rel_tol=0.01):
                    return True
            precision = max(min(get_precision(prediction), get_precision(item)), 2)
            if round(prediction, precision) == round(item, precision):
                return True
        except Exception:
            continue
    return False


def get_clean_string(s):
    s = str(s).lower().strip()
    if s.endswith("mile"):
        s.rstrip("mile").strip()
    if s.endswith("miles"):
        s.rstrip("miles").strip()
    if s.endswith("million"):
        s.rstrip("million").strip()
    # remove parenthesis
    s = re.sub(r'\s*\([^)]*\)', '', s).strip()
    # remove quotes
    s = re.sub(r"^['\"]|['\"]$", "", s).strip()
    s = s.strip().lstrip("$").strip()
    s = s.strip().rstrip("%").strip()
    return s


def is_exact_match(s):
    flag = False
    # Website
    if "https://" in s:
        flag = True
    # code file
    if s.endswith(".py") or s.endswith("ipynb"):
        flag = True
    if s.startswith("page"):
        flag = True
    # telephone number
    if re.fullmatch(r'\b\d+(-\d+|\s\d+)?\b', s):
        flag = True
    # time
    if "a.m." in s or "p.m." in s:
        flag = True
    # YYYY-MM-DD
    if re.fullmatch(r'\b\d{4}[-\s]\d{2}[-\s]\d{2}\b', s):
        flag = True
    # YYYY-MM
    if re.fullmatch(r'\b\d{4}[-\s]\d{2}\b', s):
        flag = True
    # Email address
    if re.fullmatch(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', s):
        flag = True
    return flag


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False
    
def fuzzy_match(a, b):
    return SequenceMatcher(None, a, b).ratio() > 0.8  # 阈值可调

import ast
import re

import ast
import re

def safe_literal_eval(s):
    """
    尝试使用 ast.literal_eval 解析答案字符串，
    如果失败（例如由于未闭合的括号或内部未转义的单引号），则尝试进行简单补全和转义。
    """
    try:
        return ast.literal_eval(s)
    except SyntaxError as e:
        # 尝试补全缺少的右括号：如果 s 开头为 '[' 但结尾没有 ']', 则加上
        s_fixed = s.strip()
        if s_fixed.startswith("[") and not s_fixed.endswith("]"):
            s_fixed = s_fixed + "]"
        # 对位于字母或数字中间的单引号进行转义
        s_fixed = re.sub(r"(?<=\w)'(?=\w)", r"\\'", s_fixed)
        try:
            return ast.literal_eval(s_fixed)
        except Exception as e2:
            print("safe_literal_eval: 无法解析答案，经过修正后依然失败。修正后的字符串为：")
            print(s_fixed)
            return None




def eval_score(gt, pred, answer_type):
    if answer_type == "Int":
        try:
            gt, pred = int(gt), int(float(pred))
        except:
            pred = ""
        score = (gt == pred)
    elif answer_type == "Float":
        try:
            gt = float(get_clean_string(str(gt)))
            pred = float(get_clean_string(str(pred)))
        except:
            pred = ""
        score = is_float_equal(gt, pred, include_percentage=True, is_close=True)
    elif answer_type in ["None"]:
        if isinstance(gt, str) and gt.startswith("[") and gt.endswith("]"):
            try:
                gt_list = eval(gt)
                gt = " ".join(gt_list) if isinstance(gt_list, list) else gt
                print(gt)
            except:
                pass
        if isinstance(pred, str) and pred.startswith("[") and pred.endswith("]"):
            try:
                pred_list = eval(pred)
                pred = " ".join(pred_list) if isinstance(pred_list, list) else pred
            except:
                pass
        gt = get_clean_string(gt)
        pred = get_clean_string(pred)
        score = (gt == pred)

    elif answer_type in ["Str"]:
        if isinstance(gt, str) and gt.startswith("[") and gt.endswith("]"):
            try:
                gt_list = eval(gt)
                gt = " ".join(gt_list) if isinstance(gt_list, list) else gt
                print(gt)
            except:
                pass
        if isinstance(pred, str) and pred.startswith("[") and pred.endswith("]"):
            try:
                pred_list = eval(pred)
                pred = " ".join(pred_list) if isinstance(pred_list, list) else pred
            except:
                pass
        gt = get_clean_string(gt)
        pred = get_clean_string(pred)

        if type(pred) == str:
            if gt in pred:
                score = 1.0
            else:
                if is_exact_match(gt):
                    score = (gt == pred)
                else:
                    score = anls_compute(gt, pred)
    else:
        if isinstance(gt, str) and gt.startswith("["):
            gt = eval(gt)
        if not isinstance(gt, list):
            gt = [gt]
        if isinstance(pred, str) and pred.startswith("["):
            pred = safe_literal_eval(pred)
            if pred is None:
                score=0.0

        if not isinstance(pred, list):
            pred = [pred]
        
        # 如果任一列表为空，直接返回 0.0
        if len(gt) == 0 or len(pred) == 0:
            return 0.0
        if len(gt)!=len(pred):
            score = 0.0
        else:
            gt_clean = [get_clean_string(a) for a in gt]
            pred_clean = [get_clean_string(a) for a in pred]
            # print(gt_clean, pred_clean)

            matched = []
            unmatched_pred = pred_clean.copy()

            for gt_item in gt_clean:
                found = False
                for pred_item in unmatched_pred:
                    if fuzzy_match(gt_item, pred_item):
                        matched.append(gt_item)
                        unmatched_pred.remove(pred_item)
                        found = True
                        break

            score = len(matched) / len(gt_clean)
            # print(score)

    return float(score)



def eval_acc_and_f1(samples):
    evaluated_samples = [sample for sample in samples if "score" in sample]
    if not evaluated_samples:
        return 0.0, 0.0
    
    acc = sum([sample["score"] for sample in evaluated_samples])/len(evaluated_samples)
    try:
        recall = sum([sample["score"] for sample in evaluated_samples if sample["answer"]!="Not answerable"])/len([sample for sample in evaluated_samples if sample["answer"]!="Not answerable"])
        precision = sum([sample["score"] for sample in evaluated_samples if sample["answer"]!="Not answerable"])/len([sample for sample in evaluated_samples if sample["pred"]!="Not answerable"])
        f1 = 2*recall*precision/(recall+precision) if (recall+precision)>0.0 else 0.0
    except:
        f1 = 0.0
    
    return acc, f1

def precision_recall_f1(pred, gt):
    pred_set = set(pred)
    gt_set = set(gt)
    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(gt_set) if gt_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1

import math

def eval_retrieval(samples, show_path=None):
    hit1 = hit3 = hit5 = 0
    prec_sum = rec_sum = f1_sum = ndcg_sum = 0.0
    total = 0

    for sample in samples:
        pred_pages = sample.get("pred_page", [])
        gt_pages = sample.get("evidence_pages", [])

        if isinstance(pred_pages, str):
            try:
                pred_pages = eval(pred_pages)
            except:
                pred_pages = []
        if isinstance(gt_pages, str):
            try:
                gt_pages = eval(gt_pages)
            except:
                gt_pages = []

        if not pred_pages or not gt_pages:
            continue

        # 截断预测为前5个
        top5 = pred_pages[:4]
        gt_set = set(gt_pages)

        # Hit@k
        if top5[:1] and top5[0] in gt_set:
            hit1 += 1
        if any(p in gt_set for p in top5[:3]):
            hit3 += 1
        if any(p in gt_set for p in top5):
            hit5 += 1

        # Precision, Recall, F1
        prec, rec, f1 = precision_recall_f1(top5, gt_set)
        prec_sum += prec
        rec_sum += rec
        f1_sum += f1

        # NDCG@5
        dcg = 0.0
        for i, p in enumerate(top5):
            if p in gt_set:
                dcg = 1 / math.log2(i + 2)
                break
        ndcg_sum += dcg

        total += 1

    print("Total Samples:", total)
    print("Hit@1: {:.2%}".format(hit1 / total))
    print("Hit@3: {:.2%}".format(hit3 / total))
    print("Hit@4: {:.2%}".format(hit5 / total))
    print("Avg Precision: {:.2%}".format(prec_sum / total))
    print("Avg Recall: {:.2%}".format(rec_sum / total))
    print("Avg F1: {:.2%}".format(f1_sum / total))
    print("Avg NDCG@4: {:.4f}".format(ndcg_sum / total))

    with open(show_path, 'a') as f:
        f.write("--------------------------------\n")
        f.write("Retrieval-related Scores\n")
        f.write("Retrieval total samples:{}\n".format(total))
        f.write("Hit@1:{}\n".format(hit1 / total))
        f.write("Hit@3:{}\n".format(hit3 / total))
        f.write("Hit@4:{}\n".format(hit5 / total))
        f.write("Avg Precision:{}\n".format(prec_sum / total))
        f.write("Avg Recall:{}\n".format(rec_sum / total))
        f.write("Avg F1:{}\n".format(f1_sum / total))
        f.write("Avg NDCG@4:{}\n".format(ndcg_sum / total))

def show_results(samples, show_path=None):
    for sample in samples:
        sample["evidence_pages"] = eval(sample["evidence_pages"])
        sample["evidence_sources"] = eval(sample["evidence_sources"])
    
    with open(show_path, 'w') as f:
        acc, f1 = eval_acc_and_f1(samples)
        f.write("Overall Acc: {} | Question Number: {}\n".format(acc, len(samples)))
        f.write("Overall F1-score: {} | Question Number: {}\n".format(f1, len(samples)))
        f.write("-----------------------\n")

        #####################
        acc_single_page, _ = eval_acc_and_f1([sample for sample in samples if len(sample["evidence_pages"])==1])
        acc_multi_page, _ = eval_acc_and_f1([sample for sample in samples if len(sample["evidence_pages"])!=1 and sample["answer"]!="Not answerable"])
        acc_neg, _ = eval_acc_and_f1([sample for sample in samples if sample["answer"]=="Not answerable"])

        f.write("Single-page | Accuracy: {} | Question Number: {}\n".format(
            acc_single_page, len([sample for sample in samples if len(sample["evidence_pages"])==1])
        ))
        f.write("Cross-page | Accuracy: {} | Question Number: {}\n".format(
            acc_multi_page, len([sample for sample in samples if len(sample["evidence_pages"])!=1 and sample["answer"]!="Not answerable"])
        ))
        f.write("Unanswerable | Accuracy: {} | Question Number: {}\n".format(
            acc_neg, len([sample for sample in samples if sample["answer"]=="Not answerable"])
        ))
        f.write("-----------------------\n")

        #####################
        source_sample_dict, document_type_dict = defaultdict(list), defaultdict(list)
        for sample in samples:
            for answer_source in sample["evidence_sources"]:
                source_sample_dict[answer_source].append(sample)
            document_type_dict[sample["doc_type"]].append(sample)
        for type, sub_samples in source_sample_dict.items():
            f.write(
                "Evidence Sources: {} | Accuracy: {} | Question Number: {}\n".format(type, eval_acc_and_f1(sub_samples)[0], len(sub_samples))
            )

        f.write("-----------------------\n")
        for type, sub_samples in document_type_dict.items():
            f.write(
                "Document Type: {} | Accuracy: {} | Question Number: {}\n".format(type, eval_acc_and_f1(sub_samples)[0], len(sub_samples))
            )