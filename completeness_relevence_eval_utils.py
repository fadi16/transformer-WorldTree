import json
from typing import Dict, List

from nltk import RegexpTokenizer
from rouge_score import rouge_scorer

from postprocess import postprocess_fact

RELEVANCE_RATING_JSON = "./data/v2-proper-data/wt21-dev.teacher-ratings.json"
FACTS_BANK = "./data/v2-proper-data/tablestore_shared.json"
EXPLANATIONS_BANK = "./data/v2-proper-data/dev_set_shared.json"


class AlignedFact:
    def __init__(self, id, fact_str, relevance_score, is_golden, alignment_score):
        self.id = id
        self.fact_str = fact_str.strip()
        self.relevance_score = relevance_score
        self.is_golden = is_golden
        self.alignment_score = alignment_score


class FactAlignerAndRelevanceScorer:
    def __init__(self):
        self.regexp_tokenizer = RegexpTokenizer(r'\w+')

        # qid -> factId -> data
        self.question_id_to_facts_relevance_scores: Dict[str, Dict[str, Dict]] = {}

        self.all_facts = []
        self.all_fact_ids = []

        self.question_id_to_golden_facts_ids: Dict[str, List[str]] = {}

        with open(RELEVANCE_RATING_JSON) as json_file:
            relevance_rating_corpus = json.load(json_file)["rankingProblems"]

        for question_dict in relevance_rating_corpus:
            qid = question_dict["qid"]
            new_fact_dict = {}
            for fact_dict in question_dict["documents"]:
                fact_id = fact_dict["uuid"]
                is_gold = True if int(fact_dict["isGoldWT21"]) == 1 else False
                relevance_score = fact_dict["relevance"]

                new_fact_dict[fact_id] = {"relevance_score": relevance_score, "is_golden": is_gold}
            self.question_id_to_facts_relevance_scores[qid] = new_fact_dict

        with open(FACTS_BANK) as json_file:
            facts_bank = json.load(json_file)
        for fact_id, fact_dict in facts_bank.items():
            self.all_fact_ids.append(fact_id)
            self.all_facts.append(postprocess_fact(fact_dict["fact"]))

        with open(EXPLANATIONS_BANK) as json_file:
            explanations_bank = json.load(json_file)
        for question_id, question_dict in explanations_bank.items():
            self.question_id_to_golden_facts_ids[question_id] = list(question_dict["explanation"].keys())

    def align_and_score(self, postprocessed_fact, question_id):
        rouge1f_scores = []

        index_of_highest_score = 0
        scorer = rouge_scorer.RougeScorer(["rouge1"])
        for i in range(len(self.all_facts)):
            postprocessed_actual_fact = self.all_facts[i]
            score = scorer.score(postprocessed_actual_fact, postprocessed_fact)["rouge1"].fmeasure
            rouge1f_scores.append(score)
            if score > rouge1f_scores[index_of_highest_score]:
                index_of_highest_score = i

        f_id = self.all_fact_ids[index_of_highest_score]

        try:
            relevance_score = self.question_id_to_facts_relevance_scores[question_id][f_id]["relevance_score"]
            is_golden = self.question_id_to_facts_relevance_scores[question_id][f_id]["is_golden"]
        except KeyError:
            relevance_score = 0
            is_golden = False

        return AlignedFact(f_id, self.all_facts[index_of_highest_score], relevance_score, is_golden,
                           rouge1f_scores[index_of_highest_score])

    def get_golden_facts(self, qid, relevance=3):
        relevant_golden_facts = []
        for f_id, f_dict in self.question_id_to_facts_relevance_scores[qid].items():
            if f_dict["is_golden"] and f_dict["relevance_score"] >= relevance:
                relevant_golden_facts.append(f_id)

        return relevant_golden_facts
