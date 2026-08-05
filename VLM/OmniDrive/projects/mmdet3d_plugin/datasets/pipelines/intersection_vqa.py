import json
import os
import random
from collections import defaultdict

from mmdet.datasets.builder import PIPELINES
from transformers import AutoTokenizer

from ..utils.constants import DEFAULT_IMAGE_TOKEN
from ..utils.data_utils import preprocess


@PIPELINES.register_module()
class LoadIntersectionQATest:
    def __init__(
        self,
        qa_json,
        tokenizer,
        max_length,
        system_prompt="You are monitoring a fixed urban intersection. Answer each question using only the current multi-view traffic scene.",
    ):
        self.qa_json = os.path.abspath(qa_json)
        self.system_prompt = system_prompt.strip()
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer,
            model_max_length=max_length,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.qa_groups = self._load_qa_groups()

    def _load_qa_groups(self):
        with open(self.qa_json, "r", encoding="utf-8") as file:
            payload = json.load(file)
        grouped = defaultdict(list)
        for qa_pair in payload["qa_pairs"]:
            grouped[qa_pair["frame_token"]].append(qa_pair)
        for frame_token in grouped:
            grouped[frame_token] = sorted(
                grouped[frame_token], key=lambda item: item["question_id"]
            )
        return dict(grouped)

    @staticmethod
    def _resolve_category(qa_pair):
        return (
            qa_pair.get("category")
            or qa_pair.get("subtemplate")
            or qa_pair.get("section")
            or qa_pair.get("chapter")
        )

    def __call__(self, results):
        frame_token = results["sample_idx"]
        qa_pairs = self.qa_groups.get(frame_token)
        if not qa_pairs:
            raise KeyError(f"No intersection QA pairs found for frame_token={frame_token}")

        sources = []
        questions = []
        question_ids = []
        qa_categories = []
        for qa_pair in qa_pairs:
            questions.append(qa_pair["question"])
            question_ids.append(qa_pair["question_id"])
            qa_categories.append(self._resolve_category(qa_pair))
            sources.append(
                [
                    {
                        "from": "human",
                        "value": (
                            DEFAULT_IMAGE_TOKEN
                            + "\n"
                            + self.system_prompt
                            + "\n"
                            + qa_pair["question"]
                        ),
                    },
                    {"from": "gpt", "value": ""},
                ]
            )

        vqa_converted = preprocess(sources, self.tokenizer, True, False)
        results["input_ids"] = vqa_converted["input_ids"]
        results["vlm_labels"] = questions
        results["question_ids"] = question_ids
        results["qa_categories"] = qa_categories
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class LoadIntersectionQATrain:
    def __init__(
        self,
        qa_json,
        tokenizer,
        max_length,
        sample_mode="random",
        system_prompt="You are monitoring a fixed urban intersection. Answer each question using only the current multi-view traffic scene.",
    ):
        self.qa_json = os.path.abspath(qa_json)
        self.sample_mode = sample_mode
        self.system_prompt = system_prompt.strip()
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer,
            model_max_length=max_length,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.qa_groups = self._load_qa_groups()

    @staticmethod
    def _is_trajectory_pair(qa_pair):
        return qa_pair.get("subtemplate") == "r_object_future_trajectory"

    @staticmethod
    def _resolve_category(qa_pair):
        return (
            qa_pair.get("category")
            or qa_pair.get("subtemplate")
            or qa_pair.get("section")
            or qa_pair.get("chapter")
        )

    def _load_qa_groups(self):
        with open(self.qa_json, "r", encoding="utf-8") as file:
            payload = json.load(file)
        grouped = defaultdict(list)
        for qa_pair in payload["qa_pairs"]:
            grouped[qa_pair["frame_token"]].append(qa_pair)
        for frame_token in grouped:
            grouped[frame_token] = sorted(
                grouped[frame_token], key=lambda item: item["question_id"]
            )
        return dict(grouped)

    def _select_pairs(self, qa_pairs):
        if self.sample_mode == "all":
            return qa_pairs
        if self.sample_mode == "first":
            return [qa_pairs[0]]
        trajectory_pairs = [qa_pair for qa_pair in qa_pairs if self._is_trajectory_pair(qa_pair)]
        if self.sample_mode == "trajectory_only":
            return [random.choice(trajectory_pairs or qa_pairs)]
        if self.sample_mode == "trajectory_first" and trajectory_pairs:
            return [random.choice(trajectory_pairs)]
        return [random.choice(qa_pairs)]

    def __call__(self, results):
        frame_token = results["sample_idx"]
        qa_pairs = self.qa_groups.get(frame_token)
        if not qa_pairs:
            raise KeyError(f"No intersection QA pairs found for frame_token={frame_token}")

        selected_pairs = self._select_pairs(qa_pairs)
        conversation = []
        questions = []
        question_ids = []
        qa_categories = []
        for index, qa_pair in enumerate(selected_pairs):
            human_prompt = qa_pair["question"]
            if index == 0:
                human_prompt = DEFAULT_IMAGE_TOKEN + "\n" + self.system_prompt + "\n" + human_prompt
            conversation.extend(
                [
                    {"from": "human", "value": human_prompt},
                    {"from": "gpt", "value": qa_pair["answer"]},
                ]
            )
            questions.append(qa_pair["question"])
            question_ids.append(qa_pair["question_id"])
            qa_categories.append(self._resolve_category(qa_pair))

        vqa_converted = preprocess([conversation], self.tokenizer, True, True)
        results["input_ids"] = vqa_converted["input_ids"][0]
        results["vlm_labels"] = vqa_converted["labels"][0]
        results["question_ids"] = question_ids
        results["qa_categories"] = qa_categories
        results["train_questions"] = questions
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class LoadIntersectionQASingleTrain:
    def __init__(
        self,
        tokenizer,
        max_length,
        system_prompt="You are monitoring a fixed urban intersection. Answer each question using only the current multi-view traffic scene.",
    ):
        self.system_prompt = system_prompt.strip()
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer,
            model_max_length=max_length,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token

    @staticmethod
    def _resolve_category(qa_pair):
        return (
            qa_pair.get("category")
            or qa_pair.get("subtemplate")
            or qa_pair.get("section")
            or qa_pair.get("chapter")
        )

    def __call__(self, results):
        qa_pair = results.get("qa_pair")
        if qa_pair is None:
            raise KeyError("LoadIntersectionQASingleTrain requires results['qa_pair']")

        question = qa_pair["question"]
        conversation = [
            {
                "from": "human",
                "value": DEFAULT_IMAGE_TOKEN + "\n" + self.system_prompt + "\n" + question,
            },
            {"from": "gpt", "value": qa_pair["answer"]},
        ]
        vqa_converted = preprocess([conversation], self.tokenizer, True, True)
        results["input_ids"] = vqa_converted["input_ids"][0]
        results["vlm_labels"] = vqa_converted["labels"][0]
        results["question_ids"] = [qa_pair["question_id"]]
        results["qa_categories"] = [self._resolve_category(qa_pair)]
        results["train_questions"] = [question]
        return results

    def __repr__(self):
        return self.__class__.__name__
