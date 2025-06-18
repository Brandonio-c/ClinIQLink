#!/usr/bin/env python3
"""
You must first clone the ncloze github from here:
https://github.com/ondovb/nCloze


optimize_extracted_qa_dataset.py

This script performs two main tasks on your QA datasets:

1. **Clean “short_inverse.json”**  
    Removes any span beginning with “according to” up to and including the next comma 
    in each `incorrect_explanation` field, writing the result to `short_inverse_cleaned.json`.

2. **Regenerate distractors in “MC.json”**  
    Uses a masked-language model and embedding pipeline to produce new, plausible,
    incorrect, and distinctive distractor options for your multiple‐choice QA items,
    reading from `MC.json` and writing the output to `MC_updated.json`.

— Arguments —

    --dir          Folder containing both `short_inverse.json` and `MC.json`.
    --output-dir   Folder where `short_inverse_cleaned.json` and `MC_updated.json` will be written.
    --top-k        (Optional; default=64)  Number of mask-fill candidates to retrieve.
    --select-n     (Optional; default=3)   How many distractors to select per question.
    --alpha        (Optional; default=0.3) Weight on “incorrectness” score when ranking.
    --beta         (Optional; default=0.3) Weight on “distinctiveness” score when ranking.

— Example Usage —

First, clone (ncloze gh) and navigate to the scripts directory:

git clone https://github.com/ondovb/nCloze
cd /path/to/db_interact_scripts

### Run both steps (clean **and** regenerate distractors)

python optimize_extracted_qa_dataset.py \
  --source-dir ../../database/QA_dataset/source_texts \
  --input-dir extracted_db_qa_pairs_human_expert_annotated_verified \
  --output-dir extracted_db_qa_pairs_human_expert_annotated_verified/updated \
  --unix-wordlist nCloze/dict-unix.txt \
  --info-wordlist nCloze/dict-info.txt \
  --profanity-file nCloze/profanity.json \
  --top-k 50 \
  --select-n 3 \
  --alpha 0.4 \
  --beta 0.2 \
  --method default

### Clean **only** the `short_inverse.json`

python optimize_extracted_qa_dataset.py \
  --source-dir ../../database/QA_dataset/source_texts \
  --input-dir extracted_db_qa_pairs_human_expert_annotated_verified \
  --output-dir extracted_db_qa_pairs_human_expert_annotated_verified/updated \
  --unix-wordlist nCloze/dict-unix.txt \
  --info-wordlist nCloze/dict-info.txt \
  --profanity-file nCloze/profanity.json \
  --clean


### Regenerate **only** the `MC.json` distractors

python optimize_extracted_qa_dataset.py \
  --source-dir ../../database/QA_dataset/source_texts \
  --input-dir extracted_db_qa_pairs_human_expert_annotated_verified \
  --output-dir extracted_db_qa_pairs_human_expert_annotated_verified/updated \
  --unix-wordlist nCloze/dict-unix.txt \
  --info-wordlist nCloze/dict-info.txt \
  --profanity-file nCloze/profanity.json \
  --distractors \
  --top-k 50 \
  --select-n 3 \
  --alpha 0.4 \
  --beta 0.2 \
  --method default



— e.g. - 
cd /Users/coleloughbc/Documents/VSCode-Local/ClinIQLink-QA-website/backend/db_interact_scripts

python optimize_extracted_qa_dataset.py \
  --source-dir ../database/QA_dataset/source_texts \
  --input-dir extracted_db_qa_pairs_human_expert_annotated_verified \
  --output-dir extracted_db_qa_pairs_human_expert_annotated_verified/updated \
  --unix-wordlist nCloze/dict-unix.txt \
  --info-wordlist nCloze/dict-info.txt \
  --profanity-file nCloze/profanity.json \
  --top-k 50 \
  --select-n 3 \
  --alpha 0.4 \
  --beta 0.2 \
  --method default

"""


import os
import random
import re
from string import punctuation
import sys
import math
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import logging
import argparse
from tqdm import tqdm
import io
from math import log,exp,sqrt
import json
# import stanza
import spacy
import numpy as np
from numpy import ndarray
import pandas as pd
import torch 
import torch as T
from typing import List, Dict, Tuple, Type, Union
from torch import Tensor, device
from torch.autograd import Variable
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from statistics import mean
import string
import platform
import difflib
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    AutoModel,
    pipeline as hf_pipeline
)

import nltk

# ─── ensure we cache under the script directory ────────────────────────────────
HERE = os.path.dirname(__file__)
NLTK_CACHE = os.path.join(HERE, "nltk_cache")
os.makedirs(NLTK_CACHE, exist_ok=True)
nltk.data.path.insert(0, NLTK_CACHE)

# ─── list all corpora you need ────────────────────────────────────────────────
for corpus in ("stopwords", "gutenberg", "punkt", "punkt_tab"):
    try:
        nltk.data.find(f"corpora/{corpus}")
    except LookupError:
        nltk.download(corpus, download_dir=NLTK_CACHE)

# now safe to import
from nltk.corpus      import stopwords, gutenberg
from nltk.probability import FreqDist
from nltk.stem.porter import PorterStemmer


# reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


stemmer      = PorterStemmer()
freq_dist    = FreqDist(w.lower() for w in gutenberg.words())
stop_words   = set(stopwords.words("english"))

# ─── imports for GDCP method ─────────────────────────────────────────────

from tqdm import tqdm
import os
from transformers import BertTokenizer, BertForMaskedLM, pipeline
import numpy as np
import fasttext
import nltk
from nltk.tokenize import word_tokenize
import json
from huggingface_hub import hf_hub_download

class ExternalRefCleaner:
    """
    1) Removes any span beginning with “according to” up through the next
        punctuation mark (comma, period, question‐mark, exclamation, semicolon or colon).
        e.g. “According to the paragraph, ” → “”
                “according to the study. ”  → “”
                “According to X? ” → “”
    """
    def __init__(self):
        # matches “according to …<punctuation>”
        self.pattern = re.compile(r'(?i)\baccording to\b[^,\.!\?;:]*[\,\.!\?;:]\s*')

    def clean(self, text: str) -> str:
        # Remove all "according to ...<punct>" spans
        cleaned = self.pattern.sub('', text or '')
        # collapse multiple spaces
        return re.sub(r'\s{2,}', ' ', cleaned).strip()

# original roberta-large" model used by brian in his ncloze model has a 512 token length. 
# should update to a newer model anyway, e.g. llama 3.3, llama 4
# for now, will use either:

# allenai/longformer-base-4096 or 
# google/bigbird-roberta-base

class DistractorPipeline:
    def __init__(
        self,
        source_dir: str,
        unix_wordlist_path: str,
        info_wordlist_path: str,
        profanity_path: str,
        input_path: str,
        output_path: str,
        mask_model_name: str = "allenai/longformer-base-4096",
        embed_model_name: str = "allenai/longformer-base-4096",
        top_k: int = 50,
        select_n: int = 3,
        alpha: float = 0.3,
        beta: float = 0.3,
        device: str = None,
        distractor_pool: int = 64,
        min_dist: int = 5,
        min_sent_words: int = 6,
        max_subwords: int = 100,
        debug_output: bool = False,
        extend_subwords: bool = True,
        use_cdgp=False, 
        distractors_from_text: bool = False,
    ):

        self.source_dir = source_dir
        self.input_path = input_path
        self.output_path = output_path
        self.top_k = top_k
        self.select_n = select_n
        self.alpha = alpha
        self.beta = beta
        if device:
            self.device = device
        else:
            system = platform.system().lower()
            if system == "darwin":  # macOS
                self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        self.ALPHA = alpha
        self.BETA = beta
        self.DISTRACTOR_POOL = distractor_pool
        self.MIN_DIST = min_dist
        self.MIN_SENT_WORDS = min_sent_words
        self.MAX_SUBWORDS = max_subwords
        self.DEBUG_OUTPUT = debug_output
        self.EXTEND_SUBWORDS = extend_subwords

        self.words_unix = set(open(unix_wordlist_path, "r", encoding="utf-8").read().split())
        self.words_info = set(open(info_wordlist_path, "r", encoding="utf-8").read().split())
        self.words_large = self.words_unix | self.words_info
        self.profanity = json.load(open(profanity_path, "r", encoding="utf-8"))

        self.mask_tokenizer = AutoTokenizer.from_pretrained(mask_model_name, use_fast=False)
        self.mask_tokenizer.model_max_length = 4096
        self.mask_token = self.mask_tokenizer.mask_token
        self.toker = self.mask_tokenizer
        self.mask_model = AutoModelForMaskedLM.from_pretrained(mask_model_name).to(self.device)
        self.mask_fill = hf_pipeline(
            "fill-mask",
            model=self.mask_model,
            tokenizer=self.mask_tokenizer,
            device=0 if self.device == "cuda" else -1,
            top_k=self.top_k
        )

        self.embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name, use_fast=False)
        self.embed_tokenizer.model_max_length = 4096

        self.embed_model = AutoModel.from_pretrained(embed_model_name).to(self.device)

        self._source_cache = {}
        self.stemmer = PorterStemmer()
        self.freq = FreqDist(i.lower() for i in gutenberg.words())
        vocab = self.mask_tokenizer.get_vocab()
        self.sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])


        self.suffix_mask = T.FloatTensor([
            1 if ('Ġ' != tok[0][0] and re.match("^[A-Za-z0-9']*$", tok[0])) else 0 for tok in self.sorted_vocab
        ])
        self.suffix_mask_inv = self.suffix_mask * -1 + 1
        self.word_mask = self.suffix_mask_inv * T.FloatTensor([
            1 if self.is_word(tok[0][1:]) and tok[0][1:].lower() not in self.profanity else 0 for tok in self.sorted_vocab
        ])

        if self.device == "cuda":
            self.suffix_mask = self.suffix_mask.cuda()
            self.suffix_mask_inv = self.suffix_mask_inv.cuda()
            self.word_mask = self.word_mask.cuda()
        
        self.suffix_mask = self.suffix_mask.to(self.device)
        self.suffix_mask_inv = self.suffix_mask_inv.to(self.device)
        self.word_mask = self.word_mask.to(self.device)

        self.cosine = T.nn.CosineSimilarity(dim=0)
        # logging.info(f"[INFO] Stanza device config: use_gpu={(self.device != 'cpu')}")
        # self.nlp = stanza.Pipeline(
        #     lang='en',
        #     processors='tokenize',
        #     use_gpu=(self.device != "cpu") and torch.cuda.is_available()  # avoid GPU requested but unavailable
        # )
        self.spacy_nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])
        self.spacy_nlp.add_pipe("sentencizer")

        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.stop_words = set(stopwords.words('english'))
        self.e=1e-10

        self.CSG_MODEL = "AndyChiang/cdgp-csg-bert-cloth"
        self.DS_MODEL = "./cdgp-ds-fasttext.bin"
        self.TOP_K = 3
        self.WEIGHT = {"s0": 0.6, "s1": 0.15, "s2": 0.15, "s3": 0.1}
        self.STOP_WORDS = ["[MASK]", "[SEP]", "[PAD]", "[CLS]"]
        logging.info(f"Using masked LM: {mask_model_name} on device: {self.device}")
        logging.info(f"Using embed model: {embed_model_name} on device: {self.device}")


        # Load models
        self.use_cdgp = use_cdgp
        if self.use_cdgp:
            logging.info(f"Load CSG model at {self.CSG_MODEL}...")
            self.tokenizer = BertTokenizer.from_pretrained(self.CSG_MODEL)
            self.csg_model = BertForMaskedLM.from_pretrained(self.CSG_MODEL)
            self.unmasker = pipeline('fill-mask', tokenizer=self.tokenizer, model=self.csg_model, top_k=self.TOP_K)

            logging.info(f"Load DS model at {self.DS_MODEL}...")
            self.ds_model = fasttext.load_model(self.DS_MODEL)
        
        self.nltk_sent_toker = nltk.data.load('tokenizers/punkt/english.pickle')
        self.DISTRACTORS_FROM_TEXT = distractors_from_text
        self.cos = T.nn.CosineSimilarity(dim=0)
        self.idx2ans = {0: "A", 1: "B", 2: "C", 3: "D"}


    def _clean_mcq_text_fields(self):

        all_texts = set()
        for entry in self.data:
            all_texts.update(entry.get("options", []))
            all_texts.add(entry.get("correct_answer", ""))

        suffix_map = {}
        for text in all_texts:
            for candidate in all_texts:
                if candidate != text:
                    if candidate.endswith(text) and 1 <= len(candidate) - len(text) <= 2:
                        suffix_map[text] = candidate

        for text in all_texts:
            if text not in suffix_map:
                matches = difflib.get_close_matches(text, all_texts, n=1, cutoff=0.9)
                if matches:
                    suffix_map[text] = matches[0]

        for entry in self.data:
            # Fix options
            fixed_opts = []
            seen_norms = set()
            for opt in entry.get("options", []):
                fixed = suffix_map.get(opt, opt)
                if fixed and fixed[0].islower():
                    fixed = fixed.capitalize()
                norm = fixed.lower().strip()
                if norm not in seen_norms:
                    seen_norms.add(norm)
                    fixed_opts.append(fixed)
            entry["options"] = fixed_opts

            # Fix correct answer
            ca = entry.get("correct_answer", "")
            fixed_ca = suffix_map.get(ca, ca)
            if fixed_ca and fixed_ca[0].islower():
                fixed_ca = fixed_ca.capitalize()
            entry["correct_answer"] = fixed_ca


    def is_word(self, token: str) -> bool:
        split = token.lower().split("'")
        if len(split) > 2:
            return False
        elif len(split) == 2:
            return self.is_word(split[0]) and (split[1] in ['t', 'nt', 's', 'll'])
        elif '-' in token:
            return all(self.is_word(part) for part in token.split('-'))
        else:
            return token.lower() in self.words_large or token.isalpha()

    def get_emb(self, sentence_tokens, target_tokens, layers=[4]):
        mask_idx = sentence_tokens.index(self.mask_tokenizer.mask_token_id)
        if len(target_tokens) == 0:
            raise ValueError("Target tokens for embedding are empty. Likely due to bad answer tokenization.")

        sentence_tokens = sentence_tokens.copy()

        while mask_idx + len(target_tokens) - 1 >= 4096:
            sentence_tokens = sentence_tokens[100:]
            mask_idx -= 100

        sentence_tokens[mask_idx:mask_idx+1] = target_tokens
        sentence_tokens = sentence_tokens[:4096]

        with T.no_grad():
            output = self.mask_model(
                T.tensor([sentence_tokens]).to(self.device),
                T.tensor([[1] * len(sentence_tokens)]).to(self.device),
                output_hidden_states=True
            )

        output_stack = T.stack([output.hidden_states[i] for i in layers]).sum(0).squeeze()
        return output_stack[mask_idx:mask_idx+len(target_tokens)].mean(dim=0)

    def get_emb_batch(self, token_batches: List[List[int]]) -> List[Tensor]:
        """
        Get mean-pooled embeddings for a batch of tokenized inputs.
        Each input is a list of token IDs (already masked appropriately).
        """
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in token_batches],
            batch_first=True,
            padding_value=self.mask_tokenizer.pad_token_id
        ).to(self.device)

        attention_mask = (input_ids != self.mask_tokenizer.pad_token_id).long().to(self.device)

        with torch.no_grad():
            outputs = self.mask_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            selected_layers = [hidden_states[i] for i in [4]]  # You can average more layers if needed
            summed = torch.stack(selected_layers).sum(0)  # [B, T, H]

        # Compute mean of each sequence's tokens (ignoring pad)
        embs = []
        for i, attn in enumerate(attention_mask):
            seq_len = attn.sum()
            emb = summed[i, :seq_len].mean(dim=0)
            embs.append(emb)
        return embs

    
    def extend(self, toks_sent, toks_para, suff_ids, n_masks, ctx_words):
        """
        Recursively generate the most likely suffix extension for a subword in the mask position.

        Args:
            toks_sent (List[int]): Tokenized sentence with [MASK] token(s).
            toks_para (List[int]): Tokenized paragraph with [MASK] token(s).
            suff_ids (List[int]): List of suffix token IDs already selected.
            n_masks (int): Number of remaining masks to fill.
            ctx_words (List[str]): Set of valid contextual completions.

        Returns:
            Tuple[List[int], float]: Best subword IDs and their probability score.
        """
        try:
            if n_masks < 1:
                logging.warning("[extend] Called with n_masks < 1. Returning empty.")
                return [], 0.0

            sm_sent = self.get_softmax_logits(toks_sent, n_masks, suff_ids)
            sm_para = self.get_softmax_logits(toks_para, n_masks, suff_ids)

            if not isinstance(sm_sent, torch.Tensor) or not isinstance(sm_para, torch.Tensor):
                logging.warning("[extend] One or both softmax outputs are not tensors.")
                return [], 0.0

            if sm_sent.size(0) < n_masks or sm_para.size(0) < n_masks:
                logging.warning(f"[extend] Softmax tensors have insufficient rows: "
                                f"sm_sent={sm_sent.size()}, sm_para={sm_para.size()}, n_masks={n_masks}")
                return [], 0.0

            sm_combined = torch.exp((sm_sent[-1].log() + sm_para[-1].log()) / 2)

            topk_pfx = torch.topk(sm_combined * self.suffix_mask_inv, self.EXTEND_BEAM_WIDTH)
            best_ids = []
            best_prob = 0.0

            for i, tok_id in enumerate(topk_pfx.indices.tolist()):
                decoded = self.toker.decode([tok_id] + suff_ids).strip()
                if self.is_word(decoded) or decoded.lower() in ctx_words:
                    best_ids = [tok_id]
                    best_prob = float(topk_pfx.values[i])
                    logging.debug(f"[extend] Selected prefix: '{decoded}' with prob={best_prob:.4f}")
                    break

            if n_masks > 1:
                topk_sfx = torch.topk(sm_combined * self.suffix_mask, self.EXTEND_BEAM_WIDTH)
                for i, tok_id in enumerate(topk_sfx.indices.tolist()):
                    rec_suff_ids = [tok_id] + suff_ids
                    sub_ids, prob = self.extend(toks_sent, toks_para, rec_suff_ids, n_masks - 1, ctx_words)
                    if prob > best_prob:
                        best_ids = sub_ids + [tok_id]
                        best_prob = prob
                        logging.debug(f"[extend] Found better suffix path with prob={prob:.4f}")

            return best_ids, best_prob

        except Exception as ex:
            logging.exception(f"[extend] Exception occurred during recursive extension: {ex}")
            return [], 0.0


    def energy(self, ctx, scaled_dists, scaled_sims, choices, words, ans):
        """
        Cost function to help choose the best distractors by combining contextual relevance
        and embedding diversity.

        ctx: Tensor of contextual scores.
        scaled_dists: Dict of cosine distances between embeddings, normalized.
        scaled_sims: Tensor of cosine similarity between each distractor and answer embedding.
        choices: List of chosen distractor indices.
        words: List of distractor word strings.
        ans: Correct answer string.
        
        Returns:
            - e_emb: Diversity score (embedding-based)
            - e_ctx: Cumulative contextual score
            - e_sim: Similarity penalty (inverse harmonic mean of similarities)
        """
        hm_sim = 0.0
        e_ctx = 0.0
        for i in choices:
            hm_sim += 1.0 / scaled_sims[i]
            e_ctx += ctx[i]

        e_sim = float(len(choices)) / hm_sim

        hm_emb = 0.0
        count = 0
        c = choices + [len(ctx)]
        for i in range(len(c)):
            for j in range(i):
                key = f"{max(c[i], c[j])}-{min(c[i], c[j])}"
                d = scaled_dists[key]
                hm_emb += 1.0 / d
                count += 1

        e_emb = float(count) / hm_emb if hm_emb > 0 else 0.0
        return e_emb, e_ctx, e_sim

    def anneal(self, probs_sent_context, probs_para_context, embs, emb_ans, words, k, ans):
        """
        Simulated annealing to select k distractors that are:
        - plausible in context
        - semantically distinct from each other and the correct answer

        Arguments:
        - probs_sent_context: list of sentence-level token plausibility scores
        - probs_para_context: list of paragraph-level token plausibility scores
        - embs: list of distractor embeddings
        - emb_ans: embedding of the correct answer
        - words: list of distractor word strings
        - k: number of distractors to select
        - ans: correct answer string

        Returns:
        - choices: indices of selected distractors
        """
        its = 1000
        m = len(probs_sent_context)
        choices = list(range(k))

        # ─── Calculate cosine distances between all pairs (including answer) ───────
        embsa = embs + [emb_ans]
        dists = {}
        for i in range(len(embsa)):
            for j in range(i):
                key = f"{i}-{j}"
                dists[key] = 1 - self.cos(embsa[i], embsa[j])

        dist_tensor = torch.tensor(list(dists.values()))
        dist_min = torch.min(dist_tensor)
        dist_max = torch.max(dist_tensor)
        for key in dists:
            dists[key] = (dists[key] - dist_min) / (dist_max - dist_min + 1e-10)

        # ─── Cosine similarity between each distractor and the correct answer ─────
        sims = torch.tensor([self.cos(emb_ans, emb) for emb in embs])
        scaled_sims = (sims - torch.min(sims)) / (torch.max(sims) - torch.min(sims) + 1e-10)

        # ─── Contextual scores: log probs from sent - alpha * para ────────────────
        ctx = torch.tensor(probs_sent_context).log() - self.ALPHA * torch.tensor(probs_para_context).log()
        ctx = (ctx - torch.min(ctx)) / (torch.max(ctx) - torch.min(ctx) + 1e-10)

        # ─── Initial energy computation ───────────────────────────────────────────
        e_emb, e_ctx, e_sim = self.energy(ctx, dists, scaled_sims, choices, words, ans)
        e = e_ctx + self.BETA * e_emb

        for i in range(its):
            temp = 1.0 - (i / its)
            mut_idx = random.randrange(k)
            orig = choices[mut_idx]
            new = orig
            while new in choices:
                new = random.randrange(m)
            choices[mut_idx] = new

            e_emb_new, e_ctx_new, _ = self.energy(ctx, dists, scaled_sims, choices, words, ans)
            e_new = e_ctx_new + self.BETA * e_emb_new
            delta = e_new - e
            exponent = delta / temp

            if exponent < -50:
                exponent = -50  # prevent numerical underflow

            if delta > 0 or math.exp(exponent) > random.random():
                e = e_new  # accept
            else:
                choices[mut_idx] = orig  # revert

        if self.DEBUG_OUTPUT:
            logging.info([words[j] for j in choices] + [ans], f"e: {e:.4f}")

        return choices

    def get_softmax_logits(self, toks, n_masks=1, sub_ids=None):
        """
        Get softmax logits from a masked language model.

        Args:
            toks (list[int]): Token IDs for the input sentence.
            n_masks (int): Number of [MASK] tokens to insert.
            sub_ids (list[int]): List of subword IDs to place after the masks.

        Returns:
            torch.Tensor or []: Softmaxed logits over the vocabulary for each masked token,
                                or [] if something fails.
        """
        if sub_ids is None:
            sub_ids = []

        try:
            if self.toker.mask_token_id not in toks:
                logging.warning("[get_softmax_logits] Mask token not found in token list.")
                return []

            msk_idx = toks.index(self.toker.mask_token_id)
            toks = toks.copy()
            toks[msk_idx:msk_idx + 1] = [self.toker.mask_token_id] * n_masks + sub_ids

            # Truncate if mask index is too far into sequence
            while msk_idx >= 4096:
                toks = toks[100:]
                msk_idx -= 100

            toks = toks[:4096]

            if len(toks) == 0:
                logging.warning("[get_softmax_logits] Token list is empty after truncation.")
                return []

            toks_tensor = torch.tensor([toks]).to(self.device)
            attn_mask = torch.tensor([[1] * len(toks)]).to(self.device)

            with torch.no_grad():
                output = self.mask_model(toks_tensor, attention_mask=attn_mask)
                logits = output.logits

                if logits.shape[1] < msk_idx + n_masks:
                    logging.warning(f"[get_softmax_logits] Output logits shape too short for masked span "
                                    f"(logits.shape[1]={logits.shape[1]}, required={msk_idx + n_masks}).")
                    return []

                if sm is None or sm.shape[0] == 0:
                    logging.warning("[get_softmax_logits] Softmax returned None or empty.")
                    return []

                sm = torch.nn.functional.softmax(logits[0, msk_idx:msk_idx + n_masks, :], dim=1)

                return sm

        except Exception as ex:
            logging.exception(f"[get_softmax_logits] Exception occurred: {ex}")
            return []

    
    def get_softmax_logits_batch(self, token_batches: List[List[int]], mask_positions: List[int]) -> Tensor:
        """
        Batched version of get_softmax_logits.
        Args:
            token_batches: List of token ID sequences
            mask_positions: Position of the [MASK] token in each sequence
        Returns:
            Tensor of shape (B, V) — softmax logits for each masked position
        """
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in token_batches],
            batch_first=True,
            padding_value=self.mask_tokenizer.pad_token_id
        ).to(self.device)

        attention_mask = (input_ids != self.mask_tokenizer.pad_token_id).long().to(self.device)

        with torch.no_grad():
            outputs = self.mask_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, T, V]

        # Extract logits at each masked position
        mask_logits = torch.stack([
            logits[i, mask_positions[i]] for i in range(len(mask_positions))
        ])

        return torch.nn.functional.softmax(mask_logits, dim=-1)  # [B, V]

    
    def candidates(self, text, answer, qid):
        """Create list of unique distractors that does not include the actual answer"""
        if self.DEBUG_OUTPUT:
            logging.info(text)

        # doc = self.nlp(text)
        # sents = self.nltk_sent_toker.tokenize(text)
        sents = [s.text for s in self.spacy_nlp(text).sents]

        mask_sentences = [i for i in range(len(sents)) if self.mask_token in sents[i]]
        if not mask_sentences:
            raise ValueError("No sentence contains the mask token in candidates function.")
        msk_snt_idx = mask_sentences[0]

        just_masked_sentence = sents[msk_snt_idx]

        prv_snts = sents[:msk_snt_idx]
        nxt_snts = sents[msk_snt_idx + 1:]

        while len(just_masked_sentence.split()) < self.MIN_SENT_WORDS and (prv_snts or nxt_snts):
            if torch.rand(1) < 0.5 and prv_snts:
                just_masked_sentence = ' '.join([prv_snts.pop(), just_masked_sentence])
            elif nxt_snts:
                just_masked_sentence = ' '.join([just_masked_sentence, nxt_snts.pop(0)])

        ctx = just_masked_sentence
        while len(ctx.split()) < 3 * len(just_masked_sentence.split()) and (prv_snts or nxt_snts):
            if prv_snts:
                ctx = ' '.join([prv_snts.pop(), ctx])
            if nxt_snts:
                ctx = ' '.join([ctx, nxt_snts.pop(0)])

        tiled = just_masked_sentence
        while len(tiled) < len(text):
            tiled += ' ' + just_masked_sentence
        just_masked_sentence = tiled

        if self.DEBUG_OUTPUT:
            logging.info(ctx)
            logging.info(just_masked_sentence)

        toks_para = self.toker.encode(text)
        toks_sent = self.toker.encode(just_masked_sentence)

        sent_sms_all = []
        para_sms_all = []
        para_sms_right = []

        for i in range(self.MAX_SUBWORDS):
            para_sms = self.get_softmax_logits(toks_para, i + 1)
            para_sms_all.append(para_sms)
            sent_sms = self.get_softmax_logits(toks_sent, i + 1)
            sent_sms_all.append(sent_sms)
            suffix_mask = self.suffix_mask if i != 0 else self.suffix_mask_inv
            para_sms_right.append(torch.exp((sent_sms[i].log() + para_sms[i].log()) / 2) * suffix_mask)

        
        if len(para_sms_right) == 0:
            raise ValueError("para_sms_right is empty — likely no logits could be computed for softmax candidates.")

        if any(x.shape[0] == 0 for x in para_sms_right):
            raise ValueError("One or more entries in para_sms_right has size 0.")
        
        try:
            para_sm_best, para_pos_best = torch.max(torch.vstack(para_sms_right), 0)
        except RuntimeError as e:
            raise ValueError(f"Failed to stack and max para_sms_right. Shapes: {[x.shape for x in para_sms_right]}. Error: {e}")


        distractors, stems, embs = [], [], []
        sent_probs, para_probs = [], []

        ans_stem = self.stemmer.stem(answer.lower())
        emb_ans = self.get_emb(toks_para, self.toker(answer)['input_ids'][1:-1])
        para_words = text.lower().split()
        blank_word_idx = next(i for i, w in enumerate(para_words) if self.mask_token in w)
        prev_word = para_words[blank_word_idx - 1] if blank_word_idx > 0 else 'beforeanytext'
        next_word = para_words[blank_word_idx + 1] if blank_word_idx + 1 < len(para_words) else 'afteralltext'

        if len(para_sms_all[0]) > 0:
            top_ctx = torch.topk((sent_sms_all[0][0] * self.word_mask + self.e).log() - self.ALPHA * (para_sms_all[0][0] * self.word_mask + self.e).log(), len(para_sms_all[0][0]), dim=0)
            para_top_ids = top_ctx.indices.tolist()

            for id in para_top_ids:
                sub_ids = [int(id)]
                dec = self.toker.decode(sub_ids).strip()

                if self.DEBUG_OUTPUT:
                    logging.info('Trying:', dec)

                if dec.isupper() != answer.isupper():
                    continue

                if self.EXTEND_SUBWORDS:
                    ext_ids, _ = self.extend(
                        toks_sent, toks_para, [id], n_masks=3, ctx_words=para_words
                    )
                    sub_ids = ext_ids + sub_ids
                    dec_ext = self.toker.decode(sub_ids).strip()
                    if len(dec_ext.split()) > 1 or dec_ext in para_words:
                        dec = dec_ext
                    else:
                        sub_ids = [int(id)]

                # Try embedding-based exception path early
                allow_if_embedding_unique = False
                try:
                    temp_emb = self.get_emb(toks_para, sub_ids)
                    cos_sim = float(self.cos(emb_ans, temp_emb))
                    allow_if_embedding_unique = cos_sim < 0.8
                except Exception as emb_ex:
                    logging.warning(f"[QID={qid}] Failed to compute embedding similarity for '{dec}': {emb_ex}")

                # Now perform strict filtering, unless the embedding is sufficiently different
                reject_reason = None
                if len(dec) < 2:
                    reject_reason = "too short"
                elif dec[0].isupper() != answer[0].isupper():
                    reject_reason = "case mismatch"
                elif dec.lower() in self.profanity:
                    reject_reason = "profanity"
                elif not self.is_word(dec) and dec.lower() not in para_words:
                    reject_reason = "not a valid word"
                elif dec.lower() in [prev_word, next_word]:
                    reject_reason = "neighbor word"
                elif any(c.isdigit() for c in self.toker.decode([id])):
                    reject_reason = "contains digit"
                elif self.DISTRACTORS_FROM_TEXT and dec.lower() not in para_words:
                    reject_reason = "not in source text"
                else:
                    stem = self.stemmer.stem(dec).lower()
                    if stem in stems or stem == ans_stem:
                        reject_reason = "duplicate stem"

                if reject_reason:
                    if allow_if_embedding_unique:
                        logging.info(f"[QID={qid}] Accepted '{dec}' despite rejection reason '{reject_reason}' due to low embedding sim ({cos_sim:.2f})")
                    else:
                        if self.DEBUG_OUTPUT:
                            logging.debug(f"[QID={qid}] Rejected '{dec}' — {reject_reason}")
                        continue


                distractors.append(dec)
                stems.append(stem)
                sent_logprob, para_logprob = 0, 0
                nsubs = len(sub_ids)
                for j in range(nsubs):
                    sub_id = sub_ids[j]
                    sent_logprob += torch.log(sent_sms_all[nsubs - 1][j][sub_id])
                    para_logprob += torch.log(para_sms_all[nsubs - 1][j][sub_id])
                sent_logprob /= nsubs
                para_logprob /= nsubs
                if self.DEBUG_OUTPUT:
                    logging.info(f"{dec} (p_sent={sent_logprob:.4f}, p_para={para_logprob:.4f})")
                sent_probs.append(math.exp(sent_logprob))
                para_probs.append(math.exp(para_logprob))
                embs.append(self.get_emb(toks_para, sub_ids))

                if len(distractors) >= self.DISTRACTOR_POOL:
                    break

        if self.DEBUG_OUTPUT:
            logging.info(f"Corresponding Text: {text}")
            logging.info(f"Correct Answer: {answer}")
            logging.info(f"Distractors before annealing: {distractors}")

        if self.DEBUG_OUTPUT:
            logging.info(f"[QID={qid}] Returning {len(distractors)} distractors: {distractors}")

        if len(distractors) < self.select_n:
            logging.warning(f"[QID={qid}] Too few distractors after extension — falling back to word list sampling")
            fallback_candidates = random.sample(list(self.words_unix - {answer.lower()}), self.select_n * 2)
            for cand in fallback_candidates:
                if cand not in distractors:
                    distractors.append(cand.capitalize())
                    sent_probs.append(0.5)
                    para_probs.append(1.5)
                    try:
                        embs.append(self.get_emb(toks_para, self.toker.encode(cand)[1:-1]))
                    except Exception as emb_ex:
                        logging.warning(f"[QID={qid}] Failed to embed fallback candidate '{cand}': {emb_ex}")
                        embs.append(torch.zeros_like(emb_ans))  # fallback to zero vector


        return sent_probs, para_probs, embs, emb_ans, distractors
    

    def _causal_generate(self, prompt: str) -> str:
        """
        Very simple causal fallback generator using HuggingFace pipeline (assumes self.mask_model is causal-capable).
        """
        try:
            generator = hf_pipeline(
                "text-generation",
                model=self.mask_model,
                tokenizer=self.mask_tokenizer,
                device=0 if self.device == "cuda" else -1,
                max_new_tokens=40
            )
            output = generator(prompt, num_return_sequences=1, do_sample=True, temperature=0.8)[0]["generated_text"]
            return output.replace(prompt, "").strip()
        except Exception as e:
            logging.exception("Causal generation failed")
            return ""

    
    def create_distractors(self, qid,
                   masked_sent: str,
                   answer: str,
                   existing_opts: list[str] = None
                   ) -> list[str]:
        """
        Generate distractors using masked language modeling and contextual relevance.
        """
        try:
            logging.debug(f"Starting distractor creation for answer: {answer}")
            logging.debug(f"Masked sentence: {masked_sent}")

            # Step A: build your pool of fresh candidates
            sent_probs, para_probs, embs, emb_ans, distractors = self.candidates(masked_sent, answer, qid)

            # Step B: fold in any old wrong options
            if existing_opts:
                logging.debug(f"Existing options provided: {existing_opts}")
                for old in existing_opts:
                    try:
                        if old != answer and old not in distractors:
                            ids = self.mask_tokenizer(old)["input_ids"][1:-1]
                            if not ids:
                                logging.warning(f"Tokenization failed for old option: '{old}'")
                                continue
                            paragraph_tokens = self.mask_tokenizer.encode(masked_sent)
                            embs.append(self.get_emb(paragraph_tokens, ids))
                            sent_probs.append(1.0)
                            para_probs.append(1.0)
                            distractors.append(old)
                            logging.debug(f"Appended existing distractor: {old}")
                    except Exception as ex:
                        logging.exception(f"Error incorporating existing option '{old}': {ex}")

            # Fallback to causal generation if no distractors found
            if not distractors:
                logging.warning(f"[QID={qid}] No distractors found using MLM. Falling back to causal generation.")
                from transformers import AutoModelForCausalLM, AutoTokenizer

                # Simple prompt for causal model
                prompt = f"Q: {masked_sent}\nA: {answer}\nDistractors:\n"

                generation_model = AutoModelForCausalLM.from_pretrained(
                    "meta-llama/Meta-Llama-3-8B", torch_dtype=torch.float16
                ).to(self.device)
                generation_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

                input_ids = generation_tokenizer(prompt, return_tensors="pt").to(self.device)
                outputs = generation_model.generate(
                    **input_ids,
                    max_new_tokens=64,
                    num_return_sequences=1,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.9,
                    eos_token_id=generation_tokenizer.eos_token_id
                )

                completions = generation_tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_part = completions.split("Distractors:")[-1]
                lines = [line.strip("-•\n ") for line in generated_part.strip().split("\n") if line.strip()]
                lines = [line for line in lines if line and answer.lower() not in line.lower()]
                distractors = lines[:self.select_n] if lines else ["Option A", "Option B", "Option C"][:self.select_n]
                logging.info(f"[QID={qid}] Fallback generated distractors: {distractors}")
                return distractors

            # Step C: run annealing to select top distractors
            indices = self.anneal(
                sent_probs,
                para_probs,
                embs,
                emb_ans,
                distractors,
                self.select_n,
                answer
            )

            final_distractors = [distractors[i] for i in indices]
            logging.info(f"Selected distractors: {final_distractors}")

            # ─── Optional: Fallback to causal generation if distractors are missing or too few ─────
            if not final_distractors or len(final_distractors) < self.select_n:
                logging.warning(f"[QID={qid}] Falling back to causal prompting due to insufficient distractors.")
                prompt = f"Question: {masked_sent}\nAnswer: {answer}\nGive {self.select_n} plausible but incorrect answers, comma separated:"
                try:
                    generated = self._causal_generate(prompt)
                    if generated:
                        final_distractors = [d.strip() for d in generated.split(",") if d.strip() and d.strip() != answer]
                        final_distractors = final_distractors[:self.select_n]
                        logging.info(f"[QID={qid}] Causal fallback distractors: {final_distractors}")
                except Exception as e:
                    logging.exception(f"[QID={qid}] Causal fallback generation failed: {e}")

            return final_distractors

        except Exception as e:
            logging.exception(f"Failed to create distractors for answer='{answer}' in sentence: {masked_sent}")
            return []





    def score_positions(self, text: str) -> torch.Tensor:
        """
        Score masked positions in the text to determine quality distractor locations.
        Returns the sum of top distractor position logits.
        """
        # sents = self.nltk_sent_toker.tokenize(text)
        sents = [s.text for s in self.spacy_nlp(text).sents]

        mask_sentences = [i for i in range(len(sents)) if self.mask_token in sents[i]]
        if not mask_sentences:
            raise ValueError("No sentence contains the mask token in score_position function.")
        msk_snt_idx = mask_sentences[0]
        just_masked_sentence = sents[msk_snt_idx]

        prv_snts = sents[:msk_snt_idx]
        nxt_snts = sents[msk_snt_idx + 1:]

        i = 0
        while len(just_masked_sentence.split()) < self.MIN_SENT_WORDS and (len(prv_snts) or len(nxt_snts)):
            if i % 2 == 0 and prv_snts:
                just_masked_sentence = ' '.join([prv_snts.pop(), just_masked_sentence])
            elif nxt_snts:
                just_masked_sentence = ' '.join([just_masked_sentence, nxt_snts.pop(0)])
            i += 1

        if self.DEBUG_OUTPUT:
            logging.info(just_masked_sentence)

        toks_para = self.toker.encode(text)
        toks_sent = self.toker.encode(just_masked_sentence)

        para_sms = self.get_softmax_logits(toks_para, 1)[0]
        sent_sms = self.get_softmax_logits(toks_sent, 1)[0]

        ctx = (sent_sms * self.word_mask + self.epsilon).log() - self.ALPHA * (para_sms * self.word_mask + self.epsilon).log()
        tk = T.topk(ctx, self.DISTRACTOR_POOL, dim=0)

        return T.sum(tk.values)


    def mask(self, word, cdgp = False):
        strp = word.strip(punctuation)
        return word.replace(strp, '[MASK]' if cdgp else self.mask_token)

    def insert_answer(self, distractors, answer):
        idx = random.randint(0,3)
        distractors.insert(idx, answer)
        return distractors, self.idx2ans[idx]
    
    def choose_blanks(self, text: str, count: int) -> list:
        words = text.split()
        scores = []
        for i, word in enumerate(words):
            masked = self.mask(word)
            t = ' '.join(words[:i] + [masked] + words[i+1:])
            sents = self.nltk_sent_toker.tokenize(t)
            scores.append(float(self.score_positions(t)))

        scores = (T.tensor(scores).to(self.device) / 1000).sigmoid()
        blanks = []
        blank_scores = []
        answers = set()

        while len(blanks) < count:
            tk = T.topk(scores, 1)
            idx = tk.indices[0]
            strp = words[idx].strip(string.punctuation)

            if tk.values[0] > 0 and (
                len(strp) == 0 or
                strp == 'a(n' or
                strp[0].isupper() or
                any(char.isdigit() for char in strp) or
                any(char in string.punctuation.replace("'", '').replace('-', '') for char in strp) or
                strp in answers or
                strp.lower() in self.stop_words
            ):
                scores[idx] = 0
            else:
                blanks.append(idx)
                blank_scores.append(scores[idx])
                answers.add(strp)
                for i in range(max(idx - self.MIN_DIST + 1, 0), min(idx + self.MIN_DIST, len(scores))):
                    scores[i] = 0

        logging.info(len(blanks), list(zip(blanks, blank_scores)))
        return sorted(blanks)
    
    def create_distractors_cdgp(self, qid, text, answer):
        try:
            ds = self.CDGP(text, answer)
            logging.info(f"[QID={qid}] CDGP returned distractors: {ds}")
            return ds
        except Exception as e:
            logging.exception(f"[QID={qid}] CDGP distractor generation failed.")
            return ["Option A", "Option B", "Option C"][:self.select_n]


    def create_distractors_for_blanks(self, qid, text: str, blanks: list, cdgp: bool = False):
        words = text.split()
        dists = []
        answers = []

        for blank in blanks:
            masked = self.mask(words[blank], cdgp)
            t = ' '.join(words[:blank] + [masked] + words[blank+1:])
            strp = words[blank].strip(string.punctuation)

            if cdgp:
                ds = self.create_distractors_cdgp(qid, t, strp)
            else:
                ds = self.create_distractors(qid, t, strp)

            ds, letter = self.insert_answer(ds, strp)
            dists.append(ds)
            answers.append(letter)

        for blank in sorted(blanks):
            masked = self.mask(words[blank])
            words[blank] = masked.replace('[MASK]' if cdgp else self.mask_token, '_')

        return ' '.join(words), dists, answers

    def choose_and_blank(self, qid, text: str, count: int):
        blanks = self.choose_blanks(text, count)
        return self.create_distractors_for_blanks(qid, text, blanks)

    def plot_emb(self, probs, embs, emb_ans, words, ans, ids):
        """
        Visualizes the distractor embeddings and their relative distances using PCA.
        """
        words = [ans] + words
        probs = [np.max(probs)] + probs
        embs = T.stack([emb_ans] + embs).cpu().numpy()

        # Dimensionality reduction
        pca = PCA(n_components=2)
        new_values = pca.fit(embs).transform(embs)

        ofst = np.min(probs)
        rnge = np.max(probs) - ofst

        x, y = [], []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])

        plt.figure(figsize=(24, 16))

        for i in range(len(x)):
            idx = i - 1
            scale = 0.9 * (probs[idx] - ofst) / rnge if i > 0 else 1
            plt.text(
                x[i],
                y[i],
                words[i],
                ha='center',
                va='center',
                size='large',
                c='magenta' if i == 0 else 'black',
                alpha=scale,
                bbox=dict(edgecolor='magenta' if i == 0 else 'black', facecolor='white', alpha=0.5) if i == 0 else None
            )
            plt.scatter(x[i], y[i], c='white', s=500)

        # Draw lines between all selected distractors and the answer
        ids = [0] + [x + 1 for x in ids]
        n = len(ids)
        data = []
        for i in range(n):
            for j in range(i + 1, n):
                data.append((x[ids[i]], x[ids[j]]))
                data.append((y[ids[i]], y[ids[j]]))
        plt.plot(*data, alpha=0.25, color='black')

        plt.savefig("myImagePDF.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    def score(self, text, answer, dists):
        """
        Scores a set of distractors based on:
        - Plausibility from sentence context (`phi`)
        - Implausibility in paragraph context (`Phi`)
        - Embedding diversity with respect to the correct answer (`delta`)
        """
        embs = []
        phis = []
        Phis = []

        # doc = self.nlp(text)
        # sents = self.nltk_sent_toker.tokenize(text)
        sents = [s.text for s in self.spacy_nlp(text).sents]

        mask_sentences = [i for i in range(len(sents)) if self.mask_token in sents[i]]
        if not mask_sentences:
            raise ValueError("No sentence contains the mask token in score function.")
        msk_snt_idx = mask_sentences[0]
        just_masked_sentence = sents[msk_snt_idx]

        prv_snts = sents[:msk_snt_idx]
        nxt_snts = sents[msk_snt_idx + 1:]

        if len(just_masked_sentence.split(' ')) < self.MIN_SENT_WORDS and len(prv_snts):
            just_masked_sentence = ' '.join([prv_snts.pop(), just_masked_sentence])

        while len(just_masked_sentence.split(' ')) < self.MIN_SENT_WORDS and (len(prv_snts) or len(nxt_snts)):
            if T.rand(1) < 0.5 and len(prv_snts):
                just_masked_sentence = ' '.join([prv_snts.pop(), just_masked_sentence])
            elif len(nxt_snts):
                just_masked_sentence = ' '.join([just_masked_sentence, nxt_snts.pop(0)])

        toks_para = self.toker.encode(text)
        toks_sent = self.toker.encode(just_masked_sentence)

        sm_para = self.get_softmax_logits(toks_para, n_masks=1, sub_ids=[])
        sm_sent = self.get_softmax_logits(toks_sent, n_masks=1, sub_ids=[])

        for dist in dists:
            tok_dist = self.toker.encode(dist)[1:-1]
            phis.append(float(sm_sent[0][tok_dist].mean().log()))
            Phis.append(float(-sm_para[0][tok_dist].mean().log()))
            embs.append(self.get_emb(toks_para, tok_dist))

        embs.append(self.get_emb(toks_para, self.toker.encode(answer)[1:-1]))

        delta = 0.0
        for i in range(len(embs)):
            for j in range(i):
                delta += 1.0 - float(self.cos(embs[i], embs[j]))

        return phis, Phis, delta
    
    def CDGP(self, text, answer):
        return self._generate_dis(self.unmasker, self.ds_model, text, answer)

    def _generate_dis(self, unmasker, ds_model, sent, answer):
        target_sent = sent + " [SEP] " + answer
        cs = []
        for cand in unmasker(target_sent):
            word = cand["token_str"].replace(" ", "")
            if len(word) > 0:
                cs.append({"word": word, "s0": cand["score"], "s1": 0.0, "s2": 0.0, "s3": 0.0})

        s0s = [c["s0"] for c in cs]
        new_s0s = self._min_max_y(s0s)
        for i, c in enumerate(cs):
            c["s0"] = new_s0s[i]

        answer_vector = ds_model.get_word_vector(answer)
        word_similarities = [self._similarity(answer_vector, ds_model.get_word_vector(c["word"])) for c in cs]
        new_similarities = self._min_max_y(word_similarities)
        for i, c in enumerate(cs):
            c["s1"] = 1 - new_similarities[i]

        correct_sent = sent.replace('[MASK]', answer)
        correct_sent_vector = ds_model.get_sentence_vector(correct_sent)
        cand_sents = [sent.replace('[MASK]', c["word"]) for c in cs]
        sent_similarities = [self._similarity(correct_sent_vector, ds_model.get_sentence_vector(s)) for s in cand_sents]
        new_similarities = self._min_max_y(sent_similarities)
        for i, c in enumerate(cs):
            c["s2"] = 1 - new_similarities[i]

        origin_token = word_tokenize(sent)
        origin_token = [tok for tok in origin_token if tok not in ['[', ']']]
        mask_index = origin_token.index("MASK")

        correct_token = word_tokenize(correct_sent)
        answer_pos = nltk.pos_tag(correct_token)[mask_index]
        for i, c in enumerate(cs):
            cand_token = word_tokenize(cand_sents[i])
            cand_pos = nltk.pos_tag(cand_token)[mask_index]
            c["s3"] = 1.0 if cand_pos[1] == answer_pos[1] else 0.0

        cs_rank = [(c["word"], sum(self.WEIGHT[k] * c[k] for k in self.WEIGHT)) for c in cs]
        cs_rank.sort(key=lambda x: x[1], reverse=True)
        return [d[0] for d in cs_rank[:self.TOP_K]]

    def _similarity(self, v1, v2):
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 1
        return np.dot(v1, v2) / (n1 * n2)

    def _min_max_y(self, raw_data):
        return [(d - min(raw_data)) / (max(raw_data) - min(raw_data)) if max(raw_data) != min(raw_data) else 0.0 for d in raw_data]
    
    
    def process(self, use_cdgp: bool = False):
        # load & clean
        with open(self.input_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self._clean_mcq_text_fields()

        # save the cleaned MC before distractors
        base, ext = os.path.splitext(self.output_path)
        fix_start_path = f"{base}_fix_start{ext}"
        os.makedirs(os.path.dirname(fix_start_path), exist_ok=True)
        with open(fix_start_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        logging.info(f"Written grammar-fixed MC to {fix_start_path}")

        # ─── Preload all source passages into cache ─────────────────────────────
        self._source_cache = {}
        for fname in os.listdir(self.source_dir):
            if not fname.endswith("_processed.json"):
                continue
            try:
                full_path = os.path.join(self.source_dir, fname)
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                isbn = fname.replace("_processed.json", "")
                self._source_cache[isbn] = data
            except Exception as ex:
                logging.warning(f"Failed to load {fname}: {ex}")

        # ─── Loop and generate distractors ───────────────────────────────────────
        total = len(self.data)
        logging.info(f"Starting distractor generation on {total} entries from {self.input_path}")
        success_count, skip_count = 0, 0

        for idx, e in enumerate(self.data):
            question = e.get("question", "")
            answer   = e.get("correct_answer", "")
            qid      = e.get("source", {}).get("paragraph_id", f"index_{idx}")
            isbn     = e.get("source", {}).get("isbn", "")

            passage = ""
            src = self._source_cache.get(isbn)
            if src:
                for rec in src.get("Paragraphs", []):
                    if rec.get("Paragraph ID") == qid:
                        passage = rec.get("Text", "").replace("\n", " ")
                        break
                if not passage:
                    logging.warning(f"[QID={qid}] Paragraph ID not found in cached file for ISBN={isbn}")
            else:
                logging.warning(f"[QID={qid}] No cached source file for ISBN={isbn}")

            try:
                if not passage:
                    raise ValueError(f"[QID={qid}] no passage text to mask")

                logging.info(f"[QID={qid}] Question: {question}")
                logging.info(f"[QID={qid}] Answer  : {answer}")

                # build masked sentence
                text_for_masking = f"{question} {answer} {passage}"
                answer_tokens = self.mask_tokenizer.tokenize(answer)
                if not answer_tokens:
                    raise ValueError(f"[QID={qid}] Cannot tokenize answer: '{answer}'")
                mask_span     = " ".join([self.mask_token] * len(answer_tokens))
                if mask_span not in text_for_masking:
                    logging.warning(f"[QID={qid}] Mask span '{mask_span}' not found in text. Skipping.")
                    raise ValueError(f"Mask span '{mask_span}' could not be inserted.")

                masked_sent   = text_for_masking.replace(answer, mask_span)
                logging.info(f"[QID={qid}] masked_sent for MLM: {masked_sent}")

                # fetch any existing wrong options
                existing_opts = [opt for opt in e.get("options", []) if opt and opt != answer]

                logging.debug(f"[QID={qid}] Full masking context: {text_for_masking}")
                logging.debug(f"[QID={qid}] Tokenized answer: {answer_tokens}")
                logging.debug(f"[QID={qid}] Generated mask_span: {mask_span}")
                logging.debug(f"[QID={qid}] Final masked_sent: {masked_sent}")


                # generate distractors via your pipeline
                if use_cdgp:
                    ds = self.create_distractors_cdgp(qid, masked_sent, answer)

                    distractors = [d for d in ds if d not in existing_opts][: self.select_n]
                else:
                    distractors = self.create_distractors(
                        qid, 
                        masked_sent,
                        answer,
                        existing_opts=existing_opts
                    )

                # write out exactly N distractors
                opts = [answer] + distractors
                random.shuffle(opts)
                e["options"] = opts
                success_count += 1

            except Exception as ex:
                logging.error(
                    f"[QID={qid}] Skipping item due to error: {ex}\n"
                    f"Question: {question}\nPassage: {passage}"
                )
                e["options"] = [answer] + [""] * (self.select_n)
                skip_count += 1

            if (idx + 1) % 1 == 0:
                partial_path = f"{base}_partial_{idx+1}{ext}"
                with open(partial_path, "w", encoding="utf-8") as pf:
                    json.dump(self.data[:idx+1], pf, ensure_ascii=False, indent=2)
                logging.info(f"Saved progress checkpoint to {partial_path} at index {idx+1}")


        # save final MC with distractors
        fix_distractor_path = f"{base}_fix_distractor{ext}"
        os.makedirs(os.path.dirname(fix_distractor_path), exist_ok=True)
        with open(fix_distractor_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

        logging.info(
            f"Written updated MC with distractors to {fix_distractor_path} — "
            f"{success_count} succeeded, {skip_count} skipped."
        )




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="optimize_extracted_qa_dataset.py",
        description="Clean short_inverse.json and/or regenerate MC.json distractors"
    )
    parser.add_argument("--source-dir",    required=True,
                        help="folder of *_processed.json source texts")
    parser.add_argument("--input-dir",     required=True,
                        help="folder containing short_inverse.json and MC.json")
    parser.add_argument("--output-dir",    required=True,
                        help="folder to save outputs")
    parser.add_argument("--unix-wordlist", required=True,
                        help="path to dict-unix.txt")
    parser.add_argument("--info-wordlist", required=True,
                        help="path to dict-info.txt")
    parser.add_argument("--profanity-file",required=True,
                        help="path to profanity.json")
    parser.add_argument("--top-k",    type=int,   default=64,
                        help="number of mask-fill candidates to retrieve")
    parser.add_argument("--select-n", type=int,   default=4,
                        help="number of distractors to select per question")
    parser.add_argument("--alpha",    type=float, default=0.3,
                        help="weight on incorrectness score when ranking")
    parser.add_argument("--beta",     type=float, default=0.3,
                        help="weight on distinctiveness score when ranking")
    parser.add_argument("--clean", action="store_true",
                        help="only run the short_inverse cleaning step")
    parser.add_argument("--distractors", action="store_true",
                        help="only run the MC.json distractor regeneration step")

    parser.add_argument("--method", type=str, default="default", choices=["default", "cdgp"],
                    help="Which method to use for distractor generation: 'default' or 'cdgp'")
    
    parser.add_argument("--distractor-pool", type=int, default=64)
    parser.add_argument("--min-dist", type=int, default=5)
    parser.add_argument("--min-sent-words", type=int, default=6)
    parser.add_argument("--max-subwords", type=int, default=100)
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")


    parser.add_argument(
        "--distractors-from-text",
        action="store_true",
        default=False,
        help="Require distractors to come from the source text only"
    )

    parser.add_argument(
        "--mask-model-path",
        type=str,
        default="allenai/longformer-base-4096",
        help="either a HF model name or a local directory"
    )
    parser.add_argument(
        "--embed-model-path",
        type=str,
        default="allenai/longformer-base-4096",
        help="either a HF model name or a local directory"
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default="output.log",
        help="Path to log file (default: output.log)"
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        filename=args.log_file,
        filemode="w",
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    logger = logging.getLogger(__name__)

    # if neither flag is provided, run both steps
    if not args.clean and not args.distractors:
        args.clean = args.distractors = True

    os.makedirs(args.output_dir, exist_ok=True)

    # ─── SHORT_INVERSE CLEANING ────────────────────────────────────────────────
    if args.clean:
        short_in  = os.path.join(args.input_dir, "short_inverse.json")
        short_out = os.path.join(args.output_dir, "short_inverse_cleaned.json")
        cleaner = ExternalRefCleaner()
        with open(short_in, 'r', encoding='utf-8') as f:
            sinv = json.load(f)

        cleaned = []
        for entry in sinv:
            # clean the question if present
            if 'question' in entry:
                entry['question'] = cleaner.clean(entry['question'])
            if 'answer' in entry:
                entry['answer'] = cleaner.clean(entry['answer'])
            if 'false_answer' in entry:
                entry['false_answer'] = cleaner.clean(entry['false_answer'])
            # clean any incorrect_explanation
            if 'incorrect_explanation' in entry:
                entry['incorrect_explanation'] = cleaner.clean(entry['incorrect_explanation'])
            cleaned.append(entry)

        with open(short_out, 'w', encoding='utf-8') as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)


    # ─── MC.JSON DISTRACTOR REGENERATION ───────────────────────────────────────
    if args.distractors:
        mc_in  = os.path.join(args.input_dir, "MC.json")
        mc_out = os.path.join(args.output_dir, "MC_updated.json")
        runner = DistractorPipeline(
            source_dir          = args.source_dir,
            unix_wordlist_path  = args.unix_wordlist,
            info_wordlist_path  = args.info_wordlist,
            profanity_path      = args.profanity_file,
            input_path          = mc_in,
            output_path         = mc_out,
            mask_model_name     = args.mask_model_path,
            embed_model_name    = args.embed_model_path,
            top_k               = args.top_k,
            select_n            = args.select_n,
            alpha               = args.alpha,
            beta                = args.beta,
            distractor_pool     = args.distractor_pool,
            min_dist            = args.min_dist,
            min_sent_words      = args.min_sent_words,
            max_subwords        = args.max_subwords,
            debug_output        = args.debug,
            use_cdgp            = (args.method == "cdgp"),
            distractors_from_text = args.distractors_from_text
        )


        runner.process(use_cdgp=(args.method == "cdgp"))

    # ─── WRITE README FILES ───────────────────────────────────────────────────
    # Only write a README for each output that actually exists.
    for filename, readme_name in [
        ("short_inverse_cleaned.json", "README_short_inverse_cleaned.md"),
        ("MC_updated_fix_start.json",            "README_MC_updated_fixed_start.md"),
        ("MC_updated_fix_distractor.json",            "README_MC_updated_fixed_distractor.md"),
    ]:
        out_path = os.path.join(args.output_dir, filename)
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            count = len(data)
            readme_text = (
                f"# {filename}\n\n"
                f"This file contains **{count}** question-answer pairs.\n"
            )
            with open(os.path.join(args.output_dir, readme_name), "w", encoding="utf-8") as f:
                f.write(readme_text)
            logging.info(f"Wrote {readme_name} with {count} entries")
