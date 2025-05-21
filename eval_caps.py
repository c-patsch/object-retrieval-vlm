from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice import Spice

import numpy as np
import re

def convert_to_present_tense(caption):
    caption = re.sub(r'\b(is|are|was|were) (\w+?)ing\b', r'\2s', caption)
    return caption

tokenizer = PTBTokenizer()

splits = [1] #[1, 2, 3, 4]

# Initialize score containers
bleu_scores_all = []
cider_scores = []
rouge_scores = []
meteor_scores = []
spice_scores = []

eval_folder = "wild_dataset" #"minigpt4_output_object"

for split in splits:


    #User Study
    gts = []
    with open(f"{eval_folder}/gt_user.txt", 'r') as f:
        gts = [line.rstrip('\n') for line in f]
    res = []
    with open(f"{eval_folder}/pred_noobj_user.txt", 'r') as f:
    #with open(f"{eval_folder}/pred_user.txt", 'r') as f:
        res = [line.rstrip('\n') for line in f]
    ###############
    # #GTEA
    # gts = list(np.load(f"{eval_folder}/all_gts_split{split}.npy"))
    # res = list(np.load(f"{eval_folder}/all_preds_split{split}.npy"))
    ###############
    
    gts = {f"img{i+1}": [{"caption": caption.replace('The', 'A')}] for i, caption in enumerate(gts)}
    res = {f"img{i+1}": [{"caption": convert_to_present_tense(caption)}] for i, caption in enumerate(res)}

    # Tokenize
    gts_tokenized = tokenizer.tokenize(gts)
    res_tokenized = tokenizer.tokenize(res)

    # BLEU
    bleu_scorer = Bleu(4)
    bleu_score, _ = bleu_scorer.compute_score(gts_tokenized, res_tokenized)
    bleu_scores_all.append(bleu_score)

    # CIDEr
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts_tokenized, res_tokenized)
    cider_scores.append(cider_score)

    # ROUGE-L
    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(gts_tokenized, res_tokenized)
    rouge_scores.append(rouge_score)

    # METEOR
    meteor_scorer = Meteor()
    meteor_score, _ = meteor_scorer.compute_score(gts_tokenized, res_tokenized)
    meteor_scores.append(meteor_score)

    # SPICE
    spice_scorer = Spice()
    spice_score, _ = spice_scorer.compute_score(gts_tokenized, res_tokenized)
    spice_scores.append(spice_score)

# Average scores
bleu_scores_all = np.array(bleu_scores_all)
avg_bleu = bleu_scores_all.mean(axis=0)
avg_cider = np.mean(cider_scores)
avg_rouge = np.mean(rouge_scores)
avg_meteor = np.mean(meteor_scores)
avg_spice = np.mean(spice_scores)

# Print results
print("Average BLEU-1 to BLEU-4:", avg_bleu)
print("Average ROUGE-L:", avg_rouge)
print("Average METEOR:", avg_meteor)
print("Average CIDEr:", avg_cider)
print("Average SPICE:", avg_spice)
