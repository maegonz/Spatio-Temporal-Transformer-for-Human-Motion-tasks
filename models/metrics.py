import torch
import torch.nn.functional as F
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from pycocoevalcap.cider.cider import Cider
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from typing import Optional

nltk.download('omw-1.4', quiet=True)
nltk.download('wordnet', quiet=True)

## ================================================================
## ------------------------ Motion2Text Metrics -----------------
## ================================================================

def cross_entropy_loss(logits, target, label_smoothing: float=0.1, pad_token_id: Optional[int]=-100):
    """
    Compute label-smoothed cross-entropy over a sequence, ignoring padding tokens.

    Parameters
    ----------
    logits : torch.Tensor
        Logits of shape (batch_size, seq_len, vocab_size).
    target : torch.Tensor
        Target token IDs of shape (batch_size, seq_len).
    label_smoothing : float, optional
        Amount of label smoothing to apply. Defaults to 0.1.
    pad_token_id : int or None, optional
        Target token ID to ignore. Defaults to -100.

    Returns
    -------
    torch.Tensor
        Scalar tensor containing the average cross-entropy over non-padding tokens.
    """
    labels = target.clone()

    logits = logits.permute(0, 2, 1)  # (batch_size, vocab_size, seq_len)

    # Compute cross-entropy loss, ignoring padding tokens
    loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing, ignore_index=pad_token_id)

    return loss

def contrastive_loss(motion_features, text_features, temperature=0.07):
    """
    motion_features: [batch, dim] (Mean-pooled output of MotionAdapter)
    text_features: [batch, dim] (Mean-pooled hidden states of T5 embeddings)
    """
    # Normalize to get cosine similarity
    motion_features = F.normalize(motion_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    
    # Calculate logits (batch_size, batch_size)
    logits_scale = 1.0 / temperature
    logits = torch.matmul(motion_features, text_features.T) * logits_scale
    
    # Labels are diagonal (each motion matches its own text)
    batch_size = motion_features.size(0)
    labels = torch.arange(batch_size, device=motion_features.device)
    
    loss_m = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T.contiguous(), labels)
    
    return (loss_m + loss_t) * 0.5


class Evaluator:
    def __init__(self):
        self.rouge_scorer_ = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.cider_scorer = Cider()
        self.smoothing = SmoothingFunction().method1

    def compute_metrics(self, references: list[str], generated: list[str]):
        """
        Compute corpus-level metrics for lists of references and generated texts.
        references: list of ground-truth strings
        generated: list of generated strings
        """
        assert len(references) == len(generated), "Must have equal number of refs and gens"
        
        # 1. BLEU (Corpus Level)
        # Corpus bleu expects: [[['ref1_word1', 'ref1_word2']], [['ref2_word1']]]
        refs_bleu = [[ref.split()] for ref in references]
        gens_bleu = [gen.split() for gen in generated]
        bleu_score = corpus_bleu(refs_bleu, gens_bleu, smoothing_function=self.smoothing)

        # 2. CIDEr (Corpus Level)
        # CIDEr expects dicts mapping ID to list of strings: {0: ["ref1"], 1: ["ref2"]}
        refs_cider = {i: [ref] for i, ref in enumerate(references)}
        gens_cider = {i: [gen] for i, gen in enumerate(generated)}
        cider_score, _ = self.cider_scorer.compute_score(refs_cider, gens_cider)

        # 3. METEOR (Average of sentence scores)
        # NLTK meteor_score expects tokenized lists
        meteor_scores = [
            meteor_score([ref.split()], gen.split()) 
            for ref, gen in zip(references, generated)
        ]
        avg_meteor = sum(meteor_scores) / len(meteor_scores)

        # 4. ROUGE (Average of sentence scores)
        rouge1, rouge2, rougeL = 0.0, 0.0, 0.0
        for ref, gen in zip(references, generated):
            scores = self.rouge_scorer_.score(ref, gen)
            rouge1 += scores['rouge1'].fmeasure
            rouge2 += scores['rouge2'].fmeasure
            rougeL += scores['rougeL'].fmeasure
            
        n = len(references)

        return {
            'BLEU': bleu_score,
            'METEOR': avg_meteor,
            'ROUGE-1': rouge1 / n,
            'ROUGE-2': rouge2 / n,
            'ROUGE-L': rougeL / n,
            'CIDEr': cider_score
        }

## ================================================================
## ------------------------ Text2Motion Metrics -----------------
## ================================================================

def mpjpe(predicted_motion: torch.Tensor, ground_truth_motion: torch.Tensor) -> float:
    """
    Compute the Mean Per Joint Position Error (MPJPE) between predicted and ground truth motion sequences.

    Parameters
    ----------
    predicted_motion: torch.Tensor [batch, seq_len, n_joints, spatial_dim]
        Predicted motion sequences
    ground_truth_motion: torch.Tensor [batch, seq_len, n_joints, spatial_dim]
        Ground truth motion sequences

    Returns
    -------
    mpjpe_value: float
        Mean Per Joint Position Error
    """
    error = torch.norm(predicted_motion - ground_truth_motion, dim=-1)  # [batch, seq_len, n_joints]
    mpjpe_value = error.mean()  # Average over batch, seq_len, and joints
    return mpjpe_value