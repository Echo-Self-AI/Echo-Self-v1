"""Language model evaluation utilities.

This module provides evaluation metrics and utilities for assessing
language model quality beyond training/validation loss.
"""

import math
import re
from collections import Counter
from typing import List, Dict, Any, Optional
import torch
from torch import nn


def calculate_perplexity(loss: float, max_loss: float = 20.0) -> float:
    """Calculate perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss value
        max_loss: Maximum loss to cap before exp (prevents overflow)
        
    Returns:
        Perplexity value
    """
    clamped_loss = min(loss, max_loss)
    return math.exp(clamped_loss)


def evaluate_repetition(text: str, n: int = 4) -> Dict[str, float]:
    """Evaluate repetition in generated text using n-gram analysis.
    
    Args:
        text: Generated text to analyze
        n: N-gram size (default 4-grams)
        
    Returns:
        Dictionary with repetition metrics:
        - unique_ngrams_ratio: ratio of unique n-grams to total n-grams
        - max_ngram_frequency: maximum frequency of any n-gram
        - repetition_score: 1.0 - unique_ngrams_ratio (higher = more repetitive)
    """
    # Tokenize by whitespace
    tokens = text.split()
    
    if len(tokens) < n:
        return {
            "unique_ngrams_ratio": 1.0,
            "max_ngram_frequency": 1,
            "repetition_score": 0.0,
            "total_ngrams": len(tokens),
        }
    
    # Extract n-grams
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    
    if not ngrams:
        return {
            "unique_ngrams_ratio": 1.0,
            "max_ngram_frequency": 1,
            "repetition_score": 0.0,
            "total_ngrams": 0,
        }
    
    # Count n-gram frequencies
    ngram_counts = Counter(ngrams)
    unique_count = len(ngram_counts)
    total_count = len(ngrams)
    max_frequency = max(ngram_counts.values())
    
    unique_ratio = unique_count / total_count if total_count > 0 else 1.0
    repetition_score = 1.0 - unique_ratio
    
    return {
        "unique_ngrams_ratio": unique_ratio,
        "max_ngram_frequency": max_frequency,
        "repetition_score": repetition_score,
        "total_ngrams": total_count,
    }


def evaluate_diversity(texts: List[str]) -> Dict[str, float]:
    """Evaluate diversity across multiple generated samples.
    
    Args:
        texts: List of generated text samples
        
    Returns:
        Dictionary with diversity metrics:
        - unique_texts: number of unique texts
        - unique_ratio: ratio of unique to total texts
        - avg_length: average text length
        - length_variance: variance in text lengths
    """
    if not texts:
        return {
            "unique_texts": 0,
            "unique_ratio": 0.0,
            "avg_length": 0.0,
            "length_variance": 0.0,
        }
    
    unique_texts = len(set(texts))
    unique_ratio = unique_texts / len(texts)
    
    lengths = [len(text) for text in texts]
    avg_length = sum(lengths) / len(lengths)
    
    if len(lengths) > 1:
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
    else:
        variance = 0.0
    
    return {
        "unique_texts": unique_texts,
        "unique_ratio": unique_ratio,
        "avg_length": avg_length,
        "length_variance": variance,
    }


def evaluate_coherence(text: str) -> Dict[str, Any]:
    """Evaluate basic coherence metrics of generated text.
    
    Args:
        text: Generated text to analyze
        
    Returns:
        Dictionary with coherence-related metrics:
        - sentence_count: number of sentences (approx)
        - avg_sentence_length: average words per sentence
        - has_punctuation: whether text contains punctuation
        - has_capitalization: whether text has varied case
    """
    # Simple sentence splitting (approximate)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    sentence_count = len(sentences)
    
    if sentence_count > 0:
        words_per_sentence = [len(s.split()) for s in sentences]
        avg_sentence_length = sum(words_per_sentence) / len(words_per_sentence)
    else:
        avg_sentence_length = 0.0
    
    has_punctuation = bool(re.search(r'[.!?,;:]', text))
    has_upper = bool(re.search(r'[A-Z]', text))
    has_lower = bool(re.search(r'[a-z]', text))
    has_capitalization = has_upper and has_lower
    
    return {
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "has_punctuation": has_punctuation,
        "has_capitalization": has_capitalization,
    }


def generate_evaluation_samples(
    model: nn.Module,
    device: torch.device,
    prompts: List[str],
    tokenizer,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: Optional[int] = 5,
    samples_per_prompt: int = 3,
) -> List[Dict[str, Any]]:
    """Generate evaluation samples from fixed prompts.
    
    Args:
        model: Language model to evaluate
        device: Torch device
        prompts: List of prompt strings
        tokenizer: Tokenizer with encode/decode methods
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        samples_per_prompt: Number of samples per prompt
        
    Returns:
        List of dictionaries containing prompt, generated texts, and metrics
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for prompt in prompts:
            # Encode prompt
            if hasattr(tokenizer, 'encode'):
                try:
                    encoded_prompt = tokenizer.encode(prompt)
                    if not encoded_prompt:
                        # Fallback to first token
                        encoded_prompt = [0]
                except (KeyError, AttributeError):
                    # Fallback for character tokenizer
                    if hasattr(tokenizer, 'stoi'):
                        valid_chars = [c for c in prompt if c in tokenizer.stoi]
                        if valid_chars:
                            encoded_prompt = [tokenizer.stoi[c] for c in valid_chars]
                        else:
                            encoded_prompt = [0]
                    else:
                        encoded_prompt = [0]
            else:
                encoded_prompt = [0]
            
            generated_texts = []
            
            for _ in range(samples_per_prompt):
                prompt_tensor = torch.tensor(
                    [encoded_prompt],
                    dtype=torch.long,
                    device=device
                )
                
                generated = model.generate(
                    prompt_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                )
                
                generated_tokens = generated[0].tolist()
                
                if hasattr(tokenizer, 'decode'):
                    generated_text = tokenizer.decode(generated_tokens)
                else:
                    # Fallback for character tokenizer
                    if hasattr(tokenizer, 'characters'):
                        generated_text = ''.join(
                            tokenizer.characters[t] if t < len(tokenizer.characters) else '?'
                            for t in generated_tokens
                        )
                    else:
                        generated_text = str(generated_tokens)
                
                generated_texts.append(generated_text)
            
            # Calculate metrics for this prompt
            repetition_metrics = [
                evaluate_repetition(text) for text in generated_texts
            ]
            coherence_metrics = [
                evaluate_coherence(text) for text in generated_texts
            ]
            diversity_metrics = evaluate_diversity(generated_texts)
            
            results.append({
                "prompt": prompt,
                "generated_texts": generated_texts,
                "repetition_metrics": repetition_metrics,
                "coherence_metrics": coherence_metrics,
                "diversity_metrics": diversity_metrics,
            })
    
    return results


def format_evaluation_report(results: List[Dict[str, Any]]) -> str:
    """Format evaluation results into a readable report.
    
    Args:
        results: Evaluation results from generate_evaluation_samples
        
    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("LANGUAGE MODEL EVALUATION REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    for i, result in enumerate(results, 1):
        lines.append(f"Prompt {i}: '{result['prompt']}'")
        lines.append("-" * 80)
        
        # Show generated samples
        for j, text in enumerate(result['generated_texts'], 1):
            lines.append(f"\n  Sample {j}:")
            # Truncate for display if too long
            display_text = text[:200] + "..." if len(text) > 200 else text
            lines.append(f"  {display_text}")
        
        lines.append("")
        
        # Aggregate metrics
        rep_metrics = result['repetition_metrics']
        avg_repetition = sum(m['repetition_score'] for m in rep_metrics) / len(rep_metrics)
        avg_unique_ratio = sum(m['unique_ngrams_ratio'] for m in rep_metrics) / len(rep_metrics)
        
        coh_metrics = result['coherence_metrics']
        avg_sentence_len = sum(m['avg_sentence_length'] for m in coh_metrics) / len(coh_metrics)
        
        div_metrics = result['diversity_metrics']
        
        lines.append(f"  Repetition Score (avg):   {avg_repetition:.3f} (lower is better)")
        lines.append(f"  Unique 4-grams (avg):     {avg_unique_ratio:.3f} (higher is better)")
        lines.append(f"  Avg Sentence Length:      {avg_sentence_len:.1f} words")
        lines.append(f"  Sample Diversity:         {div_metrics['unique_ratio']:.3f}")
        lines.append("")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)
