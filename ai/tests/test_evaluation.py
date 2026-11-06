"""Tests for language model evaluation utilities."""

import pytest
import torch
from echoself_ai.evaluation import (
    calculate_perplexity,
    evaluate_repetition,
    evaluate_diversity,
    evaluate_coherence,
)


def test_calculate_perplexity():
    """Test perplexity calculation from loss."""
    # Low loss should give low perplexity
    assert calculate_perplexity(0.0) == pytest.approx(1.0)
    
    # Higher loss gives higher perplexity
    ppl_low = calculate_perplexity(1.0)
    ppl_high = calculate_perplexity(2.0)
    assert ppl_high > ppl_low
    
    # Very high loss should be clamped
    ppl_max = calculate_perplexity(100.0, max_loss=20.0)
    assert ppl_max < float('inf')


def test_evaluate_repetition_unique():
    """Test repetition evaluation on non-repetitive text."""
    text = "the quick brown fox jumps over the lazy dog"
    metrics = evaluate_repetition(text, n=4)
    
    assert 'unique_ngrams_ratio' in metrics
    assert 'repetition_score' in metrics
    assert 'max_ngram_frequency' in metrics
    
    # This text has unique 4-grams
    assert metrics['unique_ngrams_ratio'] == 1.0
    assert metrics['repetition_score'] == 0.0
    assert metrics['max_ngram_frequency'] == 1


def test_evaluate_repetition_repetitive():
    """Test repetition evaluation on highly repetitive text."""
    text = "the the the the the the the the"
    metrics = evaluate_repetition(text, n=4)
    
    # High repetition
    assert metrics['repetition_score'] > 0.5
    assert metrics['max_ngram_frequency'] > 1


def test_evaluate_repetition_short():
    """Test repetition evaluation on very short text."""
    text = "hi"
    metrics = evaluate_repetition(text, n=4)
    
    # Short text handled gracefully
    assert metrics['unique_ngrams_ratio'] == 1.0
    assert metrics['total_ngrams'] == 1


def test_evaluate_diversity_unique():
    """Test diversity evaluation with unique samples."""
    texts = [
        "hello world",
        "goodbye moon",
        "the quick fox",
    ]
    metrics = evaluate_diversity(texts)
    
    assert metrics['unique_texts'] == 3
    assert metrics['unique_ratio'] == 1.0
    assert metrics['avg_length'] > 0


def test_evaluate_diversity_duplicates():
    """Test diversity evaluation with duplicate samples."""
    texts = [
        "hello world",
        "hello world",
        "hello world",
    ]
    metrics = evaluate_diversity(texts)
    
    assert metrics['unique_texts'] == 1
    assert metrics['unique_ratio'] == pytest.approx(1/3)
    assert metrics['length_variance'] == 0.0


def test_evaluate_diversity_empty():
    """Test diversity evaluation with empty list."""
    metrics = evaluate_diversity([])
    
    assert metrics['unique_texts'] == 0
    assert metrics['unique_ratio'] == 0.0


def test_evaluate_coherence_structured():
    """Test coherence evaluation on well-structured text."""
    text = "Hello world. This is a test. How are you?"
    metrics = evaluate_coherence(text)
    
    assert metrics['sentence_count'] == 3
    assert metrics['avg_sentence_length'] > 0
    assert metrics['has_punctuation'] is True
    assert metrics['has_capitalization'] is True


def test_evaluate_coherence_unstructured():
    """Test coherence evaluation on unstructured text."""
    text = "hello hello hello hello"
    metrics = evaluate_coherence(text)
    
    assert metrics['sentence_count'] == 1
    assert metrics['has_punctuation'] is False
    assert metrics['has_capitalization'] is False


def test_evaluate_coherence_empty():
    """Test coherence evaluation on empty string."""
    text = ""
    metrics = evaluate_coherence(text)
    
    assert metrics['sentence_count'] == 0
    assert metrics['avg_sentence_length'] == 0.0
