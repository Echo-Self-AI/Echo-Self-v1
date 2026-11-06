"""Tests for experiment tracking system."""

import pytest
import tempfile
from pathlib import Path
from echoself_ai.experiment import ExperimentTracker


@pytest.fixture
def temp_experiments_dir():
    """Create temporary experiments directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_create_experiment(temp_experiments_dir):
    """Test experiment creation."""
    tracker = ExperimentTracker(temp_experiments_dir)
    
    config = {
        "learning_rate": 0.001,
        "batch_size": 8,
        "n_layers": 4,
    }
    
    exp_id = tracker.create_experiment(
        name="test_experiment",
        experiment_type="language_model",
        config=config,
        description="Test experiment",
        tags=["test", "baseline"],
    )
    
    assert exp_id.startswith("language_model_")
    
    # Verify file created
    exp_file = temp_experiments_dir / f"{exp_id}.json"
    assert exp_file.exists()


def test_get_experiment(temp_experiments_dir):
    """Test retrieving experiment record."""
    tracker = ExperimentTracker(temp_experiments_dir)
    
    config = {"lr": 0.001}
    exp_id = tracker.create_experiment(
        name="test",
        experiment_type="test",
        config=config,
    )
    
    record = tracker.get_experiment(exp_id)
    
    assert record['id'] == exp_id
    assert record['name'] == "test"
    assert record['config'] == config
    assert record['status'] == "created"


def test_update_experiment(temp_experiments_dir):
    """Test updating experiment record."""
    tracker = ExperimentTracker(temp_experiments_dir)
    
    exp_id = tracker.create_experiment(
        name="test",
        experiment_type="test",
        config={},
    )
    
    tracker.update_experiment(
        exp_id,
        status="completed",
        metrics={"loss": 2.5, "accuracy": 0.85},
        artifacts={"model": "checkpoints/model.pt"},
    )
    
    record = tracker.get_experiment(exp_id)
    
    assert record['status'] == "completed"
    assert record['metrics']['loss'] == 2.5
    assert record['artifacts']['model'] == "checkpoints/model.pt"
    assert 'updated_at' in record


def test_list_experiments_empty(temp_experiments_dir):
    """Test listing experiments when none exist."""
    tracker = ExperimentTracker(temp_experiments_dir)
    
    experiments = tracker.list_experiments()
    assert len(experiments) == 0


def test_list_experiments_with_filters(temp_experiments_dir):
    """Test listing experiments with filters."""
    tracker = ExperimentTracker(temp_experiments_dir)
    
    # Create multiple experiments
    exp1 = tracker.create_experiment(
        name="exp1",
        experiment_type="language_model",
        config={},
        tags=["baseline"],
    )
    
    exp2 = tracker.create_experiment(
        name="exp2",
        experiment_type="acoustic_model",
        config={},
        tags=["test"],
    )
    
    exp3 = tracker.create_experiment(
        name="exp3",
        experiment_type="language_model",
        config={},
        tags=["improved"],
    )
    
    # Update statuses
    tracker.update_experiment(exp1, status="completed")
    tracker.update_experiment(exp2, status="running")
    tracker.update_experiment(exp3, status="completed")
    
    # Filter by type
    lm_exps = tracker.list_experiments(experiment_type="language_model")
    assert len(lm_exps) == 2
    
    # Filter by status
    completed = tracker.list_experiments(status="completed")
    assert len(completed) == 2
    
    # Filter by tags
    baseline = tracker.list_experiments(tags=["baseline"])
    assert len(baseline) == 1
    assert baseline[0]['name'] == "exp1"


def test_format_experiment_summary(temp_experiments_dir):
    """Test formatting experiment summary."""
    tracker = ExperimentTracker(temp_experiments_dir)
    
    config = {"lr": 0.001, "batch_size": 8}
    exp_id = tracker.create_experiment(
        name="test_exp",
        experiment_type="test",
        config=config,
        description="Test experiment",
    )
    
    tracker.update_experiment(
        exp_id,
        metrics={"loss": 2.5},
        artifacts={"checkpoint": "model.pt"},
    )
    
    summary = tracker.format_experiment_summary(exp_id)
    
    assert "test_exp" in summary
    assert "lr: 0.001" in summary
    assert "loss: 2.5" in summary
    assert "checkpoint: model.pt" in summary


def test_compare_experiments(temp_experiments_dir):
    """Test comparing multiple experiments."""
    tracker = ExperimentTracker(temp_experiments_dir)
    
    exp1 = tracker.create_experiment(
        name="exp1",
        experiment_type="test",
        config={"lr": 0.001, "batch_size": 8},
    )
    
    exp2 = tracker.create_experiment(
        name="exp2",
        experiment_type="test",
        config={"lr": 0.01, "batch_size": 16},
    )
    
    tracker.update_experiment(exp1, metrics={"loss": 2.5})
    tracker.update_experiment(exp2, metrics={"loss": 2.3})
    
    comparison = tracker.compare_experiments([exp1, exp2])
    
    assert "exp1" in comparison
    assert "exp2" in comparison
    assert "loss:" in comparison
    assert "2.5" in comparison
    assert "2.3" in comparison
