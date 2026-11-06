"""Experiment tracking and management.

This module provides utilities for tracking experiments, configurations,
and results in a structured, reproducible manner.
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List


class ExperimentTracker:
    """Track experiments with configurations, results, and metadata."""
    
    def __init__(self, experiments_dir: Path = Path("experiments")):
        """Initialize experiment tracker.
        
        Args:
            experiments_dir: Directory to store experiment records
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(exist_ok=True)
    
    def create_experiment(
        self,
        name: str,
        experiment_type: str,
        config: Dict[str, Any],
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> str:
        """Create a new experiment record.
        
        Args:
            name: Experiment name
            experiment_type: Type (e.g., 'language_model', 'acoustic_model')
            config: Configuration dictionary
            description: Human-readable description
            tags: Optional list of tags
            
        Returns:
            Experiment ID (hash-based unique identifier)
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Create experiment ID from config hash + timestamp
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]
        experiment_id = f"{experiment_type}_{config_hash}_{timestamp.replace(':', '-').replace('.', '-')}"
        
        experiment_record = {
            "id": experiment_id,
            "name": name,
            "type": experiment_type,
            "description": description,
            "tags": tags or [],
            "config": config,
            "created_at": timestamp,
            "status": "created",
            "results": {},
            "metrics": {},
            "artifacts": {},
        }
        
        experiment_file = self.experiments_dir / f"{experiment_id}.json"
        with open(experiment_file, 'w') as f:
            json.dump(experiment_record, f, indent=2)
        
        return experiment_id
    
    def update_experiment(
        self,
        experiment_id: str,
        status: Optional[str] = None,
        results: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        artifacts: Optional[Dict[str, str]] = None,
    ) -> None:
        """Update an existing experiment record.
        
        Args:
            experiment_id: Experiment identifier
            status: New status (e.g., 'running', 'completed', 'failed')
            results: Results dictionary to update
            metrics: Metrics dictionary to update
            artifacts: Artifacts dictionary (name -> path) to update
        """
        experiment_file = self.experiments_dir / f"{experiment_id}.json"
        
        if not experiment_file.exists():
            raise ValueError(f"Experiment {experiment_id} not found")
        
        with open(experiment_file, 'r') as f:
            experiment_record = json.load(f)
        
        # Update fields
        experiment_record["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        if status is not None:
            experiment_record["status"] = status
        
        if results is not None:
            experiment_record["results"].update(results)
        
        if metrics is not None:
            experiment_record["metrics"].update(metrics)
        
        if artifacts is not None:
            experiment_record["artifacts"].update(artifacts)
        
        # Write back
        with open(experiment_file, 'w') as f:
            json.dump(experiment_record, f, indent=2)
    
    def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Retrieve an experiment record.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Experiment record dictionary
        """
        experiment_file = self.experiments_dir / f"{experiment_id}.json"
        
        if not experiment_file.exists():
            raise ValueError(f"Experiment {experiment_id} not found")
        
        with open(experiment_file, 'r') as f:
            return json.load(f)
    
    def list_experiments(
        self,
        experiment_type: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """List experiments matching criteria.
        
        Args:
            experiment_type: Filter by type
            status: Filter by status
            tags: Filter by tags (experiments must have all specified tags)
            
        Returns:
            List of matching experiment records
        """
        experiments = []
        
        for experiment_file in self.experiments_dir.glob("*.json"):
            with open(experiment_file, 'r') as f:
                record = json.load(f)
            
            # Apply filters
            if experiment_type and record.get("type") != experiment_type:
                continue
            
            if status and record.get("status") != status:
                continue
            
            if tags:
                record_tags = set(record.get("tags", []))
                if not all(tag in record_tags for tag in tags):
                    continue
            
            experiments.append(record)
        
        # Sort by created_at descending
        experiments.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return experiments
    
    def format_experiment_summary(self, experiment_id: str) -> str:
        """Format experiment record as a readable summary.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Formatted string summary
        """
        record = self.get_experiment(experiment_id)
        
        lines = []
        lines.append("=" * 60)
        lines.append(f"Experiment: {record['name']}")
        lines.append("=" * 60)
        lines.append(f"ID:          {record['id']}")
        lines.append(f"Type:        {record['type']}")
        lines.append(f"Status:      {record['status']}")
        lines.append(f"Created:     {record['created_at']}")
        
        if record.get('description'):
            lines.append(f"Description: {record['description']}")
        
        if record.get('tags'):
            lines.append(f"Tags:        {', '.join(record['tags'])}")
        
        lines.append("")
        lines.append("Configuration:")
        lines.append("-" * 60)
        for key, value in record['config'].items():
            lines.append(f"  {key}: {value}")
        
        if record.get('metrics'):
            lines.append("")
            lines.append("Metrics:")
            lines.append("-" * 60)
            for key, value in record['metrics'].items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")
        
        if record.get('artifacts'):
            lines.append("")
            lines.append("Artifacts:")
            lines.append("-" * 60)
            for name, path in record['artifacts'].items():
                lines.append(f"  {name}: {path}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics: Optional[List[str]] = None,
    ) -> str:
        """Compare multiple experiments side-by-side.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: Optional list of specific metrics to compare
            
        Returns:
            Formatted comparison string
        """
        records = [self.get_experiment(exp_id) for exp_id in experiment_ids]
        
        lines = []
        lines.append("=" * 80)
        lines.append("EXPERIMENT COMPARISON")
        lines.append("=" * 80)
        lines.append("")
        
        # Show names and IDs
        for i, record in enumerate(records, 1):
            lines.append(f"Experiment {i}: {record['name']}")
            lines.append(f"  ID: {record['id'][:30]}...")
            lines.append(f"  Status: {record['status']}")
            lines.append("")
        
        # Compare configurations
        lines.append("Configuration Differences:")
        lines.append("-" * 80)
        
        all_config_keys = set()
        for record in records:
            all_config_keys.update(record['config'].keys())
        
        for key in sorted(all_config_keys):
            values = [record['config'].get(key, 'N/A') for record in records]
            if len(set(str(v) for v in values)) > 1:  # Only show differences
                lines.append(f"  {key}:")
                for i, value in enumerate(values, 1):
                    lines.append(f"    Exp {i}: {value}")
        
        # Compare metrics
        if metrics is None:
            # Get all metrics from all experiments
            all_metrics = set()
            for record in records:
                all_metrics.update(record.get('metrics', {}).keys())
            metrics = sorted(all_metrics)
        
        if metrics:
            lines.append("")
            lines.append("Metric Comparison:")
            lines.append("-" * 80)
            
            for metric in metrics:
                values = []
                for record in records:
                    value = record.get('metrics', {}).get(metric)
                    if value is not None:
                        if isinstance(value, float):
                            values.append(f"{value:.4f}")
                        else:
                            values.append(str(value))
                    else:
                        values.append("N/A")
                
                lines.append(f"  {metric}:")
                for i, value in enumerate(values, 1):
                    lines.append(f"    Exp {i}: {value}")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
