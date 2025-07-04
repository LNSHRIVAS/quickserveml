"""
QuickServeML Model Registry

Enterprise-grade model registry with versioning, metadata management,
and lifecycle tracking. Comparable to AWS SageMaker Model Registry
and Azure ML Model Registry.
"""

import os
import json
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from contextlib import contextmanager

from .cli_utils import get_formatter


class ModelStatus(Enum):
    """Model lifecycle status"""
    DRAFT = "draft"
    VALIDATED = "validated"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class ModelMetadata:
    """Model metadata and versioning information"""
    name: str
    version: str
    description: str = ""
    tags: List[str] = None
    author: str = ""
    created_at: str = ""
    updated_at: str = ""
    status: str = ModelStatus.DRAFT.value
    model_path: str = ""
    model_hash: str = ""
    file_size: int = 0
    framework: str = "onnx"
    accuracy: Optional[float] = None
    latency_ms: Optional[float] = None
    throughput_rps: Optional[float] = None
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    dependencies: Dict[str, str] = None
    deployment_history: List[Dict[str, Any]] = None
    notes: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.dependencies is None:
            self.dependencies = {}
        if self.deployment_history is None:
            self.deployment_history = []
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = datetime.now().isoformat()


class ModelRegistry:
    """Enterprise-grade model registry with SQLite backend"""
    
    def __init__(self, registry_path: str = None):
        self.registry_path = registry_path or os.path.expanduser("~/.quickserveml/registry")
        self.db_path = os.path.join(self.registry_path, "registry.db")
        self.models_path = os.path.join(self.registry_path, "models")
        
        # Ensure directories exist
        os.makedirs(self.registry_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with schema"""
        with self._get_db_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    description TEXT,
                    tags TEXT,
                    author TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    status TEXT,
                    model_path TEXT,
                    model_hash TEXT,
                    file_size INTEGER,
                    framework TEXT,
                    accuracy REAL,
                    latency_ms REAL,
                    throughput_rps REAL,
                    input_shape TEXT,
                    output_shape TEXT,
                    dependencies TEXT,
                    deployment_history TEXT,
                    notes TEXT,
                    UNIQUE(name, version)
                )
            """)
            conn.commit()
    
    @contextmanager
    def _get_db_connection(self):
        """Database connection context manager"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
        finally:
            conn.close()
    
    def _calculate_model_hash(self, model_path: str) -> str:
        """Calculate SHA256 hash of model file"""
        hash_sha256 = hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def add_model(self, name: str, model_path: str, version: str = None, 
                  description: str = "", tags: List[str] = None, 
                  author: str = "", **kwargs) -> ModelMetadata:
        """Add a new model to the registry"""
        formatter = get_formatter()
        
        # Validate model file
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Generate version if not provided
        if version is None:
            version = self._generate_version(name)
        
        # Calculate model hash
        model_hash = self._calculate_model_hash(model_path)
        file_size = os.path.getsize(model_path)
        
        # Check if model already exists
        if self._model_exists(name, version):
            raise ValueError(f"Model {name}:{version} already exists in registry")
        
        # Copy model to registry storage
        registry_model_path = os.path.join(self.models_path, f"{name}_{version}.onnx")
        shutil.copy2(model_path, registry_model_path)
        
        # Create metadata
        metadata = ModelMetadata(
            name=name,
            version=version,
            description=description,
            tags=tags or [],
            author=author,
            model_path=registry_model_path,
            model_hash=model_hash,
            file_size=file_size,
            **kwargs
        )
        
        # Save to database
        self._save_metadata(metadata)
        
        formatter.success(f"✅ Model '{name}:{version}' added to registry")
        formatter.info(f"📁 Stored at: {registry_model_path}")
        formatter.info(f"🔍 Hash: {model_hash[:16]}...")
        
        return metadata
    
    def _generate_version(self, name: str) -> str:
        """Generate next version number for model"""
        versions = self.list_versions(name)
        if not versions:
            return "v1.0.0"
        
        # Extract version numbers and find the highest
        version_numbers = []
        for version in versions:
            if version.startswith('v'):
                try:
                    parts = version[1:].split('.')
                    version_numbers.append([int(p) for p in parts])
                except ValueError:
                    continue
        
        if not version_numbers:
            return "v1.0.0"
        
        # Increment the highest version
        latest = max(version_numbers)
        latest[2] += 1  # Increment patch version
        return f"v{'.'.join(map(str, latest))}"
    
    def _model_exists(self, name: str, version: str) -> bool:
        """Check if model exists in registry"""
        with self._get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM models WHERE name = ? AND version = ?",
                (name, version)
            )
            return cursor.fetchone()[0] > 0
    
    def _save_metadata(self, metadata: ModelMetadata):
        """Save model metadata to database"""
        with self._get_db_connection() as conn:
            conn.execute("""
                INSERT INTO models (
                    name, version, description, tags, author, created_at, updated_at,
                    status, model_path, model_hash, file_size, framework, accuracy,
                    latency_ms, throughput_rps, input_shape, output_shape,
                    dependencies, deployment_history, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.name, metadata.version, metadata.description,
                json.dumps(metadata.tags), metadata.author, metadata.created_at,
                metadata.updated_at, metadata.status, metadata.model_path,
                metadata.model_hash, metadata.file_size, metadata.framework,
                metadata.accuracy, metadata.latency_ms, metadata.throughput_rps,
                json.dumps(metadata.input_shape), json.dumps(metadata.output_shape),
                json.dumps(metadata.dependencies), json.dumps(metadata.deployment_history),
                metadata.notes
            ))
            conn.commit()
    
    def list_models(self, status: str = None, tags: List[str] = None) -> List[ModelMetadata]:
        """List all models in registry with optional filtering"""
        query = "SELECT * FROM models"
        params = []
        
        conditions = []
        if status:
            conditions.append("status = ?")
            params.append(status)
        if tags:
            for tag in tags:
                conditions.append("tags LIKE ?")
                params.append(f"%{tag}%")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY name, version"
        
        with self._get_db_connection() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_metadata(row) for row in cursor.fetchall()]
    
    def list_versions(self, name: str) -> List[str]:
        """List all versions of a specific model"""
        with self._get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT version FROM models WHERE name = ? ORDER BY version",
                (name,)
            )
            return [row[0] for row in cursor.fetchall()]
    
    def get_model(self, name: str, version: str = None) -> Optional[ModelMetadata]:
        """Get model metadata by name and version"""
        if version is None:
            # Get latest version
            with self._get_db_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM models WHERE name = ? ORDER BY version DESC LIMIT 1",
                    (name,)
                )
                row = cursor.fetchone()
        else:
            with self._get_db_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM models WHERE name = ? AND version = ?",
                    (name, version)
                )
                row = cursor.fetchone()
        
        return self._row_to_metadata(row) if row else None
    
    def _row_to_metadata(self, row) -> ModelMetadata:
        """Convert database row to ModelMetadata"""
        return ModelMetadata(
            name=row['name'],
            version=row['version'],
            description=row['description'] or "",
            tags=json.loads(row['tags']) if row['tags'] else [],
            author=row['author'] or "",
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            status=row['status'],
            model_path=row['model_path'],
            model_hash=row['model_hash'],
            file_size=row['file_size'],
            framework=row['framework'],
            accuracy=row['accuracy'],
            latency_ms=row['latency_ms'],
            throughput_rps=row['throughput_rps'],
            input_shape=json.loads(row['input_shape']) if row['input_shape'] else None,
            output_shape=json.loads(row['output_shape']) if row['output_shape'] else None,
            dependencies=json.loads(row['dependencies']) if row['dependencies'] else {},
            deployment_history=json.loads(row['deployment_history']) if row['deployment_history'] else [],
            notes=row['notes'] or ""
        )
    
    def update_model(self, name: str, version: str, **kwargs) -> ModelMetadata:
        """Update model metadata"""
        metadata = self.get_model(name, version)
        if not metadata:
            raise ValueError(f"Model {name}:{version} not found")
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        
        metadata.updated_at = datetime.now().isoformat()
        
        # Save to database
        with self._get_db_connection() as conn:
            conn.execute("""
                UPDATE models SET
                    description = ?, tags = ?, author = ?, updated_at = ?,
                    status = ?, accuracy = ?, latency_ms = ?, throughput_rps = ?,
                    input_shape = ?, output_shape = ?, dependencies = ?,
                    deployment_history = ?, notes = ?
                WHERE name = ? AND version = ?
            """, (
                metadata.description, json.dumps(metadata.tags), metadata.author,
                metadata.updated_at, metadata.status, metadata.accuracy,
                metadata.latency_ms, metadata.throughput_rps,
                json.dumps(metadata.input_shape), json.dumps(metadata.output_shape),
                json.dumps(metadata.dependencies), json.dumps(metadata.deployment_history),
                metadata.notes, name, version
            ))
            conn.commit()
        
        return metadata
    
    def delete_model(self, name: str, version: str):
        """Delete model from registry"""
        metadata = self.get_model(name, version)
        if not metadata:
            raise ValueError(f"Model {name}:{version} not found")
        
        # Delete model file
        if os.path.exists(metadata.model_path):
            os.remove(metadata.model_path)
        
        # Delete from database
        with self._get_db_connection() as conn:
            conn.execute(
                "DELETE FROM models WHERE name = ? AND version = ?",
                (name, version)
            )
            conn.commit()
    
    def compare_models(self, name: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two versions of a model, including all available metrics dynamically."""
        model1 = self.get_model(name, version1)
        model2 = self.get_model(name, version2)
        
        if not model1 or not model2:
            raise ValueError("One or both models not found")
        
        comparison = {
            "model_name": name,
            "version1": version1,
            "version2": version2,
            "comparison_date": datetime.now().isoformat(),
            "metrics": {}
        }
        
        # Gather all metrics from both models (attributes and metrics dict)
        def extract_metrics(model):
            metrics = {}
            # Add standard attributes if present
            for attr in ["accuracy", "latency_ms", "throughput_rps", "file_size"]:
                val = getattr(model, attr, None)
                if val is not None:
                    metrics[attr] = val
            # Add metrics from metrics dict if present
            if hasattr(model, "metrics") and isinstance(model.metrics, dict):
                for k, v in model.metrics.items():
                    metrics[k] = v
            return metrics
        
        m1_metrics = extract_metrics(model1)
        m2_metrics = extract_metrics(model2)
        all_keys = set(m1_metrics.keys()) | set(m2_metrics.keys())
        
        for key in sorted(all_keys):
            v1 = m1_metrics.get(key)
            v2 = m2_metrics.get(key)
            diff = None
            if v1 is not None and v2 is not None:
                try:
                    diff = v2 - v1 if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) else None
                except Exception:
                    diff = None
            comparison["metrics"][key] = {
                "version1": v1,
                "version2": v2,
                "difference": diff
            }
        
        return comparison
    
    def export_model(self, name: str, version: str, output_path: str):
        """Export model from registry to specified path"""
        metadata = self.get_model(name, version)
        if not metadata:
            raise ValueError(f"Model {name}:{version} not found")
        
        shutil.copy2(metadata.model_path, output_path)
        get_formatter().success(f"✅ Model '{name}:{version}' exported to {output_path}")
    
    def get_model_path(self, name: str, version: str = None) -> str:
        """Get the file path of a model in the registry"""
        metadata = self.get_model(name, version)
        if not metadata:
            raise ValueError(f"Model {name}:{version} not found")
        return metadata.model_path 