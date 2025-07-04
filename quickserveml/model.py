from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime
import sqlite3
import json
import os

@dataclass
class ModelMetadata:
    name: str
    version: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    author: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "draft"  # dev, staging, prod, archived
    artifact_path: str = ""
    hash: str = ""
    file_size: int = 0
    framework: str = "onnx"
    opset_version: str = ""
    dependencies: Dict[str, str] = field(default_factory=dict)
    training_params: Dict[str, Any] = field(default_factory=dict)
    dataset_version: str = ""
    experiment_id: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)  # accuracy, F1, latency, etc.
    explainability: Dict[str, Any] = field(default_factory=dict)  # SHAP, feature importances, etc.
    approval_history: List[Dict] = field(default_factory=list)
    changelog: List[str] = field(default_factory=list)

class ModelRegistry:
    def __init__(self, db_path=None):
        self.db_path = db_path or os.path.expanduser("~/.quickserveml/registry/models.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_database()

    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    name TEXT,
                    version TEXT,
                    description TEXT,
                    tags TEXT,
                    author TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    status TEXT,
                    artifact_path TEXT,
                    hash TEXT,
                    file_size INTEGER,
                    framework TEXT,
                    opset_version TEXT,
                    dependencies TEXT,
                    training_params TEXT,
                    dataset_version TEXT,
                    experiment_id TEXT,
                    metrics TEXT,
                    explainability TEXT,
                    approval_history TEXT,
                    changelog TEXT,
                    PRIMARY KEY (name, version)
                )
            ''')
            conn.commit()

    def add_model(self, metadata: ModelMetadata):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO models (name, version, description, tags, author, created_at, updated_at, status, artifact_path, hash, file_size, framework, opset_version, dependencies, training_params, dataset_version, experiment_id, metrics, explainability, approval_history, changelog)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.name, metadata.version, metadata.description, json.dumps(metadata.tags), metadata.author, metadata.created_at, metadata.updated_at, metadata.status,
                metadata.artifact_path, metadata.hash, metadata.file_size, metadata.framework, metadata.opset_version, json.dumps(metadata.dependencies), json.dumps(metadata.training_params),
                metadata.dataset_version, metadata.experiment_id, json.dumps(metadata.metrics), json.dumps(metadata.explainability), json.dumps(metadata.approval_history), json.dumps(metadata.changelog)
            ))
            conn.commit()

    def get_model(self, name: str, version: str = None) -> ModelMetadata:
        with sqlite3.connect(self.db_path) as conn:
            if version is None:
                cursor = conn.execute('SELECT * FROM models WHERE name = ? ORDER BY version DESC LIMIT 1', (name,))
            else:
                cursor = conn.execute('SELECT * FROM models WHERE name = ? AND version = ?', (name, version))
            row = cursor.fetchone()
            if not row:
                return None
            return ModelMetadata(
                name=row[0], version=row[1], description=row[2], tags=json.loads(row[3]), author=row[4], created_at=row[5], updated_at=row[6], status=row[7],
                artifact_path=row[8], hash=row[9], file_size=row[10], framework=row[11], opset_version=row[12], dependencies=json.loads(row[13]),
                training_params=json.loads(row[14]), dataset_version=row[15], experiment_id=row[16], metrics=json.loads(row[17]), explainability=json.loads(row[18]),
                approval_history=json.loads(row[19]), changelog=json.loads(row[20])
            )

    def list_models(self) -> List[ModelMetadata]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM models')
            return [
                ModelMetadata(
                    name=row[0], version=row[1], description=row[2], tags=json.loads(row[3]), author=row[4], created_at=row[5], updated_at=row[6], status=row[7],
                    artifact_path=row[8], hash=row[9], file_size=row[10], framework=row[11], opset_version=row[12], dependencies=json.loads(row[13]),
                    training_params=json.loads(row[14]), dataset_version=row[15], experiment_id=row[16], metrics=json.loads(row[17]), explainability=json.loads(row[18]),
                    approval_history=json.loads(row[19]), changelog=json.loads(row[20])
                ) for row in cursor.fetchall()
            ] 