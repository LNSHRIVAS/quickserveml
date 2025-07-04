from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime
import sqlite3
import json
import os

@dataclass
class DatasetMetadata:
    name: str
    version: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    author: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    schema: Dict[str, Any] = field(default_factory=dict)
    profiling: Dict[str, Any] = field(default_factory=dict)
    storage_path: str = ""
    hash: str = ""
    file_size: int = 0
    linked_models: List[str] = field(default_factory=list)

class DatasetRegistry:
    def __init__(self, db_path=None):
        self.db_path = db_path or os.path.expanduser("~/.quickserveml/registry/datasets.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_database()

    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS datasets (
                    name TEXT,
                    version TEXT,
                    description TEXT,
                    tags TEXT,
                    author TEXT,
                    created_at TEXT,
                    schema TEXT,
                    profiling TEXT,
                    storage_path TEXT,
                    hash TEXT,
                    file_size INTEGER,
                    linked_models TEXT,
                    PRIMARY KEY (name, version)
                )
            ''')
            conn.commit()

    def add_dataset(self, metadata: DatasetMetadata):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO datasets (name, version, description, tags, author, created_at, schema, profiling, storage_path, hash, file_size, linked_models)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.name, metadata.version, metadata.description, json.dumps(metadata.tags), metadata.author, metadata.created_at,
                json.dumps(metadata.schema), json.dumps(metadata.profiling), metadata.storage_path, metadata.hash, metadata.file_size, json.dumps(metadata.linked_models)
            ))
            conn.commit()

    def get_dataset(self, name: str, version: str = None) -> DatasetMetadata:
        with sqlite3.connect(self.db_path) as conn:
            if version is None:
                cursor = conn.execute('SELECT * FROM datasets WHERE name = ? ORDER BY version DESC LIMIT 1', (name,))
            else:
                cursor = conn.execute('SELECT * FROM datasets WHERE name = ? AND version = ?', (name, version))
            row = cursor.fetchone()
            if not row:
                return None
            return DatasetMetadata(
                name=row[0], version=row[1], description=row[2], tags=json.loads(row[3]), author=row[4], created_at=row[5],
                schema=json.loads(row[6]), profiling=json.loads(row[7]), storage_path=row[8], hash=row[9], file_size=row[10], linked_models=json.loads(row[11])
            )

    def list_datasets(self) -> List[DatasetMetadata]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM datasets')
            return [
                DatasetMetadata(
                    name=row[0], version=row[1], description=row[2], tags=json.loads(row[3]), author=row[4], created_at=row[5],
                    schema=json.loads(row[6]), profiling=json.loads(row[7]), storage_path=row[8], hash=row[9], file_size=row[10], linked_models=json.loads(row[11])
                ) for row in cursor.fetchall()
            ] 