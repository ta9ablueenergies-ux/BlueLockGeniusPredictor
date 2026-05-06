"""
Sequence Memory Module for Temporal Pattern Storage
"""
import torch
import torch.nn as nn
import json
import os

class SequenceMemory:
    def __init__(self, memory_path="model/v11_sequence_memory.json"):
        self.memory_path = memory_path
        self.memory = self._load_memory()

    def _load_memory(self):
        """Load existing sequence memory or initialize new"""
        if os.path.exists(self.memory_path):
            with open(self.memory_path, 'r') as f:
                return json.load(f)
        return {}

    def store_sequence(self, team_id, sequence_data, form_patterns):
        """Store sequence data for a team"""
        if team_id not in self.memory:
            self.memory[team_id] = []

        self.memory[team_id].append({
            'sequence': sequence_data,
            'patterns': form_patterns,
            'timestamp': torch.datetime.now().isoformat()
        })

        # Keep only last 100 sequences per team
        if len(self.memory[team_id]) > 100:
            self.memory[team_id] = self.memory[team_id][-100:]

        self._save_memory()

    def get_team_memory(self, team_id):
        """Retrieve sequence memory for a team"""
        return self.memory.get(team_id, [])

    def _save_memory(self):
        """Save memory to file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)

        with open(self.memory_path, 'w') as f:
            json.dump(self.memory, f, indent=2)

    def update_form_patterns(self, team_id, patterns):
        """Update form patterns for a team"""
        if team_id in self.memory:
            # Update the latest entry with new patterns
            if self.memory[team_id]:
                self.memory[team_id][-1]['patterns'] = patterns
            self._save_memory()