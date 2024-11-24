# in_memory_storage.py

from .storage import BaseStorage

class InMemoryStorage(BaseStorage):
    def __init__(self):
        self.history = {
            "short_term_memory": [],
            "long_term_memory": [],
            "core_memory": []
        }

    def load_history(self):
        print("Loading history from in-memory storage.")
        return (
            self.history.get("short_term_memory", []),
            self.history.get("long_term_memory", []),
            self.history.get("core_memory", [])
        )

    def save_memory_to_history(self, memory_store):
        print("Saving history to in-memory storage.")
        self.history = {
            "short_term_memory": [],
            "long_term_memory": [],
            "core_memory": [],
            "core_memory": []
        }

        # Save short-term memory interactions
        for idx in range(len(memory_store.short_term_memory)):
            history_entry = {
                'id': memory_store.short_term_memory[idx]['id'],
                'messages': memory_store.short_term_memory[idx]['messages'],
                'embedding': memory_store.embeddings[idx].flatten().tolist(),
                'timestamp': memory_store.timestamps[idx],
                'access_count': memory_store.access_counts[idx],
                'concepts': list(memory_store.concepts_list[idx]),
                'decay_factor': memory_store.short_term_memory[idx].get('decay_factor', 1.0)
            }
            self.history["short_term_memory"].append(history_entry)

        # Save long-term memory interactions
        for idx in range(len(memory_store.long_term_memory)):
            history_entry = {
                'id': memory_store.long_term_memory[idx]['id'],
                'messages': memory_store.long_term_memory[idx]['messages'],
                'embedding': memory_store.long_term_embeddings[idx].flatten().tolist(),
                'timestamp': memory_store.long_term_timestamps[idx],
                'access_count': memory_store.long_term_access_counts[idx],
                'concepts': list(memory_store.long_term_concepts[idx]),
                'decay_factor': memory_store.long_term_memory[idx].get('decay_factor', 1.0)
            }
            self.history["long_term_memory"].append(history_entry)

        # Save core memory interactions
        for interaction in memory_store.core_memory:
            self.history["core_memory"].append(interaction)
