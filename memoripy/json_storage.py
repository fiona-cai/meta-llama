# json_storage.py

import json
import os

from .storage import BaseStorage

class JSONStorage(BaseStorage):
    def __init__(self, file_path="interaction_history.json"):
        self.file_path = file_path

    def load_history(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                print("[Storage] Loading existing interaction history from JSON...")
                try:
                    history = json.load(f)
                    # Return the entire history dict
                    return history
                except json.JSONDecodeError:
                    print("[Storage] Error: Invalid JSON format in history file")
                    return None
        print("[Storage] No existing interaction history found.")
        return None

    def save_memory_to_history(self, memory_store):
        try:
            history = {
                "core_memory": memory_store.core_memory.copy(),  # Make a copy to avoid reference issues
                "short_term_memory": [],
                "long_term_memory": []
            }

            # Save short-term memory interactions
            for idx in range(len(memory_store.short_term_memory)):
                interaction = {
                    'id': memory_store.short_term_memory[idx]['id'],
                    'prompt': memory_store.short_term_memory[idx]['prompt'],
                    'output': memory_store.short_term_memory[idx]['output'],
                    'embedding': memory_store.embeddings[idx].flatten().tolist(),
                    'timestamp': memory_store.timestamps[idx],
                    'access_count': memory_store.access_counts[idx],
                    'concepts': list(memory_store.concepts_list[idx]),
                    'decay_factor': memory_store.short_term_memory[idx].get('decay_factor', 1.0)
                }
                history["short_term_memory"].append(interaction)

            # Save long-term memory interactions
            history["long_term_memory"].extend(memory_store.long_term_memory)

            # Save the history to a file atomically
            temp_file = f"{self.file_path}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(history, f, indent=4)
            os.replace(temp_file, self.file_path)  # Atomic operation
            
            print(f"[Storage] Saved memory state. Core: {history['core_memory']}")
            print(f"[Storage] Memory size - Short-term: {len(history['short_term_memory'])}, Long-term: {len(history['long_term_memory'])}")
            
        except Exception as e:
            print(f"[Storage] Error saving memory state: {str(e)}")
