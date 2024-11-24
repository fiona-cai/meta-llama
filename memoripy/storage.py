# storage.py

class BaseStorage:
    def load_history(self):
        raise NotImplementedError("The method load_history() must be implemented.")

    def save_memory_to_history(self, memory_store):
        raise NotImplementedError("The method save_memory_to_history() must be implemented.")

    def save_core_memory(self, core_memory):
        """Save core memory to storage."""
        raise NotImplementedError("The method save_core_memory() must be implemented.")

    def load_core_memory(self):
        """Load core memory from storage."""
        raise NotImplementedError("The method load_core_memory() must be implemented.")
