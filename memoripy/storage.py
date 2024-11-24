# storage.py

class BaseStorage:
    def load_history(self):
        """Should return (short_term, long_term, core_memory)"""
        raise NotImplementedError("The method load_history() must be implemented.")

    def save_memory_to_history(self, memory_store):
        """Should save short_term, long_term, and core_memory"""
        raise NotImplementedError("The method save_memory_to_history() must be implemented.")
