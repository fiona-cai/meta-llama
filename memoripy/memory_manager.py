import numpy as np
import json
import time
import uuid
import ollama
from typing import Optional, Dict, Any
from langchain_groq import ChatGroq  # Remove GroqEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .memory_store import MemoryStore
from .storage import BaseStorage
from .in_memory_storage import InMemoryStorage

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage


class ConceptExtractionResponse(BaseModel):
    concepts: list[str] = Field(description="List of key concepts extracted from the text.")


class MemoryManager:
    """
    Manages the memory store, including loading and saving history,
    adding interactions, retrieving relevant interactions, and generating responses.
    """

    def __init__(self, api_key, chat_model="ollama", chat_model_name="llama3.1:8b", embedding_model="ollama", embedding_model_name="mxbai-embed-large", storage=None, verbose=True):
        self.api_key = api_key
        self.chat_model_name = chat_model_name
        self.embedding_model_name = embedding_model_name

        # Set chat model
        if chat_model.lower() == "openai":
            self.llm = ChatOpenAI(model=chat_model_name, api_key=self.api_key)
        elif chat_model.lower() == "ollama":
            self.llm = ChatOllama(model=chat_model_name, temperature=0)
        elif chat_model.lower() == "groq":
            self.llm = ChatGroq(model=chat_model_name, api_key=self.api_key)
        else:
            raise ValueError("Unsupported chat model. Choose either 'openai', 'ollama', or 'groq'.")

        # Set embedding model and dimension
        if embedding_model.lower() == "groq":
            if embedding_model_name == "mxbai-embed-large":
                self.dimension = 1024
            else:
                self.dimension = self.initialize_embedding_dimension()
            self.embeddings_model = lambda text: ollama.embeddings(model=self.embedding_model_name, prompt=text)["embedding"]
        
        elif embedding_model.lower() == "openai":
            if embedding_model_name == "text-embedding-3-small":
                self.dimension = 1536
            else:
                raise ValueError("Unsupported OpenAI embedding model name for specified dimension.")
            self.embeddings_model = OpenAIEmbeddings(model=embedding_model_name, api_key=self.api_key)
        elif embedding_model.lower() == "ollama":
            if embedding_model_name == "mxbai-embed-large":
                self.dimension = 1024
            else:
                self.dimension = self.initialize_embedding_dimension()
            self.embeddings_model = lambda text: ollama.embeddings(model=self.embedding_model_name, prompt=text)["embedding"]
        else:
            raise ValueError("Unsupported embedding model. Choose either 'openai', 'ollama', or 'groq'.")

        # Initialize memory store with the correct dimension
        self.memory_store = MemoryStore(dimension=self.dimension)

        if storage is None:
            self.storage = InMemoryStorage()
        else:
            self.storage = storage

        # Core memory update patterns
        self.core_memory_patterns = {
            "personality": [
                # Direct requests
                "I want you to be more",
                "please act more",
                "change your personality to",
                "behave more",
                "your personality should be",
                "be more",
                # Behavioral requests
                "can you be more",
                "could you act more",
                "try to be more",
                "start being more",
                # Action modifiers
                "from now on be",
                "be a bit more",
                "act a little more",
                # Personality traits
                "make yourself more",
                "adopt a more",
                "take on a more",
                # Style changes
                "change your style to be",
                "modify your approach to be",
                "adjust your behavior to be",
                # Tone adjustments
                "use a more",
                "speak in a more",
                "respond in a more",
                # Direct commands
                "just be",
                "become more",
                "switch to being",
                # Also forms
                "also be",
                "additionally be",
                "and be",
                # State transitions
                "from now on you should be",
                "I'd like you to be",
                "you need to be"
            ],
            "user_name": [
                "my name is",
                "call me",
                "I am called",
                "you can call me"
            ],
            "assistant_name": [
                "your name should be",
                "I want to call you",
                "change your name to",
                "you will be called",
                "call yourself"
            ]
        }
        
        # Add caching properties
        self._embedding_cache = {}
        self._pattern_embeddings = {}
        self._pattern_embeddings_initialized = False
        
        # Add caching properties
        self._core_memory_cache = None
        self._short_term_cache = []
        self._long_term_cache = []
        
        # Initialize memory caches with same reference
        self._memory_cache = {
            "core_memory": {
                "user_name": "User",
                "assistant_name": "Assistant",
                "personality": "Helpful and friendly"
            },
            "short_term_memory": [],
            "long_term_memory": []
        }
        self.memory_store.core_memory = self._memory_cache["core_memory"]  # Share reference
        self.verbose = verbose
        self.initialize_memory()

        # Add thresholds for different memory types
        self.memory_type_thresholds = {
            "personality": 60,  # Lower threshold for personality
            "user_name": 70,    # Default threshold
            "assistant_name": 70 # Default threshold
        }

    def initialize_embedding_dimension(self):
        """
        Retrieve embedding dimension from Ollama by generating a test embedding.
        """
        print("Determining embedding dimension for Ollama model...")
        test_text = "Test to determine embedding dimension"
        response = ollama.embeddings(
            model=self.embedding_model_name,
            prompt=test_text
        )
        embedding = response.get("embedding")
        if embedding is None:
            raise ValueError("Failed to retrieve embedding for dimension initialization.")
        return len(embedding)

    def initialize_groq_embedding_dimension(self):
        """
        Retrieve embedding dimension from Groq by generating a test embedding.
        """
        print("Determining embedding dimension for Groq model...")
        test_text = "Test to determine embedding dimension"
        embedding = self.embeddings_model.embed_text(test_text)
        if embedding is None:
            raise ValueError("Failed to retrieve embedding for dimension initialization.")
        return len(embedding)

    def standardize_embedding(self, embedding):
        """
        Standardize embedding to the target dimension by padding with zeros or truncating.
        """
        current_dim = len(embedding)
        if current_dim == self.dimension:
            return embedding
        elif current_dim < self.dimension:
            # Pad with zeros
            return np.pad(embedding, (0, self.dimension - current_dim), 'constant')
        else:
            # Truncate to match target dimension
            return embedding[:self.dimension]

    def load_history(self):
        """Load history from storage only if cache is empty"""
        try:
            if not self._memory_cache["short_term_memory"]:
                print(f"[Memory] Loading history from storage...")
                history_data = self.storage.load_history()
                
                if history_data and isinstance(history_data, dict):
                    print("[Memory] Successfully loaded history data")
                    self._memory_cache = history_data
                    self.memory_store.core_memory = history_data.get("core_memory", {
                        "user_name": "User",
                        "assistant_name": "Assistant",
                        "personality": "Helpful and friendly"
                    })
                else:
                    print("[Memory] Initializing with default values")
                    self._memory_cache = {
                        "core_memory": {
                            "user_name": "User",
                            "assistant_name": "Assistant",
                            "personality": "Helpful and friendly"
                        },
                        "short_term_memory": [],
                        "long_term_memory": []
                    }
                    self.memory_store.core_memory = self._memory_cache["core_memory"]

            return (
                self._memory_cache["short_term_memory"],
                self._memory_cache["long_term_memory"]
            )
        except Exception as e:
            print(f"[Memory] Error loading history: {str(e)}")
            return [], []

    def save_memory_to_history(self):
        """Save current memory state to storage"""
        try:
            print("[Memory] Persisting memory state to storage...")
            # Ensure core memory is synced before saving
            self._memory_cache["core_memory"] = self.memory_store.core_memory
            self.storage.save_memory_to_history(self.memory_store)
        except Exception as e:
            print(f"[Memory] Error saving memory: {str(e)}")

    def check_pattern_similarity(self, query_embedding, memory_type):
        """Check similarity with lazy pattern embedding initialization"""
        if not self._pattern_embeddings_initialized:
            self._initialize_pattern_embeddings()
            
        max_similarity = 0
        for pattern_embedding in self._pattern_embeddings[memory_type]:
            similarity = np.dot(query_embedding.flatten(), pattern_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(pattern_embedding)
            )
            max_similarity = max(max_similarity, similarity * 100)
        return max_similarity

    def validate_core_memory_update(self, text: str, similarity_threshold: float = 70) -> tuple[Optional[str], Optional[str]]:
        """Validate core memory updates using similarity comparison before LLM."""
        query_embedding = self.get_embedding(text)
        
        # Check similarity with patterns for each memory type
        highest_similarity = 0
        matched_memory_type = None
        
        print(f"[Memory] Checking for core memory update patterns...")
        for memory_type in self.core_memory_patterns.keys():
            similarity = self.check_pattern_similarity(query_embedding, memory_type)
            if self.verbose:
                print(f"[Memory] Pattern match for {memory_type}: {similarity:.2f}%")
                
            # Use memory type specific threshold
            type_threshold = self.memory_type_thresholds.get(memory_type, similarity_threshold)
            if similarity > highest_similarity and similarity >= type_threshold:
                highest_similarity = similarity
                matched_memory_type = memory_type
        
        # If no pattern matches with high similarity, return None
        if matched_memory_type is None:
            if self.verbose:
                print(f"[Memory] No core memory patterns matched above threshold ({similarity_threshold}%)")
            return None, None
            
        print(f"[Memory] Detected potential {matched_memory_type} update (similarity: {highest_similarity:.2f}%)")
        
        # If we have a match, use LLM to extract the new value
        if matched_memory_type == "personality":
            prompt = f"""Analyze if the following message contains a request to update personality, speech pattern, or communication style.
            Message: "{text}"
            
            Think about:
            1. Could this be a request to update or change personality traits, speech patterns, or communication style?
            2. Consider both explicit requests and implicit suggestions about how to behave, speak, or communicate
            3. If there is a slight possibility that the user wants to change the assistant's behavior, consider it as a valid request.
            4. What specific personality traits should be extracted?

            Explain your reasoning, then provide your conclusion in JSON format with the personality traits from step 4:
            {{"update": true/false, "value": "requested_personality_trait,requested_personality_trait,requested_personality_trait"}}
            """
        else:
            prompt = f"""Analyze if the following message contains a request to update {matched_memory_type}.
            Message: "{text}"
            
            Think about:
            1. Could this be a request to update {matched_memory_type}?
            2. Consider the request may be indirect, implicit or conversational. It will still be a valid request.
            3. What specific value should be extracted?
    
            Explain your reasoning, then provide your conclusion in JSON format:
            {{"update": true/false, "value": "extracted_value"}}
            """
        
        print(f"[{self.chat_model_name}] Validating core memory update...")
        messages = [
            SystemMessage(content="You're a helpful assistant that validates memory updates."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        if self.verbose:
            print(f"[{self.chat_model_name}] Validation response:\n{response.content}\n")
            
        parsed_data = self._extract_last_json(response.content)
        
        if parsed_data:
            should_update = parsed_data.get("update", False)
            new_value = parsed_data.get("value")
            
            if should_update and new_value:
                print(f"[Memory] Confirmed update for {matched_memory_type}: {new_value}")
                return matched_memory_type, new_value
            else:
                print(f"[Memory] Update rejected for {matched_memory_type}")
        else:
            print("[Memory] Failed to parse validation response")
            
        return None, None

    def add_interaction(self, prompt, output, embedding, concepts):
        # Check for core memory updates
        memory_type, new_value = self.validate_core_memory_update(prompt)
        if (memory_type and new_value):
            # Update both memory store and cache since they share reference
            self.memory_store.core_memory[memory_type] = new_value
            # Force save after core memory update
            self.save_memory_to_history()
            return
        
        timestamp = time.time()
        interaction_id = str(uuid.uuid4())
        interaction = {
            "id": interaction_id,
            "prompt": prompt,
            "output": output,
            "embedding": embedding.tolist(),
            "timestamp": timestamp,
            "access_count": 1,
            "concepts": list(concepts),
            "decay_factor": 1.0,
        }
        # Update both memory store and cache
        self.memory_store.add_interaction(interaction)
        self._short_term_cache.append(interaction)
        self.save_memory_to_history()  # Save after adding new interaction

    def get_embedding(self, text):
        """Get embedding with caching"""
        cache_key = hash(text)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
            
        print(f"[{self.embedding_model_name}] Generating embedding for text...")
        if callable(self.embeddings_model):
            embedding = self.embeddings_model(text)
        else:
            embedding = self.embeddings_model.embed_query(text)
            
        if embedding is None:
            raise ValueError("Failed to generate embedding.")
            
        if isinstance(embedding, list):
            embedding_array = np.array(embedding)
        else:
            embedding_array = embedding
            
        standardized_embedding = self.standardize_embedding(embedding_array)
        result = standardized_embedding.reshape(1, -1)
        self._embedding_cache[cache_key] = result
        return result

    def _initialize_pattern_embeddings(self):
        """Lazy initialization of pattern embeddings"""
        if not self._pattern_embeddings_initialized:
            print("[Memory] Initializing pattern embeddings...")
            for memory_type, patterns in self.core_memory_patterns.items():
                self._pattern_embeddings[memory_type] = [
                    self.get_embedding(pattern).flatten()
                    for pattern in patterns
                ]
            self._pattern_embeddings_initialized = True

    def _extract_last_json(self, response: str, verbose: bool = True) -> Optional[Dict[str, Any]]:
        """Extract the last valid JSON object from a text that may contain both text and JSON."""
        brace_count = 0
        start_index = -1
        
        for i in range(len(response) - 1, -1, -1):
            if response[i] == '}':
                brace_count += 1
                if start_index == -1:
                    start_index = i
            elif response[i] == '{':
                brace_count -= 1
                if brace_count == 0 and start_index != -1:
                    json_str = response[i:start_index + 1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        continue
        return None

    def extract_concepts(self, text):
        print(f"[{self.chat_model_name}] Extracting key concepts from the provided text...")

        # Updated prompt template to encourage JSON output within natural response
        base_prompt = """Analyze the following text and extract key concepts.
        You can explain your thinking, but make sure to include a JSON object in your response with the format:
        {{"concepts": ["concept1", "concept2", ...]}}
        
        Text to analyze: {text}"""

        messages = [
            SystemMessage(content="You're a helpful assistant that provides analysis with JSON output."),
            HumanMessage(content=base_prompt.format(text=text))
        ]
        
        response = self.llm.invoke(messages)
        response_text = response.content.strip()
        
        # Parse JSON from response
        parsed_data = self._extract_last_json(response_text)
        if (parsed_data and "concepts" in parsed_data):
            concepts = parsed_data["concepts"]
            print(f"Concepts extracted: {concepts}")
            return concepts
        
        # Fallback if no valid JSON found
        print("Warning: Could not parse JSON from response. Using empty concepts list.")
        return []

    def initialize_memory(self):
        """Initialize memory store with history data"""
        try:
            print("[Memory] Initializing memory system...")
            history_data = self.storage.load_history()
            
            if not history_data:
                print("[Memory] No existing history found, using defaults")
                return
                
            if isinstance(history_data, dict):
                print("[Memory] Loading existing history data")
                self._memory_cache = history_data
                self.memory_store.core_memory = history_data.get("core_memory", {
                    "user_name": "User",
                    "assistant_name": "Assistant",
                    "personality": "Helpful and friendly"
                })
                
                # Load interactions into memory store
                for interaction in history_data.get("short_term_memory", []):
                    if "embedding" in interaction:
                        interaction['embedding'] = self.standardize_embedding(
                            np.array(interaction['embedding'])
                        )
                        self.memory_store.add_interaction(interaction)
                
                # Load long-term memory
                self.memory_store.long_term_memory.extend(
                    history_data.get("long_term_memory", [])
                )
                
                print(f"[Memory] Core memory loaded: {self.memory_store.core_memory}")
                self.memory_store.cluster_interactions()
                print(f"[Memory] Initialized with {len(self.memory_store.short_term_memory)} short-term and {len(self.memory_store.long_term_memory)} long-term memories")
            else:
                print("[Memory] Warning: Invalid history format, using defaults")
                
        except Exception as e:
            print(f"[Memory] Error during initialization: {str(e)}")
            self._memory_cache = {
                "core_memory": {
                    "user_name": "User",
                    "assistant_name": "Assistant",
                    "personality": "Helpful and friendly"
                },
                "short_term_memory": [],
                "long_term_memory": []
            }
            self.memory_store.core_memory = self._memory_cache["core_memory"]

    def retrieve_relevant_interactions(self, query, similarity_threshold=40, exclude_last_n=0):
        query_embedding = self.get_embedding(query)
        # No longer extract concepts here - will be done after response generation
        return self.memory_store.retrieve(query_embedding, [], similarity_threshold, exclude_last_n=exclude_last_n)

    def generate_response(self, prompt, last_interactions, retrievals, context_window=3):
        print(f"[{self.chat_model_name}] Generating response...")
        # Build system message with core memories
        personality = self.memory_store.get_core_memory('personality')
        system_content = (
            f"You're a helpful assistant named {self.memory_store.get_core_memory('assistant_name')}. "
            f"You're talking to {self.memory_store.get_core_memory('user_name')}. "
            f"Your personality traits are: {personality}. Incorporate all these traits in your responses."
        )

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=prompt)  # Only pass the current prompt
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract concepts once for both prompt and response
        combined_text = f"{prompt} {response.content.strip()}"
        concepts = self.extract_concepts(combined_text)
        
        return response.content.strip(), concepts
