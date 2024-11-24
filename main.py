from memoripy import MemoryManager, JSONStorage
import os
from colorama import Fore, Style, init

def main():
    init(autoreset=True)  # Initialize colorama

    # Replace with your actual Groq API key
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError("Please set your Groq API key.")

    # Define chat and embedding models for Groq
    #chat_model = "groq"            # Use Groq for chat
    #chat_model_name = "llama-3.2-3b-preview" # Groq's chat model
    #chat_model_name = "llama3-groq-70b-8192-tool-use-preview" # Groq's chat model
    chat_model = "ollama"            # Use Groq for chat
    chat_model_name = "llama3.2:latest" # Groq's chat model
    embedding_model = "ollama"      # Choose 'openai' or 'ollama' for embeddings
    embedding_model_name = "mxbai-embed-large"  # Specific embedding model name

    # Choose your storage option
    storage_option = JSONStorage("interaction_history.json")
    # Or use in-memory storage:
    #from memoripy import InMemoryStorage
    #storage_option = InMemoryStorage()

    # Initialize the MemoryManager with the selected models and storage
    memory_manager = MemoryManager(
        api_key=api_key,
        chat_model=chat_model,
        chat_model_name=chat_model_name,
        embedding_model=embedding_model,
        embedding_model_name=embedding_model_name,
        storage=storage_option,
        verbose=True  # Enable detailed logging
    )

    print("Welcome to the conversation! (Type 'exit' to end)")
    
    while True:
        # Get user input
        new_prompt = input(Fore.GREEN + "\nYou: " + Style.RESET_ALL).strip()
        
        # Check for exit condition
        if new_prompt.lower() == 'exit':
            print("Goodbye!")
            break
            
        if not new_prompt:
            continue

        # Load recent context
        short_term, long_term = memory_manager.load_history()  # Only unpack what we need
        last_interactions = short_term[-5:] if len(short_term) >= 5 else short_term

        # Get relevant past interactions
        relevant_interactions = memory_manager.retrieve_relevant_interactions(new_prompt, exclude_last_n=5)

        # Generate and display response
        response, concepts = memory_manager.generate_response(new_prompt, last_interactions, relevant_interactions)
        print(Fore.CYAN + "\nAssistant: " + response + Style.RESET_ALL)

        # Process and store the interaction (now using already extracted concepts)
        combined_text = f"{new_prompt} {response}"
        new_embedding = memory_manager.get_embedding(combined_text)
        memory_manager.add_interaction(new_prompt, response, new_embedding, concepts)

if __name__ == "__main__":
    main()
