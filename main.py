from PIL import Image
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
    chat_model = "groq"            # Use Groq for chat
    #chat_model_name = "llama-3.1-70b-versatile" # Groq's chat model
    #chat_model_name = "llama3-groq-70b-8192-tool-use-preview" # Groq's chat model
    chat_model_name= "llama-3.2-3b-preview"
    #chat_model_name = "llama-3.1-8b-instant" # Groq's chat model
    #chat_model = "ollama"            # Use Groq for chat
    #chat_model_name = "llama3.2:latest" # Groq's chat model
    
    # Add vision model configuration
    vision_model_name = "llama-3.2-11b-vision-preview"
    
    # Define embedding model for Groq
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
        vision_model_name=vision_model_name
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

        # Check for core memory update first
        #print("\nCore memory before update:", memory_manager.memory_store.core_memory)
        is_core, processed_prompt = memory_manager.store_core_memory(new_prompt)
        print("Is core memory update:", is_core)
        #print("Core memory after update:", memory_manager.memory_store.core_memory)
        
        if is_core:
            # For core memories, just get and show the response
            response = memory_manager.generate_response(processed_prompt, [], [])
            print(Fore.CYAN + "\nAssistant: " + response + Style.RESET_ALL)
            continue

        # Check if request requires visual processing
        vision_response = memory_manager.check_visual_request(new_prompt)
        if vision_response:
            print(f"Vision check result: {vision_response.is_visual}")
            print(f"Reasoning: {vision_response.reasoning}")

            if vision_response.is_visual:
                # Use hardcoded image path
                image_path = r"C:\Users\Henrique\Desktop\stuff.jpeg"
                try:
                    response = memory_manager.process_visual_request(new_prompt, image_path)
                    print(Fore.CYAN + "\nAssistant (Vision): " + response + Style.RESET_ALL)
                    
                    # Store vision interaction in memory
                    combined_text = f"{new_prompt} {response}"
                    concepts = memory_manager.extract_concepts(combined_text)
                    new_embedding = memory_manager.get_embedding(combined_text)
                    
                    memory_manager.add_interaction(
                        new_prompt,
                        response,
                        new_embedding,
                        concepts,
                        is_core_memory=False
                    )
                except Exception as e:
                    print(f"Vision processing failed: {e}")
                    # Fall back to regular processing
                continue  # Skip regular processing if vision request was handled

        # Regular memory processing path
        # Load recent context
        history = memory_manager.load_history()
        short_term = history[0]
        last_interactions = short_term[-5:] if len(short_term) >= 5 else short_term

        # Get relevant past interactions
        relevant_interactions = memory_manager.retrieve_relevant_interactions(processed_prompt, exclude_last_n=5)

        if vision_response.is_visual:
            # Use hardcoded image path
            image_path = r"C:\Users\Henrique\Desktop\stuff.jpeg"
            try:
                response = memory_manager.process_visual_request(new_prompt, image_path)
                print(Fore.CYAN + "\nAssistant (Vision): " + response + Style.RESET_ALL)
                
                # Store vision interaction in memory
                combined_text = f"{new_prompt} {response}"
                concepts = memory_manager.extract_concepts(combined_text)
                new_embedding = memory_manager.get_embedding(combined_text)
                
                memory_manager.add_interaction(
                    new_prompt,
                    response,
                    new_embedding,
                    concepts,
                    is_core_memory=False
                )
            except Exception as e:
                print(f"Vision processing failed: {e}")
                # Fall back to regular processing
        else:
            # Generate response
            response = memory_manager.generate_response(processed_prompt, last_interactions, relevant_interactions)
            print(Fore.CYAN + "\nAssistant: " + response + Style.RESET_ALL)

            # Process and store regular interaction
            combined_text = f"{processed_prompt} {response}"
            concepts = memory_manager.extract_concepts(combined_text)
            new_embedding = memory_manager.get_embedding(combined_text)
            
            memory_manager.add_interaction(
                processed_prompt, 
                response, 
                new_embedding, 
                concepts,
                is_core_memory=False
            )

if __name__ == "__main__":
    main()
