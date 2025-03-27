import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dotenv import load_dotenv
import time
import hashlib
from google.genai import types

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny
from google import genai
import concurrent.futures
from threading import Thread
from queue import Queue
# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("query_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("query_processor")

class UgandaMedicalRAG:
    def __init__(self, 
                 collection_name: str = "uganda_medical",
                 embedding_model: str = "text-embedding-3-small",
                 llm_model: str = "gemini-2.0-flash"):
        """Initialize the Uganda Medical RAG system"""
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        # Initialize API keys
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Initialize clients
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.operouter_client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=self.openrouter_api_key,
)
        
        # Initialize Qdrant Cloud client
        self.qdrant_url = os.environ.get("QDRANT_URL")
        self.qdrant_api_key = os.environ.get("QDRANT_API_KEY")
        
        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in environment variables")
        
        self.qdrant_client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            timeout=15  
        )
        
        logger.info(f"Connected to Qdrant Cloud at {self.qdrant_url}")
        
        # Initialize Gemini
        self.gemini_client = genai.Client(api_key=self.gemini_api_key)
        
        # Verify the collection exists
        self._verify_collection()
        
        # Define content type categories for searching
        self.content_categories = {
            "qa": ["qa_pair", "instruction_qa_pair", "qa_explanation"],
            "conversation": ["patient_doctor_conversation"],
            "reference": ["medical_handbook", "generic_csv", "jsonl_object", "json_object"]
        }
        
   # Run this once to create payload indexes (add to initialization)
    def _verify_collection(self) -> None:
        """Verify and optimize collection configuration"""
        from qdrant_client.http import models as rest
        
        # Existing verification
        collections = self.qdrant_client.get_collections().collections
        if self.collection_name not in [c.name for c in collections]:
            raise ValueError(f"Collection {self.collection_name} does not exist")

        # Create payload index if not exists
        try:
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="format",
                field_type=rest.PayloadSchemaType.KEYWORD
            )
            logger.info("Created payload index for 'format' field")
        except Exception as e:
            logger.info(f"Payload index already exists: {str(e)}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text string"""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def search_by_category(self, query_vector: List[float], category: str, limit: int) -> List[Dict]:
        """Search with timeout and optimized parameters"""
        try:
            formats = self.content_categories.get(category, [])
            query_filter = Filter(
                must=[FieldCondition(key="format", match=MatchAny(any=formats))]
            )

            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                timeout=20,  # Increase timeout to 20 seconds
                search_params=models.SearchParams(
                    hnsw_ef=128,  # Balance between speed and accuracy
                )
            )
            
            # Convert ScoredPoint objects to dictionaries
            results = []
            for result in search_result:
                results.append({
                    "score": result.score,
                    "payload": result.payload
                })
                
            logger.info(f"Retrieved {len(results)} results for category: {category}")
            return results
            
        except Exception as e:
            logger.warning(f"Search error in {category} category: {str(e)}")
            return []

    # def search_with_offset(self, query_vector: List[float], limit: int, offset: int = 0) -> List[Dict]:
    #     """Search for similar vectors with offset pagination"""
    #     try:
    #         search_result = self.qdrant_client.search(
    #             collection_name=self.collection_name,
    #             query_vector=query_vector,
    #             limit=limit,
    #             offset=offset
    #         )
            
    #         # Convert to dictionary format
    #         results = [
    #             {
    #                 "score": result.score,
    #                 "payload": result.payload
    #             }
    #             for result in search_result
    #         ]
            
    #         return results
            
    #     except Exception as e:
    #         logger.warning(f"Error in offset search: {str(e)}")
    #         return []
    
    def build_diverse_context(self, query: str) -> List[Dict]:
        """Build a diverse context by searching different content categories"""
        try:
            # Generate embedding once for the query
            query_vector = self.generate_embedding(query)
            
            # Define limits for each category
            category_limits = {
                "qa": 3,        # More Q&A content as it's most relevant
                "conversation": 2,  # Some conversation examples 
                "reference": 2   # Some reference material
            }
            
            # Collect all results
            all_results = []
            
            # Search each category separately
            for category, limit in category_limits.items():
                category_results = self.search_by_category(query_vector, category, limit)
                all_results.extend(category_results)
            
            # # If we didn't get enough results from filtered searches, try simple pagination
            # if len(all_results) < 3:
            #     logger.info("Category searches yielded insufficient results, trying pagination")
                
            #     # Use offset searches to get more diverse results
            #     for offset in [0, 2, 4]:
            #         offset_results = self.search_with_offset(query_vector, limit=2, offset=offset)
                    
            #         # Add only new results (avoid duplicates)
            #         for result in offset_results:
            #             # Create a simple hash of the content to check for duplicates
            #             content_hash = hashlib.md5(result["payload"].get("content", "").encode()).hexdigest()
                        
            #             # Check if this content is already in all_results
            #             if not any(hashlib.md5(r["payload"].get("content", "").encode()).hexdigest() == content_hash 
            #                     for r in all_results):
            #                 all_results.append(result)
            
            # Sort all results by relevance
            all_results = sorted(all_results, key=lambda x: x["score"], reverse=True)
            
            logger.info(f"Built diverse context with {len(all_results)} total chunks")
            return all_results
            
        except Exception as e:
            logger.error(f"Error building diverse context: {str(e)}")
            return []
    
    def format_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Format the context chunks for LLM prompt"""
        if not context_chunks:
            return ""
            
        formatted_chunks = []
        
        # Group chunks by category for better organization
        qa_chunks = []
        conversation_chunks = []
        reference_chunks = []
        
        for chunk in context_chunks:
            format_type = chunk["payload"].get("format", "unknown")
            
            if format_type in self.content_categories["qa"]:
                qa_chunks.append(chunk)
            elif format_type in self.content_categories["conversation"]:
                conversation_chunks.append(chunk)
            elif format_type in self.content_categories["reference"]:
                reference_chunks.append(chunk)
            else:
                # Fallback for unknown formats
                reference_chunks.append(chunk)
        
        # Format chunks by category with headers
        if qa_chunks:
            formatted_chunks.append("MEDICAL Q&A EXAMPLES:")
            for chunk in qa_chunks:
                content = chunk["payload"]["content"]
                source = chunk["payload"]["source"]
                formatted_chunks.append(f"SOURCE: {source}\n{content}\n")
        
        if conversation_chunks:
            formatted_chunks.append("PATIENT-DOCTOR CONVERSATIONS:")
            for chunk in conversation_chunks:
                content = chunk["payload"]["content"]
                source = chunk["payload"]["source"]
                formatted_chunks.append(f"SOURCE: {source}\n{content}\n")
        
        if reference_chunks:
            formatted_chunks.append("MEDICAL REFERENCE MATERIAL:")
            for chunk in reference_chunks:
                content = chunk["payload"]["content"]
                source = chunk["payload"]["source"]
                formatted_chunks.append(f"SOURCE: {source}\n{content}\n")
        #print("formatted_chunks: ",formatted_chunks)
        
        # Join all formatted chunks
        return "\n".join(formatted_chunks)
    
    def get_patient_data(self, patient_id: Optional[str] = None) -> Dict[str, Any]:
        """Get patient data from API (or mock data if API not available)"""
        # For demo purposes, using mock patient records        
        # Default mock patient
        return {
            "patient_id": "UG1001",
            "name": "Leticia Okello",
            "age": 35,
            "location": "Entebbe",
            "budget": "Mid-Low",
            "critical_conditions": ['diabetic'],
            "past_medical_history": [],
            "current_medications": ['Metformin'],
            "allergies": [],
            "vaccination_history": ['COVID-19']     
        }
    
    def format_conversation_history(self, message_history: List[Dict[str, str]]) -> Tuple[str, List[str]]:
        """Format conversation history and extract previously asked questions"""
        if not message_history:
            return "", []
            
        # Format the conversation history
        history_text = "CONVERSATION HISTORY:\n"
        
        # Track questions that have been asked
        asked_questions = []
        
        # Use the recent messages
        recent_messages = message_history[-6:] if len(message_history) > 6 else message_history
        
        for message in recent_messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "user":
                history_text += f"Patient: {content}\n"
            elif role == "assistant":
                history_text += f"Medical Assistant: {content}\n\n"
                # Extract questions from assistant responses
                # if "?" in content:
                #     questions = [q.strip() + "?" for q in content.split("?")[:-1]]
                #     asked_questions.extend(questions)
        for message in message_history:
            if '?' in message['content']:
                asked_questions.append(message['content'])
        # Add explicit section on previously asked questions
        if asked_questions:
            history_text += "\nPREVIOUSLY ASKED QUESTIONS (DO NOT REPEAT THESE):\n"
            for i, question in enumerate(asked_questions):
                history_text += f"{i+1}. {question}\n"
        
        return history_text, asked_questions
    
    def construct_prompt(self, query: str, context: str, patient_data: Dict[str, Any], 
                 message_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Construct the improved prompt for the LLM to encourage step-by-step questioning"""
        # Extract first name for personalized responses
        first_name = patient_data.get('name', 'Unknown').split()[0] if patient_data.get('name') else "there"
        
        # Format patient data
        patient_info = f"""
    PATIENT INFORMATION:
    - Name: {patient_data.get('name', 'Unknown')} (First name: {first_name})
    - ID: {patient_data.get('patient_id', 'Unknown')}
    - Age: {patient_data.get('age', 'Unknown')}
    - Location: {patient_data.get('location', 'Unknown')}
    - Budget Level: {patient_data.get('budget', 'Unknown')}
    - Critical Conditions: {', '.join(patient_data.get('critical_conditions', ['None']))}
    - Past Medical History: {', '.join(patient_data.get('past_medical_history', ['None']))}
    - Current Medications: {', '.join(patient_data.get('current_medications', ['None']))}
    - Allergies: {', '.join(patient_data.get('allergies', ['None']))}
    - Vaccination History: {', '.join(patient_data.get('vaccination_history', ['None']))}
    """

        # Format conversation history
        conversation_history,asked_q = self.format_conversation_history(message_history) if message_history else ""
        print("asked_q: ",asked_q)
        print("conversation_history: ",conversation_history)

        # Main prompt with improved instructions
        prompt = f"""You are a medical assistant for patients in Uganda. You help patients by asking relevant questions about their symptoms before providing any diagnosis. Follow these instructions carefully:

    1. CONVERSATION STYLE:
    - Be warm and friendly, always address the patient as "{first_name}"
    - Use a reassuring tone but be direct with urgent issues
    - Avoid repeating symptoms back verbatim
    - Don't start every message with greetings like "Hello" or "Hi"

    2. INFORMATION GATHERING:
    - Ask ONLY ONE focused question at a time about symptoms
    - Keep questions brief and clear
    - Wait for sufficient information before diagnosing
    - For new conversations, always start with a focused question

    {asked_q}

    3. DIAGNOSIS AFTER MINIMAL INFORMATION:
    - After receiving answers to 4-5 questions OR if the initial query contains enough details:
    - Provide ONE likely diagnosis in clear, non-technical language
    - Briefly explain why you think this is the most likely cause
    - Suggest appropriate next steps or home remedies

    4. EMERGENCY HANDLING:
    - For chest pain, breathing difficulty, severe bleeding, loss of consciousness:
        - Immediately advise seeking urgent medical attention
        - Recommend nearest clinic or hospital 
        - No additional questions for emergencies

    5. LOCAL CONTEXT:
    - Use Ugandan terms like "hot body" for fever
    - Recommend MyDoctor clinic in Kampala for in-person care
    - Consider local conditions and resources

    6. PATIENT HISTORY INTEGRATION:
    - For diabetic patients, consider blood sugar issues first
    - Mention current medications when relevant
    - Adjust advice based on medical history

    {patient_info}

    {conversation_history}

    MEDICAL REFERENCE INFORMATION:
    --------------------------
    {context}
    --------------------------

    CURRENT QUERY: {query}

    RESPONSE FORMAT:
    1. For new symptoms: Start with "I'm sorry to hear that, {first_name}" then ask ONE focused question
    2. For follow-ups: Acknowledge the answer briefly, then ask ONE more question if needed
    3. For diagnosis: Structure as described in DIAGNOSIS STRUCTURE above
    4. For emergencies: Start with "Thank you for telling me {first_name}. What you're describing can be serious..."

    Remember to use {first_name} throughout your response to personalize it.
    """
        return prompt
#     def generate_response(self, prompt: str) -> str:
#         """Generate a response using Google's Gemini LLM"""
#         try:
#             # g_config = types.GenerateContentConfig(
#             #     temperature=0.1,
#             #     top_p=0.85,
#             #     top_k=40,
#             #     max_output_tokens=8192
#             # )
            
#             # response = self.gemini_client.models.generate_content(
#             #     model=self.llm_model,
#             #     contents=prompt,
#             #     config=g_config
#             # )
#             response=self.operouter_client.completions.create(
#   model="deepseek/deepseek-chat-v3-0324:free",
#     prompt=prompt,
#             )

            
#             # return response.text
#             return response.choices[0].message.content
        
#         except Exception as e:
#             logger.error(f"Error generating LLM response: {str(e)}")
#             return "I apologize, but I'm experiencing technical difficulties. Please try again later or contact healthcare support if you need immediate assistance."
        
    def build_retrieval_query(self, current_query: str, message_history: List[Dict[str, str]]) -> str:
        """Build a comprehensive query using conversation history"""
        
        # Start with the current query
        retrieval_query = current_query
        
        # If the query is very short, likely a follow-up

        # Extract previous exchanges to provide context
        recent_exchanges = message_history[-4:]  # Last 2 turns
        
        # Get the last substantive user query
        for msg in reversed(recent_exchanges):
            if msg["role"] == "user" and len(msg["content"].split()) > 3:
                retrieval_query = f"{msg['content']} {current_query}"
                break
        
        # Also include the chatbot's last question for context
        for msg in reversed(recent_exchanges):
            if msg["role"] == "assistant" and "?" in msg["content"]:
                # Extract the question
                last_question = msg["content"].split("?")[0] + "?"
                retrieval_query = f"{retrieval_query} {last_question}"
                break
        
        return retrieval_query
    # def generate_answer(self, query: str, patient_id: Optional[str] = None, 
    #                    message_history: Optional[List[Dict[str, str]]] = None) -> str:
    #     """Process a query end-to-end using diverse context building"""
    #     try:
    #         query=self.build_retrieval_query(query, message_history)
    #         print("query in generate answer : ",query)
    #         # Build context with specialized category searching
    #         context_chunks = self.build_diverse_context(query)
            
    #         if not context_chunks:
    #             return "I'm sorry, I couldn't find relevant medical information to answer your question accurately. Please contact a healthcare professional for assistance."
            
    #         # Format the context chunks
    #         formatted_context = self.format_context(context_chunks)
            
    #         # Get patient data
    #         patient_data = self.get_patient_data(patient_id)
            
    #         # Construct prompt with context and conversation history
    #         prompt = self.construct_prompt(query, formatted_context, patient_data, message_history)
            
    #         # Generate response
    #         response = self.generate_response(prompt)
            
    #         return response
            
    #     except Exception as e:
    #         logger.error(f"Error processing query: {str(e)}")
    #         return "I apologize, but I encountered an error while processing your question. Please try again or contact support if the issue persists."


    def build_diverse_context_parallel(self, query: str) -> List[Dict]:
        """Build a diverse context by searching different content categories in parallel"""
        try:
            # Generate embedding once for the query
            query_vector = self.generate_embedding(query)
            
            # Define limits for each category
            category_limits = {
                "qa": 3,        # More Q&A content as it's most relevant
                "conversation": 2,  # Some conversation examples 
                "reference": 2   # Some reference material
            }
            
            # Dictionary to store results from each category
            all_results = []
            
            # Function to execute category search in parallel
            def search_category(category, limit):
                try:
                    results = self.search_by_category(query_vector, category, limit)
                    return results
                except Exception as e:
                    logger.warning(f"Error in parallel search for {category}: {str(e)}")
                    return []
            
            # Use ThreadPoolExecutor to run searches in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(category_limits)) as executor:
                # Start searches for each category
                future_to_category = {
                    executor.submit(search_category, category, limit): category
                    for category, limit in category_limits.items()
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_category):
                    category = future_to_category[future]
                    try:
                        results = future.result()
                        all_results.extend(results)
                        logger.info(f"Parallel search for '{category}' complete: {len(results)} results")
                    except Exception as e:
                        logger.error(f"Exception in parallel search for '{category}': {str(e)}")
            
            # # If we didn't get enough results from filtered searches, try fallback search in parallel
            # if len(all_results) < 3:
            #     logger.info("Category searches yielded insufficient results, trying parallel pagination")
                
            #     offsets = [0, 2, 4]
            #     # Function for offset search
            #     def search_with_offset_wrapper(offset):
            #         return self.search_with_offset(query_vector, limit=2, offset=offset)
                
            #     # Run offset searches in parallel
            #     with concurrent.futures.ThreadPoolExecutor(max_workers=len(offsets)) as executor:
            #         offset_results_list = list(executor.map(search_with_offset_wrapper, offsets))
                
            #     # Process all offset results
            #     for offset_results in offset_results_list:
            #         # Add only new results (avoid duplicates)
            #         for result in offset_results:
            #             # Create a simple hash of the content to check for duplicates
            #             content_hash = hashlib.md5(result["payload"].get("content", "").encode()).hexdigest()
                        
            #             # Check if this content is already in all_results
            #             if not any(hashlib.md5(r["payload"].get("content", "").encode()).hexdigest() == content_hash 
            #                     for r in all_results):
            #                 all_results.append(result)
            
            # Sort all results by relevance
            all_results = sorted(all_results, key=lambda x: x["score"], reverse=True)
            
            logger.info(f"Built diverse context with {len(all_results)} total chunks in parallel")
            return all_results
            
        except Exception as e:
            logger.error(f"Error building diverse context in parallel: {str(e)}")
            return []
    def generate_answer_stream(self, query: str, patient_id: Optional[str] = None,
                            message_history: Optional[List[Dict[str, str]]] = None):
        """Process a query and stream the answer with conversation-aware context building"""
        try:
            # Store context globally for session reuse
            context_chunks = []
            
            # If no message history or very short conversation, use normal processing
            if not message_history or len(message_history) <= 1:
                # Use a standard search for initial query
                context_chunks = self.build_diverse_context_parallel(query)
            else:
                # For follow-up questions, use a more intelligent context strategy
                    # 2. Build a context-aware query from conversation
                    retrieval_query = self.build_retrieval_query(query, message_history)
                    logger.info(f"Built context-aware query: '{retrieval_query}'")
                    
                    # 3. Search using the enhanced query
                    context_chunks = self.build_diverse_context_parallel(retrieval_query)
                    

            
            # Handle empty context results
            if not context_chunks:
                yield "I'm sorry, I couldn't find relevant medical information to answer your question accurately. Please contact a healthcare professional for assistance."
                return
            
            # Format the context chunks
            formatted_context = self.format_context(context_chunks)
            
            # Get patient data
            patient_data = self.get_patient_data(patient_id)
            
            # Construct prompt with context and conversation history
            prompt = self.construct_prompt(query, formatted_context, patient_data, message_history)
            
            # Stream response using Gemini
            # response_stream = self.gemini_client.models.generate_content_stream(
            #     model=self.llm_model,
            #     contents=prompt
            # )

            response_stream=self.operouter_client.completions.create(
  model="deepseek/deepseek-chat-v3-0324:free",
  prompt=prompt,
#   stop=["##", "BEGIN ASSISTANT RESPONSE"],
  extra_body={},
  stream=True
            )
            
            # Stream the response chunks
            # for chunk in response_stream:
            #     if hasattr(chunk, 'text'):
            #         yield chunk.text
            # for chunk in response_stream:
            #     if chunk.choices[0].text is not None:
            #         yield(chunk.choices[0].text)
            # Stream the response chunks
            for chunk in response_stream:
                # Check if there's a response in the chunk
                if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                    # Extract the text from the choice
                    choice = chunk.choices[0]
                    if hasattr(choice, 'text') and choice.text:
                        yield choice.text
            
        except Exception as e:
            logger.error(f"Error streaming response: {str(e)}")
            yield "I apologize, but I encountered an error while processing your question. Please try again or contact support if the issue persists."


# if __name__ == "__main__":
#     # # Simple test
#     # rag = UgandaMedicalRAG()
    
#     # # Create a mock conversation history
#     # conversation_history = [
#     #     {"role": "user", "content": "Hello, I'm not feeling well"},
#     #     {"role": "assistant", "content": "I'm sorry to hear that. Can you describe your symptoms?"}
#     # ]
    
#     # # Test with diabetic patient
#     # diabetic_patient = "UG1234"
#     # mock_query = "I've been feeling very thirsty lately and urinating frequently. What could this be?"
    
#     # response = rag.generate_answer(
#     #     query=mock_query,
#     #     patient_id=diabetic_patient,
#     #     message_history=conversation_history
#     # )
    
#     # print(f"Response for diabetic patient (ID: {diabetic_patient}):")
#     # print(response)