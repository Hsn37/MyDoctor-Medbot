import os
import json
import logging
import re

from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dotenv import load_dotenv
import time
import hashlib
from collections import defaultdict
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny
import concurrent.futures
from threading import Thread
from queue import Queue
# Load environment variables
load_dotenv()
def compute_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\n>> Function '{func.__name__}' took {execution_time:.4f} seconds to execute.\n")
        return result
    return wrapper
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
                 embedding_model: str = "text-embedding-3-small"):
        """Initialize the Uganda Medical RAG system"""
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Initialize API keys
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        
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
        

        
        # Verify the collection exists
        self._verify_collection()
        
        # Define content type categories for searching
        self.content_categories = {
            "qa": ["qa_pair", "instruction_qa_pair", "qa_explanation"],
            "conversation": ["patient_doctor_conversation"],
            "reference": ["medical_handbook", "generic_csv", "jsonl_object", "json_object"]
        }
        self.patient_data_cache = {}
        
   # Run this once to create payload indexes (add to initialization)
    @compute_time
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

    @compute_time
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
    @compute_time
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
                    hnsw_ef=32,  # Balance between speed and accuracy
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


  
    @compute_time
    def build_diverse_context_qdrant_batch(self, query: str) -> List[Dict]:
        """Build a diverse context using Qdrant's batch search feature with error handling"""
        try:
            # Generate embedding once for the query
            query_vector = self.generate_embedding(query)
            
            # Define category limits
            category_limits = {
                "qa": 3,
                "conversation": 2,
                "reference": 2
            }
            
            # Prepare batch search requests
            search_requests = []
            
            # Create a search request for each category
            for category, limit in category_limits.items():
                formats = self.content_categories.get(category, [])
                search_requests.append(
                    models.SearchRequest(
                        vector=query_vector,
                        filter=Filter(
                            must=[FieldCondition(key="format", match=MatchAny(any=formats))]
                        ),
                        limit=limit,
                        params=models.SearchParams(hnsw_ef=32),
                        with_payload=True  # Explicitly request payload
                    )
                )
                
            # Execute batch search
            logger.info(f"Executing Qdrant batch search with {len(search_requests)} requests")
            batch_results = self.qdrant_client.search_batch(
                collection_name=self.collection_name,
                requests=search_requests,
                timeout=20
            )
            
            # Process and combine results
            all_results = []
            seen_content_hashes = set()
            
            # Each batch_result corresponds to a category
            for i, category_result in enumerate(batch_results):
                category = list(category_limits.keys())[i]
                logger.info(f"Batch search: {category} results: {len(category_result)}")
                
                # Convert ScoredPoint objects to dictionaries and deduplicate
                for result in category_result:
                    # Guard against None payloads
                    if result.payload is None:
                        logger.warning(f"Received null payload in search results for category: {category}")
                        continue
                        
                    # Guard against missing content
                    content = result.payload.get("content", "")
                    if not content:
                        logger.warning(f"Missing content in payload for category: {category}")
                        continue
                        
                    content_hash = hashlib.md5(content.encode()).hexdigest()
                    
                    if content_hash not in seen_content_hashes:
                        seen_content_hashes.add(content_hash)
                        all_results.append({
                            "score": result.score,
                            "payload": result.payload
                        })
            
            # Sort all results by relevance
            all_results = sorted(all_results, key=lambda x: x["score"], reverse=True)
            
            logger.info(f"Built diverse context with {len(all_results)} total chunks using Qdrant batch search")
            return all_results
            
        except Exception as e:
            logger.error(f"Error in Qdrant batch search: {str(e)}")
            # Add more detailed error logging
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    @compute_time
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
       # print("formatted_chunks: ",formatted_chunks)
        
        # Join all formatted chunks
        return "\n".join(formatted_chunks)
    def set_patient_data(self, patient_id: str, patient_data: Dict[str, Any]) -> None:
        """Set patient data in the cache"""
        self.patient_data_cache[patient_id] = patient_data
        logger.info(f"Patient data set for patient_id: {patient_id}")
    
    def get_patient_data(self, patient_id: Optional[str] = None) -> Dict[str, Any]:
        """Get patient data from cache or return empty data if not available"""
        # If patient_id is provided and exists in cache, return it
        if patient_id and patient_id in self.patient_data_cache:
            logger.info(f"Retrieved patient data for patient_id: {patient_id}")
            return self.patient_data_cache[patient_id]
                
        # Return empty patient data
        logger.info("Using default empty patient data")
        return {
            "patient_id": "",
            "name": "",
            "age": "",
            "location": "",
            "budget": "",
            "critical_conditions": [],
            "past_medical_history": [],
            "current_medications": [],
            "allergies": [],
            "vaccination_history": []     
        }
    
    @compute_time
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
    @compute_time
    def construct_prompt(self, query: str, context: str, patient_data: Dict[str, Any], 
                    message_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Construct the improved prompt for the LLM with JSON structured output instructions"""
        # Convert patient_data to a defaultdict with "" as default value
        patient = defaultdict(lambda: "", patient_data or {})

        # Check if we have valid patient data
        has_patient_data = bool(patient["name"])
        
        # Extract first name for personalized responses if available
        first_name = ""
        if has_patient_data and patient["name"]:
            first_name = patient["name"].split()[0]
        
        # Only include patient info section if we have actual patient data
        patient_info = ""
        if has_patient_data:
            patient_info = f"""
    PATIENT INFORMATION:
    - Name: {patient["name"]} (First name: {first_name})
    - ID: {patient["patient_id"]}
    - Age: {patient["age"]}
    - Location: {patient["location"]}
    - Budget Level: {patient["budget"]}
    - Critical Conditions: {', '.join(patient.get("critical_conditions", []) or ["None"])}
    - Past Medical History: {', '.join(patient.get("past_medical_history", []) or ["None"])}
    - Current Medications: {', '.join(patient.get("current_medications", []) or ["None"])}
    - Allergies: {', '.join(patient.get("allergies", []) or ["None"])}
    - Vaccination History: {', '.join(patient.get("vaccination_history", []) or ["None"])}
    """

        # Format conversation history
        conversation_history, asked_q = self.format_conversation_history(message_history) if message_history else ("", [])
        
        # Main prompt with improved instructions for JSON output
        prompt = f"""You are a medical assistant for patients in Uganda. Help patients by asking relevant questions about their symptoms before providing any diagnosis. Follow these instructions carefully:

    If patient first name {first_name} is empty, use "you" or "your" in the response.

    1. CONVERSATION STYLE:
    - Be warm and friendly
    - Use a reassuring tone but be direct with urgent issues
    - Avoid repeating symptoms back verbatim
    - Don't start every message with greetings like "Hello" or "Hi"

    2. DIAGNOSIS TIMING - VERY IMPORTANT:
    - PRIORITIZE THOROUGH INFORMATION GATHERING. Do NOT provide a diagnosis until you have gathered detailed information covering at least the symptom's onset (when it started), duration (how long it's lasted), severity (how bad it is), character (what it feels like, e.g., sharp/dull pain, wet/dry cough), and any associated symptoms (what else is happening?).
    - Avoid diagnosing after only one or two symptom exchanges unless it is an obvious emergency or the patient has already volunteered comprehensive details covering the points above in their initial message. Err on the side of asking more questions if unsure.

    3. PERSISTENT INFORMATION GATHERING (REQUIRED BEFORE DIAGNOSIS):
    - Ask ONLY ONE focused question at a time to delve deeper into the patient's symptoms.
    - Actively seek details about onset, duration, severity, character, associated symptoms, and anything that makes the main symptom better or worse.
    - Continue asking relevant questions until the criteria in point 2 are met, unless it's an emergency.
    - Keep questions brief and clear.
    - For new conversations, always start with a focused question aimed at understanding the primary symptom better (e.g., "Could you tell me more about this headache, like exactly where you feel it and what kind of pain it is?").

    {asked_q}

    4. DIAGNOSIS AFTER SUFFICIENT INFORMATION:
    - Once sufficient information has been gathered (as defined in point 2), provide ONE likely diagnosis in clear, non-technical language.
    - Briefly explain why you think this is the most likely cause, referencing the specific symptoms discussed.
    - Suggest appropriate next steps or home remedies relevant to Uganda.
    - Diagnosis explanation should be detailed enough to be helpful but remain clear.
    - Relate the diagnosis and advice to the patient's location (Uganda) and context (e.g., recommending MyDoctor clinic).

    5. EMERGENCY HANDLING:
    - Dont add anything to the response other than the final answer to the patient
    - Format your response as if speaking directly to the patient
    - Follow the response format strictly
    - For chest pain, breathing difficulty, severe bleeding, loss of consciousness, basically anything that is life threatening or can cause severe harm if left untreated should be flagged emergency:
        - Immediately advise seeking urgent medical attention
        - Recommend nearest clinic or hospital 
        - No additional questions for emergencies

    6. LOCAL CONTEXT:
    - Use Ugandan terms like "hot body" for fever
    - Recommend MyDoctor clinic in Kampala for in-person care
    - Consider local conditions and resources

    7. PATIENT HISTORY INTEGRATION:
    - For diabetic patients, consider blood sugar issues first
    - Mention current medications when relevant
    - Adjust advice based on medical history

    8. RESPONSE FORMAT - VERY IMPORTANT:
    - You MUST format your response as a valid JSON object with these exact fields:
    "response": "Your actual response text here...",
    "is_emergency": true or false (boolean, not string)





    - In the "response" field:
    * Format your response as if speaking directly to the patient
    * Provide only the final answer to the patient
    * For very first symptom: Start with 'I’m really sorry to hear you're feeling unwell {first_name}. To understand things better before suggesting any possible causes, I’d like to ask you a few quick questions to get a better idea of what might be going on'. However, also be smart enough to not use this line if someone is saying completely different thing in the start which is not related to symptoms. Respond accordingly in a nice polite way in this case.
    * For diagnosis: Use clear, non-technical language
    * For emergencies: Start with "Thank you for telling me {first_name}. What you're describing can be serious...". However, also be smart enough to not use this line if someone is saying I am going to die or help me please or things along these lines. Respond accordingly to these.
    * Don't use {first_name} in the response, just use "you" or "your"

    - Set the "is_emergency" field to true ONLY for situations requiring immediate medical attention (e.g., severe chest pain, difficulty breathing, signs of stroke, severe bleeding, etc.)

    Example JSON response format for non-emergency:
    {{
    "response": "I understand you've been having headaches. Could you tell me more about when they started and how severe they are?",
    "is_emergency": false
    }}

    Example JSON response format for emergency:
    {{
    "response": "What you're describing sounds like a medical emergency. Chest pain that radiates to your arm and jaw, along with difficulty breathing, could indicate a heart attack. Please seek immediate medical attention at the nearest hospital emergency department.",
    "is_emergency": true
    }}

    {patient_info}

    {conversation_history}

    MEDICAL REFERENCE INFORMATION:
    --------------------------
    {context}
    --------------------------

    CURRENT QUERY: {query}

    IMPORTANT: Your entire response must be a valid, properly formatted JSON object exactly as shown in the examples. The response field should contain your complete response to the patient, and the is_emergency field must be a boolean (true/false).
    """
        return prompt

    @compute_time  
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
    @compute_time
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
            
  
            # Sort all results by relevance
            all_results = sorted(all_results, key=lambda x: x["score"], reverse=True)
            
            logger.info(f"Built diverse context with {len(all_results)} total chunks in parallel")
            return all_results
            
        except Exception as e:
            logger.error(f"Error building diverse context in parallel: {str(e)}")
            return []

    @compute_time
    def generate_answer_stream(self, query: str, patient_id: Optional[str] = None,
                            message_history: Optional[List[Dict[str, str]]] = None):
        """Process a query and return a JSON response with an explicit emergency flag"""
        try:
            # Retrieve context
            context_chunks = []
            if not message_history or len(message_history) <= 1:
                context_chunks = self.build_diverse_context_parallel(query)
            else:
                retrieval_query = self.build_retrieval_query(query, message_history)
                logger.info(f"Built context-aware query: '{retrieval_query}'")
                context_chunks = self.build_diverse_context_qdrant_batch(retrieval_query)

            # Handle empty context results
            if not context_chunks:
                yield json.dumps({
                    "response": "I'm sorry, I couldn't find relevant medical information to answer your question accurately. Please contact a healthcare professional for assistance.", 
                    "is_emergency": False
                })
                return
            
            # Format the context chunks
            formatted_context = self.format_context(context_chunks)
            
            # Get patient data
            patient_data = self.get_patient_data(patient_id)
            
            # Construct prompt with JSON output instructions
            full_prompt = self.construct_prompt(query, formatted_context, patient_data, message_history)
            
            # Generate response
            response_stream = self.operouter_client.completions.create(
                model="google/gemini-2.0-flash-001",
                prompt=full_prompt,
                stream=False,  # Non-streaming for simpler processing
            )
            
            # Process the response
            if hasattr(response_stream, 'choices') and response_stream.choices:
                full_response = response_stream.choices[0].text.strip()
                
                # Try to parse JSON response
                try:
                    # Clean up any potential issues with JSON formatting
                    # Some models might include markdown code block markers
                    if full_response.startswith("```json"):
                        full_response = full_response.replace("```json", "", 1)
                    if full_response.endswith("```"):
                        full_response = full_response.rsplit("```", 1)[0]
                    
                    full_response = full_response.strip()
                    
                    # Parse JSON response
                    response_data = json.loads(full_response)
                    
                    # Ensure required fields are present
                    if "response" not in response_data:
                        response_data["response"] = "I apologize, but I encountered an error while processing your question."
                    
                    if "is_emergency" not in response_data:
                        # If missing emergency flag, check text for emergency indicators
                        emergency_indicators = ["emergency", "immediate", "urgent", "call ambulance", 
                                            "go to hospital", "911", "life-threatening"]
                        response_text = response_data["response"].lower()
                        response_data["is_emergency"] = any(ind in response_text for ind in emergency_indicators)
                    
                    # Log the emergency status
                    if response_data["is_emergency"]:
                        logger.warning(f"EMERGENCY detected in query: '{query}'")
                    
                    # Return the structured response
                    yield json.dumps(response_data)
                    
                except json.JSONDecodeError:
                    # Handle case where model didn't return valid JSON
                    logger.warning(f"Failed to parse JSON from LLM response: {full_response}")
                    
                    # Check for emergency indicators in the text
                    emergency_indicators = ["emergency", "immediate", "urgent", "call ambulance", 
                                        "go to hospital", "911", "life-threatening"]
                    is_emergency = any(ind in full_response.lower() for ind in emergency_indicators)
                    
                    # Create properly formatted JSON
                    result = {
                        "response": full_response,
                        "is_emergency": is_emergency
                    }
                    
                    if is_emergency:
                        logger.warning(f"Emergency detected through text analysis in query: '{query}'")
                    
                    yield json.dumps(result)
            
        except Exception as e:
            logger.error(f"Error streaming response: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            yield json.dumps({
                "response": "I apologize, but I encountered an error while processing your question. Please try again or contact support if the issue persists.",
                "is_emergency": False
            })
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