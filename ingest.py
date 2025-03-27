import os
import json
import pandas as pd
import PyPDF2
import re
import hashlib
import logging
from typing import List, Dict, Any, Tuple
#from dotenv import load_dotenv
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ingestion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_ingestor")

class UgandaMedicalIngestor:
    def __init__(self, data_dir: str = None):
        """Initialize the data ingestor with the directory containing the data files"""
        self.data_dir = data_dir or os.getcwd()
        self.content_hashes = set()  # For deduplication
        self.processed_data = []
    
    def process_all_files(self) -> List[Dict[str, Any]]:
        """Process all PDF, CSV, and JSON files in the data directory"""
        logger.info(f"Processing files in: {self.data_dir}")
        
        # Get all files
        files = [f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f))]
        target_files = [f for f in files if f.lower().endswith(('.pdf', '.json', '.csv'))]
        
        if not target_files:
            logger.warning("No PDF, JSON, or CSV files found in directory")
            return []
        
        logger.info(f"Found {len(target_files)} files to process")
        
        # Process each file
        for file in target_files:
            file_path = os.path.join(self.data_dir, file)
            file_extension = os.path.splitext(file)[1].lower()
            
            logger.info(f"Processing file: {file}")
            
            try:
                # Process based on file type
                if file_extension == '.json':
                    self._process_json_file(file_path)
                elif file_extension == '.csv':
                    self._process_csv_file(file_path)
                elif file_extension == '.pdf':
                    self._process_pdf_file(file_path)
            except Exception as e:
                logger.error(f"Error processing {file}: {str(e)}")
        
        logger.info(f"Total chunks created: {len(self.processed_data)}")
        return self.processed_data
    
    def _add_chunk(self, content: str, metadata: Dict[str, Any]) -> bool:
        """Add a chunk to processed data with deduplication"""
        if not content or not content.strip():
            return False
            
        # Create hash for deduplication
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Skip if duplicate
        if content_hash in self.content_hashes:
            return False
            
        # Add to processed data
        self.content_hashes.add(content_hash)
        self.processed_data.append({
            "content": content.strip(),
            "metadata": metadata
        })
        
        return True
    
    def _process_json_file(self, file_path: str) -> None:
        """Process a JSON file and extract chunks"""
        filename = os.path.basename(file_path)
        chunks_added = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    # Try to parse as a JSON array/object
                    data = json.load(file)
                    
                    # Handle array format
                    if isinstance(data, list):
                        # Check for instruction/input/output format
                        if all("instruction" in item and "input" in item and "output" in item 
                              for item in data[:5] if isinstance(item, dict)):
                            logger.info(f"Processing instruction/input/output format")
                            
                            for item in data:
                                content = f"Question: {item.get('input', '')}\nAnswer: {item.get('output', '')}"
                                metadata = {
                                    "source": filename,
                                    "format": "instruction_qa_pair"
                                }
                                if self._add_chunk(content, metadata):
                                    chunks_added += 1
                        else:
                            # Try to detect structure
                            if len(data) > 0 and isinstance(data[0], dict):
                                for item in data:
                                    content_parts = []
                                    for key, value in item.items():
                                        if isinstance(value, str) and value.strip():
                                            content_parts.append(f"{key}: {value}")
                                    
                                    if content_parts:
                                        content = "\n".join(content_parts)
                                        metadata = {
                                            "source": filename,
                                            "format": "json_object"
                                        }
                                        if self._add_chunk(content, metadata):
                                            chunks_added += 1
                except json.JSONDecodeError:
                    # Try line-by-line JSON objects (JSONL format)
                    file.seek(0)
                    
                    for line in file:
                        if not line.strip():
                            continue
                        
                        try:
                            item = json.loads(line)
                            
                            # Format JSONL content based on structure
                            if "question" in item:
                                content = f"Question: {item.get('question', '')}\n"
                                if "exp" in item:
                                    content += f"Answer: {item.get('exp', '')}"
                                elif "answer" in item:
                                    content += f"Answer: {item.get('answer', '')}"
                                    
                                metadata = {
                                    "source": filename,
                                    "format": "qa_explanation"
                                }
                                if self._add_chunk(content, metadata):
                                    chunks_added += 1
                            else:
                                # Generic object extraction
                                content_parts = []
                                for key, value in item.items():
                                    if isinstance(value, str) and value.strip():
                                        content_parts.append(f"{key}: {value}")
                                
                                if content_parts:
                                    content = "\n".join(content_parts)
                                    metadata = {
                                        "source": filename,
                                        "format": "jsonl_object"
                                    }
                                    if self._add_chunk(content, metadata):
                                        chunks_added += 1
                        except:
                            continue
            
            logger.info(f"Added {chunks_added} chunks from {filename}")
            
        except Exception as e:
            logger.error(f"Error processing JSON file {filename}: {str(e)}")
    
    def _process_csv_file(self, file_path: str) -> None:
        """Process a CSV file and extract chunks"""
        filename = os.path.basename(file_path)
        chunks_added = 0
        
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Check column structure for common formats
            columns = df.columns.tolist()
            
            if 'question' in columns and 'answer' in columns:
                logger.info(f"Processing Q&A format CSV")
                
                # Extract Q&A pairs
                for _, row in df.iterrows():
                    if pd.notna(row['question']):
                        content = f"Question: {row['question']}"
                        if pd.notna(row['answer']):
                            content += f"\nAnswer: {row['answer']}"
                        else:
                            content += "\nAnswer: [No answer provided]"
                            
                        metadata = {
                            "source": filename,
                            "format": "qa_pair"
                        }
                        if self._add_chunk(content, metadata):
                            chunks_added += 1
            
            elif 'Patient' in columns and 'Doctor' in columns:
                logger.info(f"Processing patient-doctor conversation format")
                
                # Extract conversations
                for _, row in df.iterrows():
                    content_parts = []
                    
                    if 'Description' in columns and pd.notna(row['Description']):
                        content_parts.append(f"Context: {row['Description']}")
                    
                    if pd.notna(row['Patient']):
                        content_parts.append(f"Patient: {row['Patient']}")
                    
                    if pd.notna(row['Doctor']):
                        content_parts.append(f"Doctor: {row['Doctor']}")
                    
                    if content_parts:
                        content = "\n".join(content_parts)
                        metadata = {
                            "source": filename,
                            "format": "patient_doctor_conversation"
                        }
                        if self._add_chunk(content, metadata):
                            chunks_added += 1
            
            else:
                # Generic CSV format
                logger.info(f"Processing generic CSV format")
                
                for _, row in df.iterrows():
                    content_parts = []
                    
                    for col in columns:
                        if pd.notna(row[col]):
                            content_parts.append(f"{col}: {row[col]}")
                    
                    if content_parts:
                        content = "\n".join(content_parts)
                        metadata = {
                            "source": filename,
                            "format": "generic_csv"
                        }
                        if self._add_chunk(content, metadata):
                            chunks_added += 1
            
            logger.info(f"Added {chunks_added} chunks from {filename}")
            
        except Exception as e:
            logger.error(f"Error processing CSV file {filename}: {str(e)}")
    
    def _process_pdf_file(self, file_path: str) -> None:
        """Process a PDF file using sliding window chunking"""
        filename = os.path.basename(file_path)
        chunks_added = 0
        
        try:
            with open(file_path, 'rb') as file:
                # Extract text based on PyPDF2 version
                try:
                    pdf_reader = PyPDF2.PdfReader(file)
                    num_pages = len(pdf_reader.pages)
                except AttributeError:
                    pdf_reader = PyPDF2.PdfFileReader(file)
                    num_pages = pdf_reader.numPages
                
                logger.info(f"Processing PDF with {num_pages} pages")
                
                # Extract full text
                full_text = ""
                
                for page_num in range(num_pages):
                    try:
                        # Extract text based on PyPDF2 version
                        try:
                            page = pdf_reader.pages[page_num]
                            text = page.extract_text()
                        except AttributeError:
                            page = pdf_reader.getPage(page_num)
                            text = page.extractText()
                        
                        if text.strip():
                            full_text += text + "\n\n"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num}: {str(e)}")
                
                # Create sliding window chunks
                if full_text.strip():
                    chunks = self._create_sliding_window_chunks(full_text)
                    
                    for i, chunk in enumerate(chunks):
                        metadata = {
                            "source": filename,
                            "format": "medical_handbook",
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        }
                        if self._add_chunk(chunk, metadata):
                            chunks_added += 1
                
                logger.info(f"Added {chunks_added} chunks from {filename}")
                
        except Exception as e:
            logger.error(f"Error processing PDF file {filename}: {str(e)}")
    
    def _create_sliding_window_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Create sliding window chunks from text"""
        chunks = []
        start = 0
        
        while start < len(text):
            # Define chunk end
            end = min(start + chunk_size, len(text))
            
            # Find a natural breakpoint (period followed by space)
            if end < len(text):
                breakpoint_window = text[max(end-100, start):end]
                last_period = breakpoint_window.rfind('. ')
                
                if last_period != -1:
                    end = max(end-100, start) + last_period + 2
            
            # Extract the chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = max(start + 1, end - overlap)
        
        return chunks
    
    def save_processed_data(self, output_file: str = "processed_data.json") -> None:
        """Save processed data to a JSON file"""
        output_path = os.path.join(self.data_dir, output_file)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.processed_data, f, indent=2)
            
            logger.info(f"Saved {len(self.processed_data)} processed chunks to {output_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")

if __name__ == "__main__":
    ingestor = UgandaMedicalIngestor()
    ingestor.process_all_files()
    ingestor.save_processed_data()