"""
LangChain-based RAG (Retrieval-Augmented Generation) system
Handles document processing, vector search, and response generation using LangChain
"""

import logging
import os
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class LangChainRAGSystem:
    """LangChain-based RAG system for processing queries and generating responses"""

    def __init__(self, config: Dict = None):
        # Configuration with sensible defaults
        self.config = {
            "enable_advanced_calibration": False,
            "enable_statistical_optimization": False,
            "enable_semantic_similarity": True,
            "enable_caching": True,
            "max_cache_size": 5000,
            "enable_performance_monitoring": False,
            "enable_error_tracking": True,
            "calibration_data_required": 50,
        }

        # Override with user config
        if config:
            self.config.update(config)

        # Initialize LangChain components
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-ada-002"
        )

        self.llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4",
            temperature=0.7,
            max_tokens=500,
        )

        # Initialize components
        self.vectorstore = None
        self.retriever = None
        self.conversation_chain = None
        self.documents = []

        # Calibration parameters (can be tuned based on performance)
        self.confidence_calibration = {
            "coverage_thresholds": {"high": 0.8, "medium": 0.6, "low": 0.4},
            "similarity_thresholds": {"redundant": 0.8, "similar": 0.6},
            "complexity_factors": {"simple": 1.1, "complex": 0.9, "out_of_domain": 0.3},
            "temporal_ranges": {
                "revolution_period": (1908, 1920),
                "extended": (1900, 1930),
            },
        }

        # Cache for performance optimization
        self._concept_cache = {}
        self._similarity_cache = {}
        self.max_cache_size = self.config["max_cache_size"]
        self.cache_access_order = []

        # Conversation memory for context tracking
        self.conversation_memory = {}

        # Performance monitoring (optional)
        self.performance_stats = (
            {
                "confidence_accuracy_pairs": [],
                "calibration_history": [],
                "error_counts": {"embedding": 0, "division_by_zero": 0, "other": 0},
            }
            if self.config["enable_performance_monitoring"]
            else None
        )

    async def initialize(self):
        """Initialize the LangChain RAG system with document processing"""
        logger.info("Initializing LangChain RAG system...")

        # Load and process the Mexican Revolution document
        await self.load_documents()
        await self.create_vectorstore()
        await self.setup_retrieval_chain()

        logger.info("LangChain RAG system initialized successfully")

    async def load_documents(self):
        """Load and chunk the Mexican Revolution documents using LangChain"""
        try:
            raw_documents = []

            # Load PDF documents using LangChain's PyPDFLoader
            pdf_files = [
                "data/source_documents/Sr AI Eng_Challenge_Doc.pdf",
            ]

            for pdf_file in pdf_files:
                try:
                    logger.info(f"Loading PDF: {pdf_file}")
                    loader = PyPDFLoader(pdf_file)
                    pdf_documents = loader.load()
                    raw_documents.extend(pdf_documents)
                    logger.info(f"Loaded {len(pdf_documents)} pages from {pdf_file}")
                except Exception as e:
                    logger.warning(f"Failed to load {pdf_file}: {e}")
                    continue

            # Fallback to text file if no PDFs loaded successfully
            if not raw_documents:
                logger.info("No PDFs loaded, falling back to text file")
                try:
                    loader = TextLoader(
                        "data/source_documents/challenge_content.txt", encoding="utf-8"
                    )
                    raw_documents = loader.load()
                    logger.info("Loaded text file as fallback")
                except Exception as e:
                    logger.error(f"Failed to load text file: {e}")
                    raise

            # Split documents using LangChain's RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=50,
                length_function=len,
                separators=["\n\n", "\n", " ", ""],
            )

            self.documents = text_splitter.split_documents(raw_documents)

            # Add metadata to documents
            for i, doc in enumerate(self.documents):
                # Preserve original PDF metadata if available
                original_metadata = doc.metadata.copy()
                doc.metadata.update(
                    {
                        "chunk_id": i,
                        "length": len(doc.page_content),
                        "source": f"Document Section {i + 1}",
                        "original_page": original_metadata.get("page", "unknown"),
                        "original_source": original_metadata.get("source", "unknown"),
                    }
                )

            logger.info(f"Loaded {len(self.documents)} document chunks")

        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise

    async def create_vectorstore(self):
        """Create FAISS vector store using LangChain"""
        try:
            # Create FAISS vector store from documents
            self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)

            logger.info(
                f"Created FAISS vector store with {len(self.documents)} documents"
            )

        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise

    async def setup_retrieval_chain(self):
        """Set up the retrieval and QA chain with conversation memory"""
        try:
            # Create retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 5}
            )

            # Create conversation memory
            memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True, output_key="answer"
            )

            # Create a simple, effective prompt template
            prompt_template = """You are a knowledgeable assistant specializing in the Mexican Revolution. 
Use the following context to answer the question. If this is a follow-up question, build upon the previous context.

Context: {context}

Question: {question}

Instructions:
- Provide accurate, detailed answers about the Mexican Revolution
- If this is a follow-up question (like "expand on that"), build upon what was discussed previously
- Stay focused on the Mexican Revolution and related topics
- Use the provided context to give specific information

Answer:"""

            prompt = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )

            # Create conversation chain
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt},
                return_source_documents=True,
            )

            logger.info("Retrieval and conversation chains setup completed")

        except Exception as e:
            logger.error(f"Error setting up retrieval chain: {e}")
            raise

    def _sync_conversation_memory(
        self, conversation_history: List[Dict] = None, max_exchanges: int = 10
    ):
        """Sync external conversation history with LangChain's memory (optimized)"""
        try:
            self.conversation_chain.memory.clear()

            if not conversation_history:
                return

            # Limit to recent exchanges for performance
            if len(conversation_history) > max_exchanges * 2:
                recent_history = conversation_history[-(max_exchanges * 2) :]
                logger.info(
                    f"Limited conversation history to last {max_exchanges} exchanges"
                )
            else:
                recent_history = conversation_history

            # Group by pairs, handling misaligned conversations
            for i in range(0, len(recent_history) - 1, 2):
                user_entry = recent_history[i]
                ai_entry = recent_history[i + 1]

                # Validate structure
                if (
                    user_entry.get("role") == "user"
                    and ai_entry.get("role") == "assistant"
                ):
                    user_msg = user_entry.get("content", "")
                    ai_msg = ai_entry.get("content", "")
                    self.conversation_chain.memory.chat_memory.add_user_message(
                        user_msg
                    )
                    self.conversation_chain.memory.chat_memory.add_ai_message(ai_msg)
                else:
                    logger.warning(
                        f"Skipping malformed conversation pair at index {i}: "
                        f"user_role={user_entry.get('role')}, ai_role={ai_entry.get('role')}"
                    )

        except Exception as e:
            logger.error(f"Error syncing conversation memory: {e}")
            # Continue with empty memory rather than failing the entire request
            self.conversation_chain.memory.clear()

    async def process_query(
        self, query: str, conversation_history: List[Dict] = None
    ) -> Tuple[str, List[str], float]:
        """Process a user query and generate a response using LangChain (stateless approach)"""
        try:
            # Sync memory from external history (with error recovery)
            self._sync_conversation_memory(conversation_history)

            # Process query with synced memory
            result = self.conversation_chain({"question": query})
            response = result["answer"]
            source_documents = result["source_documents"]

            # Extract source names
            sources = []
            for doc in source_documents:
                source_name = doc.metadata.get("source", "Unknown Source")
                if source_name not in sources:
                    sources.append(source_name)

            # Calculate confidence score
            confidence = self.calculate_confidence(source_documents, response, query)

            return response, sources, confidence

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    def _build_conversation_context(self, conversation_history: List[Dict]) -> str:
        """Build conversation context from history"""
        context_parts = []

        for entry in conversation_history[-4:]:  # Last 4 entries (2 exchanges)
            role = entry.get("role", "")
            content = entry.get("content", "")
            if role == "user":
                context_parts.append(f"User: {content}")
            elif role == "assistant":
                context_parts.append(f"Assistant: {content}")

        return "\n".join(context_parts)

    def calculate_confidence(
        self, source_documents: List[Document], response: str, query: str = None
    ) -> float:
        """
        Calculate confidence score using multi-factor approach
        """
        if not source_documents:
            return 0.0

        return self._calculate_confidence_standard(source_documents, response, query)

    def _calculate_confidence_standard(
        self, source_documents: List[Document], response: str, query: str = None
    ) -> float:
        """Standard confidence calculation - balanced approach"""
        try:
            # Multiple factors for confidence calculation
            factors = []

            # 1. Source coverage
            num_sources = len(source_documents)
            source_coverage = min(1.0, num_sources / 3.0)  # Normalize to 0-1
            factors.append(("source_coverage", source_coverage, 0.3))

            # 2. Response quality
            response_length = len(response)
            response_quality = min(1.0, response_length / 200.0)  # Normalize to 0-1
            factors.append(("response_quality", response_quality, 0.25))

            # 3. Query relevance (simple heuristic)
            if query:
                query_lower = query.lower()
                domain_terms = [
                    "mexican",
                    "revolution",
                    "mexico",
                    "diaz",
                    "madero",
                    "zapata",
                    "villa",
                ]
                domain_relevance = sum(
                    1 for term in domain_terms if term in query_lower
                ) / len(domain_terms)
                factors.append(("domain_relevance", domain_relevance, 0.25))
            else:
                factors.append(("domain_relevance", 0.5, 0.25))

            # 4. Source diversity
            unique_sources = len(
                set(doc.metadata.get("source", "") for doc in source_documents)
            )
            source_diversity = min(1.0, unique_sources / 3.0)
            factors.append(("source_diversity", source_diversity, 0.2))

            # Calculate weighted confidence
            weighted_confidence = sum(factor * weight for _, factor, weight in factors)

            return max(0.0, min(1.0, weighted_confidence))

        except Exception as e:
            if self.config["enable_error_tracking"] and self.performance_stats:
                self.performance_stats["error_counts"]["other"] += 1
            logger.error(f"Error in standard confidence calculation: {e}")
            return 0.5

    @classmethod
    def create_config(cls) -> Dict:
        """Create configuration for the RAG system"""
        return {
            "enable_advanced_calibration": False,
            "enable_statistical_optimization": False,
            "enable_semantic_similarity": True,
            "enable_caching": True,
            "max_cache_size": 5000,
            "enable_performance_monitoring": False,
            "enable_error_tracking": True,
            "calibration_data_required": 50,
        }

    def get_config_summary(self) -> Dict:
        """Get a summary of current configuration"""
        return {
            "features_enabled": {
                "semantic_similarity": self.config["enable_semantic_similarity"],
                "caching": self.config["enable_caching"],
                "performance_monitoring": self.config["enable_performance_monitoring"],
                "advanced_calibration": self.config["enable_advanced_calibration"],
                "statistical_optimization": self.config[
                    "enable_statistical_optimization"
                ],
            },
            "cache_size": self.max_cache_size,
            "calibration_data_required": self.config["calibration_data_required"],
        }
