"""
RAGAS-based evaluation framework for the Mexican Revolution RAG system
Updated for RAGAS v0.3.2 API
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from ragas.metrics import (
    AnswerCorrectness,
    AnswerSimilarity,
    ContextRecall,
    ContextRelevance,
    Faithfulness,
    ResponseRelevancy,
)

from src.core.langchain_rag_system import LangChainRAGSystem

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """RAGAS-based evaluator for the Mexican Revolution RAG system"""

    def __init__(
        self,
        minimal_test: bool = False,
        selected_metrics: List[str] = None,
    ):
        self.minimal_test = minimal_test
        self.selected_metrics = selected_metrics or []

        # Initialize RAG system
        config = LangChainRAGSystem.create_config()
        self.rag_system = LangChainRAGSystem(config)

        # Test queries for evaluation
        self.test_queries = [
            # Basic factual queries
            "What was the Mexican Revolution?",
            "Who was Porfirio Díaz?",
            "When did the Mexican Revolution start?",
            "Who were the main leaders of the revolution?",
            "What was the Plan of San Luis?",
            # Detailed historical queries
            "What were the main causes of the Mexican Revolution?",
            "How did Francisco Madero contribute to the revolution?",
            "What role did Emiliano Zapata play in the revolution?",
            "What was the significance of the Constitution of 1917?",
            "How did the revolution end?",
            # Complex analytical queries
            "Compare the roles of Zapata and Villa in the revolution",
            "What were the social and economic reforms after the revolution?",
            "How did the Mexican Revolution influence other Latin American countries?",
            "What were the major battles of the Mexican Revolution?",
            "How did the revolution change Mexican society?",
            # Edge cases
            "What is the weather like in Mexico today?",
            "How to cook Mexican food?",
            "What is the capital of France?",
            "Tell me about World War II",
            "What is quantum physics?",
        ]

        # For faster testing, you can use a subset
        self.test_queries_fast = [
            "What was the Mexican Revolution?",
            "Who was Porfirio Díaz?",
        ]

        # For minimal testing to avoid rate limits
        self.test_queries_minimal = [
            "What was the Mexican Revolution?",
        ]

        # Use appropriate test queries based on options
        if self.minimal_test:
            self.test_queries = self.test_queries_minimal

    def get_available_metrics(self) -> List[str]:
        """Get list of available RAGAS metrics"""
        return [
            "Faithfulness",
            "ResponseRelevancy",
            "ContextRelevance",
            "ContextRecall",
            "AnswerCorrectness",
            "AnswerSimilarity",
        ]

    def select_metrics(self) -> List:
        """Select metrics based on configuration"""
        available_metrics = self.get_available_metrics()

        if self.selected_metrics:
            # Use only selected metrics
            valid_metrics = [m for m in self.selected_metrics if m in available_metrics]
            if not valid_metrics:
                logger.warning(
                    f"No valid metrics found in {self.selected_metrics}. Using all metrics."
                )
                return self._create_metrics_list(available_metrics)
            return self._create_metrics_list(valid_metrics)
        else:
            # Default: use all available metrics
            return self._create_metrics_list(available_metrics)

    def _create_metrics_list(self, metric_names: List[str]) -> List:
        """Create RAGAS metric objects from metric names"""
        metrics = []

        for metric_name in metric_names:
            if metric_name == "Faithfulness":
                metrics.append(Faithfulness())
            elif metric_name == "ResponseRelevancy":
                metrics.append(ResponseRelevancy())
            elif metric_name == "ContextRelevance":
                metrics.append(ContextRelevance())
            elif metric_name == "ContextRecall":
                metrics.append(ContextRecall())
            elif metric_name == "AnswerCorrectness":
                metrics.append(AnswerCorrectness())
            elif metric_name == "AnswerSimilarity":
                metrics.append(AnswerSimilarity())

        return metrics

    async def initialize(self):
        """Initialize the evaluator and RAG system"""
        logger.info("Initializing RAGAS evaluator...")
        await self.rag_system.initialize()
        logger.info("RAGAS evaluator initialized successfully")

    async def generate_evaluation_dataset(self) -> Dataset:
        """Generate evaluation dataset with queries and ground truth"""
        logger.info("Generating evaluation dataset...")

        # Ground truth answers for basic queries (simplified for demonstration)
        ground_truth = {
            "What was the Mexican Revolution?": "The Mexican Revolution was an armed movement that began in 1910 to end the dictatorship of Porfirio Díaz. It culminated in 1917 with the new Political Constitution of the United Mexican States.",
            "Who was Porfirio Díaz?": "Porfirio Díaz was the President of Mexico whose long-term dictatorship was the catalyst for the Mexican Revolution. He ruled from 1876 to 1911.",
            "When did the Mexican Revolution start?": "The Mexican Revolution started in 1910.",
            "Who were the main leaders of the revolution?": "The main leaders included Francisco Madero, Emiliano Zapata, Pancho Villa, and Venustiano Carranza.",
            "What was the Plan of San Luis?": "The Plan of San Luis was a political manifesto issued by Francisco Madero in 1910, calling for armed rebellion against the Díaz regime.",
        }

        # Generate responses for all queries
        results = []
        for query in self.test_queries:
            try:
                start_time = time.time()
                response, sources, confidence = await self.rag_system.process_query(
                    query
                )
                response_time = time.time() - start_time

                # Get ground truth if available
                ground_truth_answer = ground_truth.get(query, "")

                # For RAGAS evaluation, we need actual document content, not just source names
                # Get the actual retrieved documents from the RAG system
                try:
                    # Re-run the retrieval to get the actual document content for RAGAS
                    retriever = self.rag_system.vectorstore.as_retriever(
                        search_kwargs={"k": 5}
                    )
                    retrieved_docs = retriever.get_relevant_documents(query)

                    # Extract actual content for RAGAS evaluation
                    contexts = []
                    for doc in retrieved_docs:
                        content = doc.page_content[
                            :500
                        ]  # Limit content length for API efficiency
                        if content not in contexts:  # Avoid duplicates
                            contexts.append(content)

                    # Fallback to source names if content extraction fails
                    if not contexts:
                        contexts = (
                            [str(source) for source in sources]
                            if isinstance(sources, list)
                            else [str(sources)]
                            if sources
                            else []
                        )

                except Exception as e:
                    logger.warning(f"Failed to extract document content for RAGAS: {e}")
                    # Fallback to source names
                    contexts = (
                        [str(source) for source in sources]
                        if isinstance(sources, list)
                        else [str(sources)]
                        if sources
                        else []
                    )

                results.append(
                    {
                        "question": query,
                        "answer": response,
                        "contexts": contexts,
                        "ground_truth": ground_truth_answer,
                        "confidence": confidence,
                        "response_time": response_time,
                    }
                )

                logger.info(
                    f"Processed query: {query[:50]}... (confidence: {confidence:.3f})"
                )

            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                results.append(
                    {
                        "question": query,
                        "answer": "Error processing query",
                        "contexts": [],
                        "ground_truth": ground_truth.get(query, ""),
                        "confidence": 0.0,
                        "response_time": 0.0,
                    }
                )

        # Convert to HuggingFace Dataset format for RAGAS v0.3.2
        # Use only the v0.3.2 expected format to avoid confusion
        dataset_dict = {
            "user_input": [r["question"] for r in results],
            "response": [r["answer"] for r in results],
            "retrieved_contexts": [r["contexts"] for r in results],
            "reference": [r["ground_truth"] for r in results],
        }

        dataset = Dataset.from_dict(dataset_dict)
        logger.info(f"Generated evaluation dataset with {len(results)} queries")

        return dataset, results

    async def run_ragas_evaluation(self, dataset: Dataset) -> Dict:
        """Run RAGAS evaluation with current API"""
        logger.info("Running RAGAS evaluation...")

        try:
            # Import RAGAS components
            from ragas import evaluate

            # Use the new metric selection method
            metrics = self.select_metrics()

            logger.info(f"Starting RAGAS evaluation with {len(dataset)} samples")
            logger.info(
                "Using metrics: " + ", ".join([m.__class__.__name__ for m in metrics])
            )
            logger.info("This may take several minutes...")

            # Add a small delay to avoid overwhelming the API
            import time

            time.sleep(2)

            # Add debug logging to see the actual dataset structure
            logger.info(f"Dataset schema: {dataset.features}")
            logger.info(f"Dataset columns: {list(dataset.column_names)}")
            logger.info(f"Sample data: {dataset[0] if len(dataset) > 0 else 'No data'}")

            # Run RAGAS evaluation using the current API
            results = evaluate(
                dataset=dataset,
                metrics=metrics,
            )

            # Convert results to dictionary format
            evaluation_results = {}
            if hasattr(results, "to_pandas"):
                df = results.to_pandas()
                logger.info(f"Results DataFrame columns: {list(df.columns)}")

                # Debug: Print actual values for all columns
                for column in df.columns:
                    if column not in [
                        "user_input",
                        "response",
                        "retrieved_contexts",
                        "reference",
                    ]:
                        value = df[column].iloc[0] if len(df) > 0 else None
                        logger.info(f"Debug - {column}: {value} (type: {type(value)})")

                # Calculate mean scores for each metric
                for column in df.columns:
                    if column not in [
                        "user_input",
                        "response",
                        "retrieved_contexts",
                        "reference",
                    ]:
                        try:
                            value = df[column].iloc[0] if len(df) > 0 else None
                            if pd.isna(value) or value is None:
                                logger.warning(
                                    f"Metric {column} is NaN/None - setting to 0.0"
                                )
                                evaluation_results[column] = 0.0
                            else:
                                evaluation_results[column] = float(value)
                                logger.info(
                                    f"Metric {column}: {evaluation_results[column]}"
                                )
                        except Exception as e:
                            logger.warning(f"Error processing metric {column}: {e}")
                            evaluation_results[column] = 0.0
            else:
                # Fallback if results format is different
                evaluation_results = (
                    dict(results)
                    if isinstance(results, dict)
                    else {"error": "Unknown results format"}
                )

            logger.info("RAGAS evaluation completed successfully")
            return evaluation_results

        except ImportError as e:
            logger.error(f"Import error - please install required packages: {e}")
            return {
                "error": f"Missing required packages: {e}",
                "faithfulness": 0.0,
                "response_relevancy": 0.0,
                "llm_context_recall": 0.0,
                "llm_context_precision_without_reference": 0.0,
                "answer_correctness": 0.0,
                "semantic_similarity": 0.0,
            }
        except Exception as e:
            logger.error(f"Error in RAGAS evaluation: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "error": str(e),
                "faithfulness": 0.0,
                "response_relevancy": 0.0,
                "llm_context_recall": 0.0,
                "llm_context_precision_without_reference": 0.0,
                "answer_correctness": 0.0,
                "semantic_similarity": 0.0,
            }

    async def run_performance_evaluation(self, results: List[Dict]) -> Dict:
        """Run performance evaluation metrics"""
        logger.info("Running performance evaluation...")

        try:
            # Calculate performance metrics
            total_queries = len(results)
            successful_queries = len([r for r in results if r["confidence"] > 0])

            # Response time statistics
            response_times = [
                r["response_time"] for r in results if r["response_time"] > 0
            ]
            avg_response_time = (
                sum(response_times) / len(response_times) if response_times else 0
            )
            qps = 1.0 / avg_response_time if avg_response_time > 0 else 0

            # Confidence statistics
            confidences = [r["confidence"] for r in results]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            # Domain relevance analysis
            domain_queries = [
                r for r in results if self._is_domain_relevant(r["question"])
            ]
            domain_relevance_rate = (
                len(domain_queries) / total_queries if total_queries > 0 else 0
            )

            performance_metrics = {
                "total_queries": total_queries,
                "successful_queries": successful_queries,
                "success_rate": successful_queries / total_queries
                if total_queries > 0
                else 0,
                "average_response_time": avg_response_time,
                "queries_per_second": qps,
                "average_confidence": avg_confidence,
                "domain_relevance_rate": domain_relevance_rate,
            }

            logger.info("Performance evaluation completed")
            return performance_metrics

        except Exception as e:
            logger.error(f"Error in performance evaluation: {e}")
            return {"error": str(e)}

    def _is_domain_relevant(self, query: str) -> bool:
        """Check if query is relevant to Mexican Revolution domain"""
        query_lower = query.lower()
        domain_terms = [
            "mexican",
            "revolution",
            "mexico",
            "diaz",
            "madero",
            "zapata",
            "villa",
            "carranza",
            "constitution",
            "1910",
            "1917",
        ]
        return any(term in query_lower for term in domain_terms)

    async def generate_evaluation_report(
        self,
        ragas_results: Dict,
        performance_results: Dict,
        detailed_results: List[Dict],
    ) -> Dict:
        """Generate comprehensive evaluation report"""
        logger.info("Generating evaluation report...")

        timestamp = datetime.now().isoformat()

        report = {
            "evaluation_summary": {
                "timestamp": timestamp,
                "ragas_version": "0.3.2",
                "ragas_metrics": ragas_results,
                "performance_metrics": performance_results,
            },
            "detailed_results": {
                "timestamp": timestamp,
                "queries": detailed_results,
            },
        }

        # Save report to file
        import os

        # Ensure the evaluation reports directory exists
        reports_dir = "data/evaluation_reports"
        os.makedirs(reports_dir, exist_ok=True)

        filename = os.path.join(
            reports_dir,
            f"ragas_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation report saved to {filename}")
        return report, filename

    async def run_complete_evaluation(self) -> Tuple[Dict, str]:
        """Run complete evaluation including RAGAS and performance metrics"""
        logger.info("Starting complete evaluation...")

        # Generate evaluation dataset
        dataset, detailed_results = await self.generate_evaluation_dataset()

        # Run RAGAS evaluation
        ragas_results = await self.run_ragas_evaluation(dataset)

        # Run performance evaluation
        performance_results = await self.run_performance_evaluation(detailed_results)

        # Generate report
        report, filename = await self.generate_evaluation_report(
            ragas_results, performance_results, detailed_results
        )

        # Print summary
        self._print_evaluation_summary(ragas_results, performance_results)

        return report, filename

    def _print_evaluation_summary(self, ragas_results: Dict, performance_results: Dict):
        """Print evaluation summary to console"""
        print("\n" + "=" * 60)
        print("RAGAS EVALUATION SUMMARY (v0.3.2)")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        print("RAGAS Metrics:")
        print("-" * 30)
        for metric, value in ragas_results.items():
            if isinstance(value, (int, float)) and metric != "error":
                print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
            elif metric == "error":
                print(f"Error: {value}")

        print()
        print("Performance Metrics:")
        print("-" * 30)
        for metric, value in performance_results.items():
            if isinstance(value, (int, float)):
                if "rate" in metric or "success" in metric:
                    print(f"{metric.replace('_', ' ').title()}: {value:.2%}")
                elif "time" in metric:
                    print(f"{metric.replace('_', ' ').title()}: {value:.2f}s")
                elif "confidence" in metric:
                    print(f"{metric.replace('_', ' ').title()}: {value:.3f}")
                else:
                    print(f"{metric.replace('_', ' ').title()}: {value:.2f}")
            else:
                print(f"{metric.replace('_', ' ').title()}: {value}")

        print("=" * 60)
        if "error" not in ragas_results:
            print("🎉 Evaluation completed successfully!")
        else:
            print("⚠️  Evaluation completed with errors - check logs for details")


async def main():
    """Main function for running evaluation"""
    parser = argparse.ArgumentParser(
        description="RAGAS-based RAG system evaluation (v0.3.2)"
    )
    parser.add_argument("--env-file", default=".env", help="Path to environment file")
    parser.add_argument(
        "--minimal-test",
        action="store_true",
        help="Use only one query for minimal evaluation",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List available RAGAS metrics and exit",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Specify which RAGAS metrics to evaluate (e.g., --metrics Faithfulness ResponseRelevancy). If not specified, all metrics will be used.",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv(args.env_file)

    # Check required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable is required")
        return

    # Handle list metrics option
    if args.list_metrics:
        evaluator = RAGASEvaluator()
        print("Available RAGAS metrics:")
        for metric in evaluator.get_available_metrics():
            print(f"  - {metric}")
        return

    # Initialize evaluator
    evaluator = RAGASEvaluator(
        minimal_test=args.minimal_test,
        selected_metrics=args.metrics,
    )
    await evaluator.initialize()

    # Run evaluation
    try:
        report, filename = await evaluator.run_complete_evaluation()
        print(f"\n✅ Evaluation completed! Report saved to: {filename}")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"\n❌ Evaluation failed: {e}")


if __name__ == "__main__":
    import asyncio

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    asyncio.run(main())
