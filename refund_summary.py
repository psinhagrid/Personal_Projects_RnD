import logging
from typing import List, Dict
import json
from tool_registry import ToolRegistry


tool_registry = ToolRegistry()  # ← Instantiate the registry

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_similarity_results(results: List[Dict]) -> List[Dict]:
    """Process and filter similarity search results to get only relevant ones."""
    relevant_docs = []
    try:
        for doc in results:
            if doc.get('score', 0) >= 0.7:  # Threshold for relevance
                relevant_docs.append({
                    'content': doc.get('content', ''),
                    'file_path': doc.get('file_path', ''),
                    'score': doc.get('score', 0)
                })
    except Exception as e:
        logger.error(f"Error processing similarity results: {str(e)}")
        raise
    
    return relevant_docs

def main(query: str, folder_path: str, tool_registry) -> None:
    """
    Main function to process the query and generate clarification questions.
    """
    try:
        logger.info(f"Starting search for query: '{query}' in folder: {folder_path}")

        # Step 1: Perform similarity search
        similarity_results = tool_registry.invoke_tool(
            "similarity_searcher.tool_similarity_search",
            query=query,
            folder_path=folder_path
        )
        
        logger.info(f"Found {len(similarity_results)} initial results")

        # Step 2: Filter relevant documents
        relevant_docs = process_similarity_results(similarity_results)
        logger.info(f"Filtered to {len(relevant_docs)} relevant documents")

        if not relevant_docs:
            logger.warning("No relevant documents found")
            return

        # Prepare content for clarification questions
        combined_content = "\n".join([doc['content'] for doc in relevant_docs])

        # Step 3: Generate clarification questions
        clarification_questions = tool_registry.invoke_tool(
            "clarification.tool_generate_clarification_questions",
            query=query,
            content=combined_content
        )

        # Step 4: Print results
        print("\n=== Search Results ===")
        for idx, doc in enumerate(relevant_docs, 1):
            print(f"\nDocument {idx}:")
            print(f"File: {doc['file_path']}")
            print(f"Relevance Score: {doc['score']:.2f}")
            print("-" * 50)

        print("\n=== Clarification Questions ===")
        for idx, question in enumerate(clarification_questions, 1):
            print(f"{idx}. {question}")

    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}")
        raise

if __name__ == "__main__":
    # Define the query and folder path
    query = "Refund policy"
    folder_path = "/Users/psinha/Documents/sample_html_cleaned"

    try:
        main(query, folder_path, tool_registry)
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        raise
