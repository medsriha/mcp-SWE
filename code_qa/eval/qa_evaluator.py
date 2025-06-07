"""
Evaluation script for CodeQAAgent using grip_qa dataset.
"""

import os
import asyncio
from pathlib import Path
from typing import List, Tuple, Dict
import json
from datetime import datetime
from rouge_score import rouge_scorer
from code_qa.agents.code_qa_agent import CodeQAAgent
from code_qa.config.settings import get_settings

settings = get_settings()

class QAEvaluator:
    def __init__(self, qa_data_dir: str, openai_api_key: str = None):
        """Initialize the evaluator.
        
        Args:
            qa_data_dir: Directory containing Q/A markdown files
            openai_api_key: Optional OpenAI API key
        """
        self.qa_data_dir = Path(qa_data_dir)
        self.agent = CodeQAAgent(openai_api_key=openai_api_key)
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def load_qa_pairs(self) -> List[Tuple[str, str]]:
        """Load all Q/A pairs from the data directory."""
        qa_pairs = []
        
        # Get all question files
        q_files = sorted(self.qa_data_dir.glob("*.q.md"))
        
        for q_file in q_files:
            # Get corresponding answer file
            a_file = q_file.parent / q_file.name.replace(".q.md", ".a.md")
            if not a_file.exists():
                print(f"Warning: No answer file found for {q_file}")
                continue
                
            with open(q_file) as qf, open(a_file) as af:
                question = qf.read().strip()
                answer = af.read().strip()
                qa_pairs.append((question, answer))
                
        return qa_pairs
    
    async def evaluate_single(self, question: str, ground_truth: str) -> Dict:
        """Evaluate a single Q/A pair."""
        # Get agent's response
        agent_response = await self.agent.answer_question(
            question=question,
            repository_url=settings.repo_url
        )
        
        # Calculate ROUGE scores
        scores = self.scorer.score(ground_truth, agent_response)
        
        return {
            "question": question,
            "ground_truth": ground_truth,
            "agent_response": agent_response,
            "metrics": {
                "rouge1": scores["rouge1"].fmeasure,
                "rouge2": scores["rouge2"].fmeasure,
                "rougeL": scores["rougeL"].fmeasure
            }
        }
    
    async def evaluate_all(self) -> List[Dict]:
        """Evaluate all Q/A pairs."""
        qa_pairs = self.load_qa_pairs()
        results = []
        
        for i, (question, answer) in enumerate(qa_pairs, 1):
            try:
                print(f"\nEvaluating Q/A pair {i}/{len(qa_pairs)}...")
                result = await self.evaluate_single(question, answer)
                results.append(result)
            except Exception as e:
                print(f"Error evaluating question: {question[:100]}...")
                print(f"Error: {str(e)}")
                
        return results
    
    def calculate_aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate metrics across all results."""
        if not results:
            return {}
            
        total_metrics = {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0
        }
        
        for result in results:
            for metric, value in result["metrics"].items():
                total_metrics[metric] += value
                
        avg_metrics = {
            metric: value / len(results)
            for metric, value in total_metrics.items()
        }
        
        return avg_metrics

    def generate_report(self, results: List[Dict], aggregate_metrics: Dict) -> str:
        """Generate a detailed evaluation report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = [
            "=" * 80,
            f"CodeQAAgent Evaluation Report - Generated at {timestamp}",
            "=" * 80,
            "\n"
        ]
        
        # Add aggregate metrics section
        report.extend([
            "AGGREGATE METRICS",
            "-" * 20,
            f"Number of Q/A pairs evaluated: {len(results)}",
            "Average ROUGE scores:"
        ])
        for metric, value in aggregate_metrics.items():
            report.append(f"- {metric}: {value:.4f}")
        report.append("\n")
        
        # Add detailed results section
        report.extend([
            "DETAILED RESULTS",
            "-" * 20
        ])
        
        for i, result in enumerate(results, 1):
            report.extend([
                f"\nQ/A Pair #{i}",
                "-" * 40,
                "\nQUESTION:",
                result["question"],
                "\nGROUND TRUTH ANSWER:",
                result["ground_truth"],
                "\nAGENT RESPONSE:",
                result["agent_response"],
                "\nMETRICS:",
                f"- ROUGE-1: {result['metrics']['rouge1']:.4f}",
                f"- ROUGE-2: {result['metrics']['rouge2']:.4f}",
                f"- ROUGE-L: {result['metrics']['rougeL']:.4f}",
                "\n" + "=" * 80 + "\n"
            ])
            
        return "\n".join(report)

async def main():
    """Run the evaluation."""
    # Initialize evaluator
    evaluator = QAEvaluator(
        qa_data_dir=settings.qa_pairs_dir,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Run evaluation
    print("Starting evaluation...")
    results = await evaluator.evaluate_all()
    
    # Calculate aggregate metrics
    aggregate_metrics = evaluator.calculate_aggregate_metrics(results)
    
    # Generate and save detailed report
    report = evaluator.generate_report(results, aggregate_metrics)
    report_path = "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nDetailed report saved to: {report_path}")
    
    # Save raw results as JSON
    output = {
        "results": results,
        "aggregate_metrics": aggregate_metrics
    }
    with open("evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Raw results saved to: evaluation_results.json")
    
    # Print summary metrics
    print("\nAggregate Metrics:")
    for metric, value in aggregate_metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    asyncio.run(main()) 