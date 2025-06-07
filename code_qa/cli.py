"""CLI interface for the Code QA agent."""

import asyncio
from typing import Optional

from code_qa.agents.code_qa_agent import CodeQAAgent

async def async_main():
    """Run the Code QA agent CLI."""
    print("Code Q&A Agent initialized!")
    print("You can ask questions about code in repositories.")
    print("Example: 'Explain the setup.py file in https://github.com/user/repo'")
    print("Type 'quit' to exit.\n")
    
    agent = CodeQAAgent()
    current_repo: Optional[str] = None
    
    while True:
        try:
            user_input = input("\nHuman: ")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            # Check if user provided a new repository URL
            if "github.com" in user_input or "gitlab.com" in user_input or user_input.startswith("http"):
                # Extract repository URL from the input
                parts = user_input.split()
                repo_candidates = [part for part in parts if "github.com" in part or "gitlab.com" in part or part.startswith("http")]
                if repo_candidates:
                    current_repo = repo_candidates[0]
            
            response = await agent.answer_question(user_input, current_repo)
            print(f"\nAgent: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

def main():
    """Entry point for the CLI."""
    asyncio.run(async_main())

if __name__ == "__main__":
    main() 