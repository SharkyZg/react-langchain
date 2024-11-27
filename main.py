from typing import Union, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool, tool

# Load environment variables
load_dotenv()

# Define the tool
@tool
def get_text_length(text: str) -> int:
    """Returns the length of a string by characters."""
    text = text.strip("\n")
    return len(text)

if __name__ == "__main__":
    # Define the tools list
    tools = [get_text_length]

    # Define the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Initialize the agent with Zero-Shot ReAct Agent and the custom prompt
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )

    # Run the agent with the question
    res = agent.run({"input": "What is the length of 'Hello, World!'? Give me just the number."})

    print(f"The final answer is: {res.strip()}")