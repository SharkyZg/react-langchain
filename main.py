from typing import Union, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import Tool, tool
from langchain import hub



# Load environment variables
load_dotenv()

# Define the tool
@tool
def get_string_length(text: str) -> int:
    """Returns the length of a string by characters."""
    text = text.strip("\n")
    return len(text)

if __name__ == "__main__":
    # Define the tools list
    tools = [get_string_length]

    # Define the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create the prompt using PromptTemplate
    prompt = hub.pull("hwchase17/openai-functions-agent")

    # Initialize the agent using the new method
    agent = create_openai_functions_agent(
        tools=tools,
        llm=llm,
        prompt=prompt
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools)

    # Run the agent with the question
    result = agent_executor.invoke({"input":"What is the length of 'What is the length of word dog 12345 multiplied by cat 2 times?'? Explain your reasoning, are you using any tools? What is the name of the tool?"})

    # Print the final answer
    print(f"The final answer is: {result['output']}")