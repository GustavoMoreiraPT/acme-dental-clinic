"""Simple AI Agent for the Acme Dental Clinic."""

from langchain.agents import create_agent


def create_acme_dental_agent():
    agent = create_agent(
        tools=[],
        system_prompt="You are a helpful AI assistant",
        model="claude-sonnet-4-5-20250929",
    )
    return agent
