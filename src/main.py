"""Main entry point for the Acme Dental AI Agent."""

from dotenv import load_dotenv

from src.agent import create_acme_dental_agent


def main():
    load_dotenv()
    agent = create_acme_dental_agent()
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break

        try:
            message = {"role": "user", "content": user_input}
            result = agent.invoke({"messages": [message]})
            messages = result.get("messages", [])
            if not messages:
                print("Agent: No response generated.\n")
                continue
            last_message = messages[-1]
            print(f"Agent: {last_message.content}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
