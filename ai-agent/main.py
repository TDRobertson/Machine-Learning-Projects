import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import sys

# load the environment variables from the .env file
load_dotenv()

# get the API key from the environment variables
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def main():
    # check if the user has provided an input with the script run in the terminal
    if len(sys.argv) > 1:
        user_input = sys.argv[1]
        if user_input == "":
            print("No input provided. Exiting...")
            sys.exit(1) # exit the entire program with a status code of 1
        else:
            messages = [types.Content(role="user", parts=[types.Part(text=user_input)]),]
            print("Generating response...")
            response = client.models.generate_content(model='gemini-2.0-flash-001', 
            contents=messages)
            print(response.text)

            if "--verbose" in sys.argv:
                print(f"User prompt: {user_input}")
                print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
                print(f"Response tokens: {response.usage_metadata.candidates_token_count}")
            else:
                pass

    # if the user has not provided an input, exit the program with a status code of 1
    else:
        print("No input provided. Exiting...")
        sys.exit(1) 

# run the main function if the script is executed directly from the terminal
if __name__ == "__main__":
    main()
