from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
from json_sample import json_sample  # Ensure this is a string (not a dict)
import os
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers.fix import OutputFixingParser
from pydantic import BaseModel
from typing import Optional
import json


# Step 1: Define Pydantic Object for Output
class PolicyOverview(BaseModel):
    payer_name: Optional[str]
    policy_id: Optional[str]
    version: Optional[str]
    revision_date: Optional[str]
    summary: Optional[str]

# Put in API KEy
os.environ['GEMINI_API_KEY'] = ""
llm = "gemini/gemini-2.5-flash"

parse_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

parser = PydanticOutputParser(pydantic_object=PolicyOverview)
fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=parse_llm)


# Step 2: Create the agent
agent = Agent(
    role="Policy Overview Extractor",
    goal="Extract payer name, policy ID, version, revision date, and summary from the policy document",
    backstory="You are trained to identify policy overview information like payer name and document versioning.",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Step 3: Create the task
task1 = Task(
    name="policy_overview",
    description=(
        "You are given a policy document:\n\n"
        "{document}\n\n"
        "Extract the policy overview fields from the document. "
        "Return the result in this format:\n"
        '''{
            "policy_overview": {
                "title": "string or null"
                "payer_name": "string or null",
                "policy_id": "string or null",
                "version": "string or null",
                "revision_date": "string (ISO date) or null",
                "summary": "string or null"
            }
        }'''
    ),
    expected_output='''{
        "policy_overview": {
            "payer_name": "string or null",
            "policy_id": "string or null",
            "version": "string or null",
            "revision_date": "string (ISO date) or null",
            "summary": "string or null"
        }
    }''',
    agent=agent
)

# Create the crew
crew = Crew(
    agents=[agent],
    tasks=[task1],
    process=Process.sequential,  # Required argument
    verbose=True
)

# Run the crew
results = crew.kickoff(inputs={"document": json_sample})

results = fixing_parser.parse(results.raw).dict()
print("\n=== Final Output ===\n")
print(json.dumps(results, indent=2))
