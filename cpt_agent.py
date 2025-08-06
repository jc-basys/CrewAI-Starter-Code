from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
import os
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers.fix import OutputFixingParser
from pydantic import BaseModel
from typing import Optional, List
import json


# Step 1: Define Pydantic Object for Output
class CPTCode(BaseModel):
    text: str
    reference: List[str]

class CPTCodes(BaseModel):
    cpt_codes: List[CPTCode]

# Load test.json
with open('test.json', 'r') as f:
    test_data = json.load(f)

# Put in API KEy
os.environ['GEMINI_API_KEY'] = "AIzaSyC2Ng1QZfIabv7jN4yKa10c7oVPLdLbUVA"
llm = "gemini/gemini-2.5-flash"

parse_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

parser = PydanticOutputParser(pydantic_object=CPTCodes)
fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=parse_llm)


# Step 2: Create the agent
agent = Agent(
    role="CPT Codes Extractor",
    goal="Extract all CPT codes in the document",
    backstory="You are trained to identify and extract CPT (Current Procedural Terminology) codes from medical policy documents, including their text values and reference locations.",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Step 3: Create the task
task1 = Task(
    name="cpt_codes",
    description=(
        "You are given a policy document:\n\n"
        "{document}\n\n"
        "Extract all CPT codes from the document. CPT codes are 5-digit numeric codes used for medical procedures.\n\n"
        "Instructions:\n"
        "1. Look for 5-digit numeric codes (e.g., 72141, 72148, 73721, 70553)\n"
        "2. These codes are typically found in sections about covered procedures, billing codes, or service descriptions\n"
        "3. For each unique CPT code found, create an entry with:\n"
        "   - The exact 5-digit code as text\n"
        "   - The reference ID of the section where it appears\n"
        "4. If the same CPT code appears in multiple sections, include all reference IDs\n"
        "5. Only include actual CPT codes, not other numeric identifiers\n\n"
        "Return the result in this format:\n"
        '''{
            "cpt_codes": [
                {
                    "text": "72141",
                    "reference": ["ref_002"]
                },
                {
                    "text": "72148",
                    "reference": ["ref_002"]
                }
            ]
        }'''
    ),
    expected_output='''{
        "cpt_codes": [
            {
                "text": "string (5-digit CPT code)",
                "reference": ["string (format: ref_XXX)"]
            }
        ]
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
results = crew.kickoff(inputs={"document": json.dumps(test_data, indent=2)})

results = fixing_parser.parse(results.raw).model_dump()
print("\n=== Final Output ===\n")
print(json.dumps(results, indent=2))
