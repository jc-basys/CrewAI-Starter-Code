from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
import os
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers.fix import OutputFixingParser
from pydantic import BaseModel
from typing import Optional, List
import json


# Step 1: Define Pydantic Object for Output
class ClinicalIndication(BaseModel):
    id: str
    text: str
    reference: str

class ClinicalIndications(BaseModel):
    clinical_indications: List[ClinicalIndication]

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

parser = PydanticOutputParser(pydantic_object=ClinicalIndications)
fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=parse_llm)

# CPT Parser
cpt_parser = PydanticOutputParser(pydantic_object=CPTCodes)
cpt_fixing_parser = OutputFixingParser.from_llm(parser=cpt_parser, llm=parse_llm)


# Step 2: Create the agents
clinical_agent = Agent(
    role="Clinical Indications Extractor",
    goal="Extract clinical indications with ID, text, and reference from the policy document",
    backstory="You are trained to identify and extract clinical indications from medical policy documents, including their unique identifiers, descriptive text, and reference information.",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

cpt_agent = Agent(
    role="CPT Codes Extractor",
    goal="Extract all CPT codes in the document",
    backstory="You are trained to identify and extract CPT (Current Procedural Terminology) codes from medical policy documents, including their text values and reference locations.",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Step 3: Create the tasks
clinical_task = Task(
    name="clinical_indications",
    description=(
        "You are given a policy document:\n\n"
        "{document}\n\n"
        "Extract all clinical indications from the document. "
        "For each clinical indication, identify:\n"
        "- A unique ID (format: ci_XXX where XXX is a sequential number)\n"
        "- The clinical indication text\n"
        "- A reference identifier (format: ref_XXX where XXX is a sequential number)\n\n"
        "Return the result in this format:\n"
        '''{
            "clinical_indications": [
                {
                    "id": "ci_001",
                    "text": "Persistent low back pain not responding to conservative treatment for 6 weeks.",
                    "reference": "ref_008"
                },
                {
                    "id": "ci_002",
                    "text": "Suspected spinal tumor or infection based on preliminary exam.",
                    "reference": "ref_009"
                }
            ]
        }'''
    ),
    expected_output='''{
        "clinical_indications": [
            {
                "id": "string (format: ci_XXX)",
                "text": "string (clinical indication description)",
                "reference": "string (format: ref_XXX)"
            }
        ]
    }''',
    agent=clinical_agent
)

cpt_task = Task(
    name="cpt_codes",
    description=(
        "You are given a policy document:\n\n"
        "{document}\n\n"
        "Extract all CPT codes from the document. "
        "For each CPT code, identify:\n"
        "- The CPT code text (e.g., '72141', '72148')\n"
        "- All reference identifiers where this CPT code appears (format: ref_XXX where XXX is a sequential number)\n\n"
        "Return the result in this format:\n"
        '''{
            "cpt_codes": [
                {
                    "text": "72141",
                    "reference": ["ref_003", "ref_005"]
                },
                {
                    "text": "72148",
                    "reference": ["ref_007"]
                }
            ]
        }'''
    ),
    expected_output='''{
        "cpt_codes": [
            {
                "text": "string (CPT code)",
                "reference": ["string (format: ref_XXX)"]
            }
        ]
    }''',
    agent=cpt_agent
)

# Create the crews
clinical_crew = Crew(
    agents=[clinical_agent],
    tasks=[clinical_task],
    process=Process.sequential,  # Required argument
    verbose=True
)

cpt_crew = Crew(
    agents=[cpt_agent],
    tasks=[cpt_task],
    process=Process.sequential,  # Required argument
    verbose=True
)

# Run the crews
print("=== Clinical Indications Extraction ===")
clinical_results = clinical_crew.kickoff(inputs={"document": json.dumps(test_data, indent=2)})
clinical_parsed = fixing_parser.parse(clinical_results.raw).model_dump()
print(json.dumps(clinical_parsed, indent=2))

print("\n=== CPT Codes Extraction ===")
cpt_results = cpt_crew.kickoff(inputs={"document": json.dumps(test_data, indent=2)})
cpt_parsed = cpt_fixing_parser.parse(cpt_results.raw).model_dump()
print(json.dumps(cpt_parsed, indent=2))
