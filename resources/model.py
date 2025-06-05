import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
temperature = 0.3

class Model:
    def __init__(self):
        self.llm = ChatGroq(
            model = "llama-3.3-70b-versatile",
            temperature=temperature,
            api_key=api_key
        )
    
    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
                ### SCRAPED TEXT FROM WEBSITE:
                {page_data}
                ### INSTRUCTION:
                The scraped text is from the career's page of a website.
                Your job is to extract the job postings and return them in JSON format containing the 
                following keys: `role`, `experience`, `skills` and `description`.
                Only return the valid JSON.
                ### VALID JSON (NO PREAMBLE):    
            """
        )
        
        chain_extract = prompt_extract | self.llm
        response = chain_extract.invoke(input = {'page_data' : cleaned_text})
        
        try:
            json_parser = JsonOutputParser()
            response = json_parser.parse(response.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return response if isinstance(response,list) else [response]
    
    def write_email(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
                ### JOB DESCRIPTION:
                {job_description}

                ### INSTRUCTION:
                You are Abhinav, a business development executive at TwistTech. TwistTech is an AI & Software Consulting company dedicated to facilitating
                the seamless integration of business processes through automated tools. 
                Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
                process optimization, cost reduction, and heightened overall efficiency. 
                Your job is to write a cold email to the client regarding the job mentioned above describing the capability of TwistTech 
                in fulfilling their needs.
                Also add the most relevant ones from the following links in points to showcase TwistTech's portfolio: {link_list}
                Remember you are Abhinav, BDE at TwistTech. 
                Do not provide a preamble.
                ### EMAIL (NO PREAMBLE):
        """
        )
        
        chain_email = prompt_email | self.llm
        response = chain_email.invoke(input={'job_description' : job, 'link_list' : links})
        return response.content 