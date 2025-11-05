import json
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import and_, or_, text
from db import models
from config.settings import settings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- LLM Initialization (Moved from agent.py) ---

# Initialize the Groq Chat LLM
# This is needed here so the query_database_assistant tool can use it.
llm = ChatGroq(
	model="openai/gpt-oss-20b", # Using a model good at following instructions
	groq_api_key=settings.GROQ_API_KEY,
	temperature=0.0 # Low temp for predictable tool use and SQL generation
)

# --- Tool Pydantic Schemas ---

class SearchEligibilityInput(BaseModel):
	"""Input schema for searching the eligibility matrix."""
	program_name: str = Field(description="The name of the loan program to search within.")
	fico_score: Optional[int] = Field(None, description="Borrower's FICO score.")
	loan_purpose: Optional[str] = Field(None, description="The purpose of the loan (e.g., 'PURCHASE', 'CASH_OUT').")
	occupancy_type: Optional[str] = Field(None, description="The property occupancy type (e.g., 'PRIMARY', 'INVESTOR').")
	loan_amount: Optional[float] = Field(None, description="The total loan amount.")

class GetGuidelinesInput(BaseModel):
	"""Input schema for retrieving specific guidelines."""
	program_name: str = Field(description="The name of the loan program.")
	category: Optional[str] = Field(None, description="The specific category of guideline to retrieve (e.g., 'DTI', 'RESERVES').")

class GetProgramsByLenderInput(BaseModel):
	"""Input schema for listing programs for a specific lender."""
	lender_name: str = Field(description="The name of the lender.")

class DatabaseQueryInput(BaseModel):
	"""Input schema for the general database query assistant."""
	user_query: str = Field(description="A clear, natural language question to be translated into a SQL query.")


# --- Tool Functions ---

async def get_all_lenders(db: AsyncSession) -> List[str]:
	"""Retrieves a list of all lender names from the database."""
	result = await db.execute(select(models.Lender.name))
	lenders = result.scalars().all()
	return lenders

async def get_programs_by_lender(db: AsyncSession, lender_name: str) -> List[Dict[str, Any]]:
	"""Retrieves all loan programs associated with a specific lender name."""
	query = (
		select(models.LoanProgram.name, models.LoanProgram.description)
		.join(models.Lender)
		.where(models.Lender.name.ilike(f"%{lender_name}%"))
	)
	result = await db.execute(query)
	programs = result.mappings().all()
	return [dict(p) for p in programs]

async def get_guidelines(db: AsyncSession, program_name: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
	"""Retrieves specific guidelines for a given loan program, optionally filtered by category."""
	query = (
		select(models.Guideline.category, models.Guideline.content, models.Guideline.sourceReference)
		.join(models.LoanProgram)
		.where(models.LoanProgram.name.ilike(f"%{program_name}%"))
	)
    
	if category:
		# Try to match the enum by name
		try:
			category_enum = models.GuidelineCategory[category.upper()]
			query = query.where(models.Guideline.category == category_enum)
		except KeyError:
			# If no enum match, try a text search in the content
			query = query.where(models.Guideline.content.ilike(f"%{category}%"))
            
	result = await db.execute(query)
	guidelines = result.mappings().all()
	return [dict(g) for g in guidelines]

async def search_eligibility_matrix(
	db: AsyncSession, 
	program_name: str, 
	fico_score: Optional[int] = None, 
	loan_purpose: Optional[str] = None, 
	occupancy_type: Optional[str] = None, 
	loan_amount: Optional[float] = None
) -> List[Dict[str, Any]]:
	"""Searches the eligibility matrix for rules matching the given criteria."""
    
	query = (
		select(
			models.EligibilityMatrixRule.minLoanAmount,
			models.EligibilityMatrixRule.maxLoanAmount,
			models.EligibilityMatrixRule.minFicoScore,
			models.EligibilityMatrixRule.maxFicoScore,
			models.EligibilityMatrixRule.occupancyType,
			models.EligibilityMatrixRule.loanPurpose,
			models.EligibilityMatrixRule.dscrValue,
			models.EligibilityMatrixRule.maxLtv,
			models.EligibilityMatrixRule.reservesMonths,
			models.EligibilityMatrixRule.notes
		)
		.join(models.LoanProgram)
		.where(models.LoanProgram.name.ilike(f"%{program_name}%"))
	)
    
	filters = []
	if fico_score:
		filters.append(
			or_(
				models.EligibilityMatrixRule.minFicoScore == None,
				models.EligibilityMatrixRule.minFicoScore <= fico_score
			)
		)
		filters.append(
			or_(
				models.EligibilityMatrixRule.maxFicoScore == None,
				models.EligibilityMatrixRule.maxFicoScore >= fico_score
			)
		)
        
	if loan_amount:
		filters.append(
			or_(
				models.EligibilityMatrixRule.minLoanAmount == None,
				models.EligibilityMatrixRule.minLoanAmount <= loan_amount
			)
		)
		filters.append(
			or_(
				models.EligibilityMatrixRule.maxLoanAmount == None,
				models.EligibilityMatrixRule.maxLoanAmount >= loan_amount
			)
		)
        
	if loan_purpose:
		try:
			purpose_enum = models.LoanPurposeType[loan_purpose.upper()]
			filters.append(
				or_(
					models.EligibilityMatrixRule.loanPurpose == None,
					models.EligibilityMatrixRule.loanPurpose == purpose_enum
				)
			)
		except KeyError:
			pass # Ignore if invalid enum value is passed
            
	if occupancy_type:
		try:
			occupancy_enum = models.OccupancyType[occupancy_type.upper()]
			filters.append(
				or_(
					models.EligibilityMatrixRule.occupancyType == None,
					models.EligibilityMatrixRule.occupancyType == occupancy_enum
				)
			)
		except KeyError:
			pass # Ignore if invalid enum value is passed

	if filters:
		query = query.where(and_(*filters))
        
	result = await db.execute(query)
	rules = result.mappings().all()
	return [dict(r) for r in rules]

# --- Backup SQL Tool ---

DB_SCHEMA_DOCS = """
Available Tables:

1. `lender`
   - Description: Stores mortgage lender information.
   - Columns: `id` (String, PK), `name` (String, Unique)

2. `loan_program`
   - Description: Stores specific loan programs offered by lenders.
   - Columns: `id` (String, PK), `lenderId` (String, FK to `lender.id`), `name` (String), `programCode` (String), `description` (Text), `minLoanAmount` (Numeric), `maxLoanAmount` (Numeric)

3. `eligibility_matrix_rule`
   - Description: Stores specific LTV/FICO/Reserve rules for a loan program.
   - Columns: `id` (String, PK), `loanProgramId` (String, FK to `loan_program.id`), `minLoanAmount` (Numeric), `maxLoanAmount` (Numeric), `minFicoScore` (Integer), `maxFicoScore` (Integer), `occupancyType` (Enum: 'PRIMARY', 'SECOND_HOME', 'INVESTMENT', 'INVESTOR'), `loanPurpose` (Enum: 'PURCHASE', 'RATE_TERM', 'CASH_OUT', 'SECOND_LIEN'), `dscrValue` (String), `maxLtv` (Numeric), `reservesMonths` (Integer), `notes` (Text)

4. `guideline`
   - Description: Stores textual guidelines (e.g., DTI, Reserves) for a loan program.
   - Columns: `id` (String, PK), `loanProgramId` (String, FK to the `loan_program.id`), `category` (Enum: 'DTI', 'RESERVES', 'CREDIT_EVENT_SEASONING', etc.), `content` (Text), `sourceReference` (String)

Relationships:
- `lender` (1) -> (M) `loan_program`
- `loan_program` (1) -> (M) `eligibility_matrix_rule`
- `loan_program` (1) -> (M) `guideline`
"""

async def query_database_assistant(db: AsyncSession, llm_instance, user_query: str) -> str:
	"""
	A backup tool that generates and executes a SQL query based on the user's natural language question
	and the database schema. Only use this if no other specific tool can answer the query.
	"""
    
	# 1. Create a prompt to generate SQL
	sql_generation_prompt = ChatPromptTemplate.from_messages([
		("system", f"""
You are an expert SQL query writer. Given a user question and a database schema,
write a single, syntactically correct PostgreSQL query to answer the question.
Only output the SQL query and nothing else.

Database Schema:
{DB_SCHEMA_DOCS}
"""),
		("human", "{question}")
	])
    
	# 2. Create a chain to generate SQL
	sql_chain = (
		sql_generation_prompt |
		llm_instance |
		StrOutputParser()
	)
    
	try:
		# 3. Generate the SQL query
		sql_query = await sql_chain.ainvoke({"question": user_query})
		sql_query = sql_query.strip().replace("```sql", "").replace("```", "")
		print(f"Generated SQL: {sql_query}") # For logging
        
		# 4. Execute the query
		if not sql_query.lower().startswith("select"):
			return "Error: Only SELECT queries are allowed."
            
		result = await db.execute(text(sql_query))
		rows = result.mappings().all()
        
		if not rows:
			return "Query executed successfully, but no results were found."
            
		# 5. Format and return results
		return json.dumps([dict(r) for r in rows])
        
	except Exception as e:
		print(f"Error in query_database_assistant: {e}")
		return f"Error executing query: {str(e)}. Please check the query syntax and schema."
