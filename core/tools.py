# tools.py
import os
import json
from typing import List, Optional
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from sqlalchemy.future import select
from sqlalchemy import text
from thefuzz import process
from db.session import AsyncSessionFactory
from db.models import (
    Lender, LoanProgram, Guideline, EligibilityMatrixRule,
    GuidelineCategory, OccupancyType, LoanPurposeType
)

# Ensure the API key is accessible for the new tool
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Private Helper Functions ---

async def _get_db_schema_for_llm() -> str:
    """
    Generates a simplified schema description for the LLM to use.
    This is much simpler than full introspection and safer.
    """
    return """
Here are the available tables and their most important columns:

1.  **lender**
    * `id` (string, Primary Key)
    * `name` (string, Unique): The name of the lending institution.

2.  **loan_program**
    * `id` (string, Primary Key)
    * `lenderId` (string, Foreign Key to lender.id): Which lender offers this program.
    * `name` (string): The name of the loan program (e.g., "DSCR Plus", "Non-QM Select").
    * `programCode` (string, Unique): The lender's code for this program.
    * `description` (text): A brief description of the program.
    * `minLoanAmount` (numeric): The minimum loan amount.
    * `maxLoanAmount` (numeric): The maximum loan amount.

3.  **guideline**
    * `id` (string, Primary Key)
    * `loanProgramId` (string, Foreign Key to loan_program.id): Which program this guideline belongs to.
    * `category` (string): The category of the guideline.
    * `content` (text): The text of the guideline rule.

4.  **eligibility_matrix_rule**
    * `id` (string, Primary Key)
    * `loanProgramId` (string, Foreign Key to loan_program.id): Which program this rule belongs to.
    * `minLoanAmount` (numeric): Minimum loan amount for this rule.
    * `maxLoanAmount` (numeric): Maximum loan amount for this rule.
    * `minFicoScore` (integer): Minimum FICO score for this rule.
    * `maxFicoScore` (integer): Maximum FICO score for this rule.
    * `occupancyType` (string): e.g., 'PRIMARY', 'INVESTMENT'.
    * `loanPurpose` (string): e.g., 'PURCHASE', 'CASH_OUT'.
    * `dscrValue` (string): DSCR value, if applicable.
    * `maxLtv` (numeric): The **output** max Loan-to-Value (LTV) for this rule.
    * `reservesMonths` (integer): The **output** required reserve months.
    * `notes` (text): Specific notes for this rule.

---
**Available Enums:**

* **GuidelineCategory**: [LOAN_PURPOSE, EXCEPTIONS, PREPAYMENT_PENALTY, PRODUCT_TYPES, INTEREST_ONLY, LOAN_AMOUNTS, OCCUPANCY, PROPERTY_TYPES, PROPERTY_RESTRICTIONS, CASH_OUT, ACREAGE, APPRAISALS, DECLINING_MARKET, TRADELINES, HOUSING_HISTORY, CREDIT_EVENT_SEASONING, RESERVES, SELLER_CONCESSIONS, GIFT_FUNDS, SUBORDINATE_FINANCING, CITIZENSHIP, HOMEOWNER_EDUCATION, INELIGIBLE_STATES, INELIGIBLE_LOCATIONS, GEOGRAPHIC_RESTRICTIONS, FIRST_TIME_INVESTOR, FIRST_TIME_HOMEBUYER, INCOME_DOCUMENTATION, DTI, ASSET_UTILIZATION, STATE_SPECIFIC, ESCROWS, SECONDARY_FINANCING, BORROWER_ELIGIBILITY, NON_ARM_LENGTH, DELAYED_FINANCING, LEASE_PURCHASE, DU_RULES, ITIN_SPECIFICS, DSCR_HIGHLIGHTS, SECOND_LIEN_LIMITS, DSCR_MULTI_RULES, DSCR_RULES, MISCELLANEOUS]
* **OccupancyType**: [PRIMARY, SECOND_HOME, INVESTMENT, INVESTOR]
* **LoanPurposeType**: [PURCHASE, RATE_TERM, CASH_OUT, SECOND_LIEN]
"""

async def _find_program_by_name(session, program_name: str) -> Optional[LoanProgram]:
    """
    Finds a loan program using fuzzy string matching.
    """
    CONFIDENCE_THRESHOLD = 85
    
    query = select(LoanProgram.id, LoanProgram.name)
    result = await session.execute(query)
    all_programs = result.fetchall()
    
    if not all_programs:
        return None

    # Create a mapping of {name: id}
    choices = {prog.name: prog.id for prog in all_programs}
    
    best_match = process.extractOne(program_name, choices.keys())
    
    if best_match and best_match[1] >= CONFIDENCE_THRESHOLD:
        program_id = choices[best_match[0]]
        # Fetch the full program object
        query = select(LoanProgram).where(LoanProgram.id == program_id)
        result = await session.execute(query)
        return result.scalar_one_or_none()
    
    return None

# --- Specialized Tools ---

@tool
async def get_available_lenders() -> str:
    """
    Retrieves a list of all available lender names from the database.
    Use this when the user asks "who are your lenders?" or "list all lenders".
    """
    async with AsyncSessionFactory() as session:
        try:
            query = select(Lender.name).order_by(Lender.name)
            result = await session.execute(query)
            lenders = result.scalars().all()
            
            if not lenders:
                return "No lenders found in the database."
            
            return "Available Lenders:\n- " + "\n- ".join(lenders)
        
        except Exception as e:
            return f"Error retrieving lenders: {e}"

@tool
async def get_loan_programs_by_lender(lender_name: str) -> str:
    """
    Retrieves all loan programs offered by a specific lender.
    Use this when the user asks "what programs does [Lender Name] have?"
    
    Args:
        lender_name (str): The name of the lender to search for.
    """
    async with AsyncSessionFactory() as session:
        try:
            query = select(LoanProgram.name, LoanProgram.programCode, LoanProgram.description) \
                    .join(Lender) \
                    .where(Lender.name.ilike(f"%{lender_name}%")) \
                    .order_by(LoanProgram.name)
            
            result = await session.execute(query)
            programs = result.fetchall()
            
            if not programs:
                return f"No loan programs found for a lender matching '{lender_name}'."
            
            # Find the actual lender name from the first result if needed (to confirm)
            # This is a bit complex, let's just use the user's input for now.
            result_str = f"Loan Programs for lender '{lender_name}':\n"
            for prog in programs:
                result_str += f"\n- **{prog.name}** (Code: {prog.programCode})\n"
                result_str += f"  Description: {prog.description}\n"
            
            return result_str
        
        except Exception as e:
            return f"Error retrieving loan programs: {e}"

@tool
async def get_program_guidelines(program_name: str, category: str = None) -> str:
    """
    Retrieves specific guidelines for a given loan program, optionally filtered by category.
    This tool uses fuzzy matching, so the program_name does not need to be exact.
    
    Args:
        program_name (str): The name of the loan program (e.g., "DSCR Plus").
        category (str, optional): The specific category of guideline to fetch.
                                  Must be one of {', '.join([e.name for e in GuidelineCategory])}.
    """
    async with AsyncSessionFactory() as session:
        try:
            program = await _find_program_by_name(session, program_name)
            if not program:
                return f"Could not find a loan program matching '{program_name}'."

            query = select(Guideline.category, Guideline.content) \
                    .where(Guideline.loanProgramId == program.id) \
                    .order_by(Guideline.category)
            
            if category:
                # Validate category
                try:
                    cat_enum = GuidelineCategory[category.upper()]
                    query = query.where(Guideline.category == cat_enum)
                except KeyError:
                    valid_cats = ', '.join([e.name for e in GuidelineCategory])
                    return f"Invalid category '{category}'. Valid categories are: {valid_cats}"
            
            result = await session.execute(query)
            guidelines = result.fetchall()
            
            if not guidelines:
                filter_msg = f" in category '{category}'" if category else ""
                return f"No guidelines found for program '{program.name}'{filter_msg}."

            result_str = f"Guidelines for Program: {program.name}\n"
            current_cat = ""
            for g in guidelines:
                if g.category.name != current_cat:
                    current_cat = g.category.name
                    result_str += f"\n**--- {current_cat} ---**\n"
                result_str += f"- {g.content}\n"
            
            return result_str
        
        except Exception as e:
            return f"Error retrieving guidelines: {e}"

@tool
async def find_eligibility_rules(
    program_name: str, 
    fico_score: Optional[int] = None, 
    loan_amount: Optional[float] = None, 
    occupancy: Optional[str] = None, 
    loan_purpose: Optional[str] = None
) -> str:
    """
    Finds matching eligibility matrix rules (e.g., max LTV, reserves) for a loan program 
    based on a set of criteria.
    This tool uses fuzzy matching, so the program_name does not need to be exact.

    Args:
        program_name (str): The name of the loan program (e.g., "Non-QM Select").
        fico_score (int, optional): The borrower's FICO score.
        loan_amount (float, optional): The loan amount.
        occupancy (str, optional): The occupancy type. 
                                   Must be one of {', '.join([e.name for e in OccupancyType])}.
        loan_purpose (str, optional): The purpose of the loan. 
                                      Must be one of {', '.join([e.name for e in LoanPurposeType])}.
    """
    async with AsyncSessionFactory() as session:
        try:
            program = await _find_program_by_name(session, program_name)
            if not program:
                return f"Could not find a loan program matching '{program_name}'."

            query = select(
                EligibilityMatrixRule.maxLtv, 
                EligibilityMatrixRule.reservesMonths, 
                EligibilityMatrixRule.notes,
                EligibilityMatrixRule.minFicoScore,
                EligibilityMatrixRule.maxFicoScore,
                EligibilityMatrixRule.minLoanAmount,
                EligibilityMatrixRule.maxLoanAmount,
                EligibilityMatrixRule.occupancyType,
                EligibilityMatrixRule.loanPurpose,
                EligibilityMatrixRule.dscrValue
            ).where(EligibilityMatrixRule.loanProgramId == program.id)
            
            # --- Build dynamic filters ---
            filters_applied = [f"Program: {program.name}"]
            
            if fico_score:
                query = query.where(
                    EligibilityMatrixRule.minFicoScore <= fico_score,
                    EligibilityMatrixRule.maxFicoScore >= fico_score
                )
                filters_applied.append(f"FICO >= {fico_score}")
                
            if loan_amount:
                query = query.where(
                    EligibilityMatrixRule.minLoanAmount <= loan_amount,
                    EligibilityMatrixRule.maxLoanAmount >= loan_amount
                )
                filters_applied.append(f"Loan Amount: {loan_amount}")
            
            if occupancy:
                try:
                    occ_enum = OccupancyType[occupancy.upper()]
                    query = query.where(EligibilityMatrixRule.occupancyType == occ_enum)
                    filters_applied.append(f"Occupancy: {occ_enum.name}")
                except KeyError:
                    valid_occs = ', '.join([e.name for e in OccupancyType])
                    return f"Invalid occupancy '{occupancy}'. Valid types are: {valid_occs}"

            if loan_purpose:
                try:
                    lp_enum = LoanPurposeType[loan_purpose.upper()]
                    query = query.where(EligibilityMatrixRule.loanPurpose == lp_enum)
                    filters_applied.append(f"Loan Purpose: {lp_enum.name}")
                except KeyError:
                    valid_lps = ', '.join([e.name for e in LoanPurposeType])
                    return f"Invalid loan purpose '{loan_purpose}'. Valid types are: {valid_lps}"

            result = await session.execute(query)
            rules = result.fetchall()
            
            if not rules:
                return f"No eligibility rules found matching the criteria:\n" + "\n".join(filters_applied)
            
            result_str = f"Found {len(rules)} matching eligibility rule(s) for:\n" + "\n".join(filters_applied) + "\n"
            
            for i, rule in enumerate(rules, 1):
                result_str += f"\n**--- Match {i} ---**\n"
                result_str += f"- **Max LTV**: {rule.maxLtv}%\n"
                result_str += f"- **Reserves**: {rule.reservesMonths} months\n"
                if rule.notes:
                    result_str += f"- **Notes**: {rule.notes}\n"
                
                # Add context of the rule
                context = []
                if rule.minFicoScore or rule.maxFicoScore:
                    context.append(f"FICO: {rule.minFicoScore}-{rule.maxFicoScore}")
                if rule.occupancyType:
                    context.append(f"Occupancy: {rule.occupancyType.name}")
                if rule.loanPurpose:
                    context.append(f"Purpose: {rule.loanPurpose.name}")
                if rule.dscrValue:
                    context.append(f"DSCR: {rule.dscrValue}")
                if context:
                    result_str += f"- *Rule Context*: {'; '.join(context)}\n"
                    
            return result_str

        except Exception as e:
            return f"Error finding eligibility rules: {e}"


# --- Fallback "Backup" Tool ---

@tool
async def query_database_assistant(question: str) -> str:
    """
    Use this tool **ONLY** as a last resort for complex analytical questions
    that the other tools cannot answer.
    This is for questions like:
    - "What is the average max LTV for all programs from 'Lender X'?"
    - "Count all programs that allow 'INVESTMENT' occupancy."
    - "List all lenders and the count of their 'DSCR' programs."
    
    The input must be a complete, natural language question. The tool will
    generate and execute a SQL query.
    
    Args:
        question (str): The full natural language question from the user.
    """
    # 1. Initialize the LLM for SQL generation
    try:
        # Using a powerful model for reliable SQL generation
        llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="openai/gpt-oss-20b")
    except Exception as e:
        return f"Error initializing LLM: {e}"

    # 2. Get the database schema
    db_schema = await _get_db_schema_for_llm()

    # 3. Create the prompt for the LLM to generate SQL
    # Note: We specify PostgreSQL as the dialect, matching the asyncpg driver.
    prompt_template = f"""
    You are an expert PostgreSQL query writer. Given a database schema and a user's question, 
    generate a single, valid PostgreSQL query to answer the question.

    **Database Schema:**
    {db_schema}

    **User Question:**
    {question}

    **Instructions:**
    - **Only output the raw SQL query.**
    - Do not include any explanations, markdown (`sql ...`), or any text other than the SQL query itself.
    - Ensure the query is syntactically correct for PostgreSQL.
    - Use table and column names exactly as they appear in the schema.
    - When comparing strings (like names), use `ILIKE` for case-insensitive matching.
    - Pay close attention to joins. `loan_program.lenderId` joins to `lender.id`. 
    `guideline.loanProgramId` joins to `loan_program.id`. 
    `eligibility_matrix_rule.loanProgramId` joins to `loan_program.id`.

    **SQL Query:**
    """

    # 4. Get the SQL query from the LLM
    try:
        sql_query = llm.invoke(prompt_template).content.strip()
        # Clean up potential markdown formatting
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        sql_query = sql_query.strip().rstrip(';') # Remove trailing semicolon if present
            
    except Exception as e:
        return f"Error generating SQL query: {e}"

    # 5. **Security Check**: Only allow SELECT statements.
    if not sql_query.lstrip().upper().startswith("SELECT"):
        return "Error: For security reasons, only SELECT queries are allowed."

    # 6. Execute the query and return the result
    async with AsyncSessionFactory() as session:
        try:
            # Use sqlalchemy.text() to execute the raw SQL
            query_result = await session.execute(text(sql_query))
            rows = query_result.fetchall()

            if not rows:
                return "The query executed successfully, but returned no results."

            # Format the results into a readable string
            column_names = query_result.keys()
            result_str = "**Query Result:**\n"
            result_str += ", ".join(column_names) + "\n"
            result_str += "-" * (len(result_str) - 20) + "\n"
            for row in rows:
                result_str += ", ".join(map(str, row)) + "\n"
                
            return result_str

        except Exception as e:
            return f"Database error: {e}. The generated query was: {sql_query}"