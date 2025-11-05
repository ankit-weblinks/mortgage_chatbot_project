import json
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from db.session import AsyncSessionFactory
from db.models import Lender, LoanProgram, EligibilityMatrixRule, Guideline


async def import_data(json_path="db/data.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lenders_data = data.get("lender", [])
    loan_programs_data = data.get("loan_programs", [])

    async with AsyncSessionFactory() as session:  # Use async session
        # Insert Lenders
        for lender in lenders_data:
            existing = await session.get(Lender, lender["id"])
            if not existing:
                session.add(Lender(
                    id=lender["id"],
                    name=lender["name"]
                ))
        await session.commit()

        # Insert Loan Programs
        for program in loan_programs_data:
            existing = await session.get(LoanProgram, program["id"])
            if not existing:
                session.add(LoanProgram(
                    id=program["id"],
                    lenderId=program["lenderId"],
                    name=program["name"],
                    programCode=program.get("programCode"),
                    description=program.get("description"),
                    sourceDocument=program.get("sourceDocument"),
                    minLoanAmount=program.get("minLoanAmount"),
                    maxLoanAmount=program.get("maxLoanAmount")
                ))
        await session.commit()

        # Insert Eligibility Matrix Rules
        for program in loan_programs_data:
            for rule in program.get("eligibility_matrix_rules", []):
                existing = await session.get(EligibilityMatrixRule, rule["id"])
                if not existing:
                    
                    # FIX: Check for 'dscrValue' OR 'minDscr' from the JSON.
                    dscr_val = rule.get("dscrValue") or rule.get("minDscr")
                    
                    session.add(EligibilityMatrixRule(
                        id=rule["id"],
                        loanProgramId=rule["loanProgramId"],
                        minLoanAmount=rule.get("minLoanAmount"),
                        maxLoanAmount=rule.get("maxLoanAmount"),
                        minFicoScore=rule.get("minFicoScore"),
                        maxFicoScore=rule.get("maxFicoScore"),
                        occupancyType=rule.get("occupancyType"),
                        loanPurpose=rule.get("loanPurpose"),
                        
                        # FIX: Cast the value to a string if it's not None.
                        dscrValue=str(dscr_val) if dscr_val is not None else None,
                        
                        maxLtv=rule.get("maxLtv"),
                        reservesMonths=rule.get("reservesMonths"),
                        notes=rule.get("notes"),
                    ))
        await session.commit()

        # Insert Guidelines
        for program in loan_programs_data:
            for guide in program.get("guidelines", []):
                existing = await session.get(Guideline, guide["id"])
                if not existing:
                    session.add(Guideline(
                        id=guide["id"],
                        loanProgramId=guide["loanProgramId"],
                        category=guide.get("category"),
                        content=guide.get("content"),
                        sourceReference=guide.get("sourceReference")
                    ))
        await session.commit()

        print("✅ Data successfully imported into PostgreSQL (async).")


if __name__ == "__main__":
    asyncio.run(import_data())