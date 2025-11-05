import json
from sqlalchemy.orm import Session
from db.session import get_db_session
from db.models import Lender, LoanProgram, EligibilityMatrixRule, Guideline

def import_data(json_path="data.json"):
    # Open and load JSON data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lenders_data = data.get("lender", [])
    loan_programs_data = data.get("loan_programs", [])

    db: Session = get_db_session()

    try:
        # Insert Lenders
        for lender in lenders_data:
            if not db.query(Lender).filter_by(id=lender["id"]).first():
                db.add(Lender(
                    id=lender["id"],
                    name=lender["name"]
                ))

        db.commit()

        # Insert Loan Programs
        for program in loan_programs_data:
            if not db.query(LoanProgram).filter_by(id=program["id"]).first():
                db.add(LoanProgram(
                    id=program["id"],
                    lender_id=program["lenderId"],
                    name=program["name"],
                    program_code=program.get("programCode"),
                    description=program.get("description"),
                    source_document=program.get("sourceDocument"),
                    min_loan_amount=program.get("minLoanAmount"),
                    max_loan_amount=program.get("maxLoanAmount")
                ))
        db.commit()

        # Insert Eligibility Matrix Rules
        for program in loan_programs_data:
            for rule in program.get("eligibility_matrix_rules", []):
                if not db.query(EligibilityMatrixRule).filter_by(id=rule["id"]).first():
                    db.add(EligibilityMatrixRule(
                        id=rule["id"],
                        loan_program_id=rule["loanProgramId"],
                        min_loan_amount=rule.get("minLoanAmount"),
                        max_loan_amount=rule.get("maxLoanAmount"),
                        min_fico_score=rule.get("minFicoScore"),
                        max_fico_score=rule.get("maxFicoScore"),
                        occupancy_type=rule.get("occupancyType"),
                        loan_purpose=rule.get("loanPurpose"),
                        dscr_value=rule.get("dscrValue"),
                        max_ltv=rule.get("maxLtv"),
                        reserves_months=rule.get("reservesMonths"),
                        notes=rule.get("notes"),
                        source_reference=rule.get("sourceReference")
                    ))
        db.commit()

        # Insert Guidelines
        for program in loan_programs_data:
            for guide in program.get("guidelines", []):
                if not db.query(Guideline).filter_by(id=guide["id"]).first():
                    db.add(Guideline(
                        id=guide["id"],
                        loan_program_id=guide["loanProgramId"],
                        category=guide.get("category"),
                        content=guide.get("content"),
                        source_reference=guide.get("sourceReference")
                    ))
        db.commit()

        print("✅ Data successfully imported into PostgreSQL.")

    except Exception as e:
        db.rollback()
        print("❌ Error while inserting data:", e)
    finally:
        db.close()


if __name__ == "__main__":
    import_data("data.json")
