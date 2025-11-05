import enum
import uuid
from sqlalchemy import (
    Column, String, DateTime, Enum as SAEnum, Text, ForeignKey,
    Integer, Numeric, UniqueConstraint, Index
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()

# --- Enums ---

class ChatMessageRole(enum.Enum):
    USER = "USER"
    AI = "AI"

class OccupancyType(enum.Enum):
    PRIMARY = "PRIMARY"
    SECOND_HOME = "SECOND_HOME"
    INVESTMENT = "INVESTMENT"

class LoanPurposeType(enum.Enum):
    PURCHASE = "PURCHASE"
    RATE_TERM = "RATE_TERM"
    CASH_OUT = "CASH_OUT"

class GuidelineCategory(enum.Enum):
    LOAN_PURPOSE = "LOAN_PURPOSE"
    EXCEPTIONS = "EXCEPTIONS"
    PREPAYMENT_PENALTY = "PREPAYMENT_PENALTY"
    PRODUCT_TYPES = "PRODUCT_TYPES"
    INTEREST_ONLY = "INTEREST_ONLY"
    LOAN_AMOUNTS = "LOAN_AMOUNTS"
    OCCUPANCY = "OCCUPANCY"
    PROPERTY_TYPES = "PROPERTY_TYPES"
    PROPERTY_RESTRICTIONS = "PROPERTY_RESTRICTIONS"
    CASH_OUT = "CASH_OUT"
    ACREAGE = "ACREAGE"
    APPRAISALS = "APPRAISALS"
    DECLINING_MARKET = "DECLINING_MARKET"
    TRADELINES = "TRADELINES"
    HOUSING_HISTORY = "HOUSING_HISTORY"
    CREDIT_EVENT_SEASONING = "CREDIT_EVENT_SEASONING"
    RESERVES = "RESERVES"
    SELLER_CONCESSIONS = "SELLER_CONCESSIONS"
    GIFT_FUNDS = "GIFT_FUNDS"
    SUBORDINATE_FINANCING = "SUBORDINATE_FINANCING"
    CITIZENSHIP = "CITIZENSHIP"
    HOMEOWNER_EDUCATION = "HOMEOWNER_EDUCATION"
    INELIGIBLE_STATES = "INELIGIBLE_STATES"
    INELIGIBLE_LOCATIONS = "INELIGIBLE_LOCATIONS"
    GEOGRAPHIC_RESTRICTIONS = "GEOGRAPHIC_RESTRICTIONS"
    FIRST_TIME_INVESTOR = "FIRST_TIME_INVESTOR"
    FIRST_TIME_HOMEBUYER = "FIRST_TIME_HOMEBUYER"
    INCOME_DOCUMENTATION = "INCOME_DOCUMENTATION"
    DTI = "DTI"
    ASSET_UTILIZATION = "ASSET_UTILIZATION"
    MISCELLANEOUS = "MISCELLANEOUS"

# --- Tables ---

class User(Base):
    __tablename__ = "user"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True)
    createdAt = Column(DateTime, server_default=func.now())
    conversations = relationship("Conversation", back_populates="user")

class Conversation(Base):
    __tablename__ = "conversation"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    userId = Column(String, ForeignKey("user.id"), nullable=True)
    createdAt = Column(DateTime, server_default=func.now())
    updatedAt = Column(DateTime, onupdate=func.now())
    summary = Column(Text, nullable=True)
    
    user = relationship("User", back_populates="conversations")
    messages = relationship("ChatMessage", back_populates="conversation", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_message"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversationId = Column(String, ForeignKey("conversation.id", ondelete="CASCADE"), nullable=False)
    role = Column(SAEnum(ChatMessageRole), nullable=False)
    content = Column(Text, nullable=False)
    createdAt = Column(DateTime, server_default=func.now())
    
    conversation = relationship("Conversation", back_populates="messages")
    __table_args__ = (Index("idx_conversationId", "conversationId"),)

class Lender(Base):
    __tablename__ = "lender"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, unique=True, nullable=False)
    programs = relationship("LoanProgram", back_populates="lender")

class LoanProgram(Base):
    __tablename__ = "loan_program"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    lenderId = Column(String, ForeignKey("lender.id"), nullable=False)
    name = Column(String, nullable=False)
    programCode = Column(String, unique=True, nullable=True, index=True)
    description = Column(Text, nullable=True)
    sourceDocument = Column(String, nullable=True)
    minLoanAmount = Column(Numeric, nullable=True)
    maxLoanAmount = Column(Numeric, nullable=True)
    
    lender = relationship("Lender", back_populates="programs")
    matrixRules = relationship("EligibilityMatrixRule", back_populates="loanProgram", cascade="all, delete-orphan")
    guidelines = relationship("Guideline", back_populates="loanProgram", cascade="all, delete-orphan")
    
    __table_args__ = (UniqueConstraint("lenderId", "name", name="uq_lender_program"),)

class EligibilityMatrixRule(Base):
    __tablename__ = "eligibility_matrix_rule"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    loanProgramId = Column(String, ForeignKey("loan_program.id", ondelete="CASCADE"), nullable=False)
    
    # Inputs
    minLoanAmount = Column(Numeric, nullable=False)
    maxLoanAmount = Column(Numeric, nullable=False)
    minFicoScore = Column(Integer, nullable=False)
    maxFicoScore = Column(Integer, nullable=True)
    occupancyType = Column(SAEnum(OccupancyType), nullable=False)
    loanPurpose = Column(SAEnum(LoanPurposeType), nullable=False)
    dscrValue = Column(String, nullable=True)
    
    # Outputs
    maxLtv = Column(Numeric, nullable=False)
    reservesMonths = Column(Integer, nullable=True)
    notes = Column(Text, nullable=True)
    
    loanProgram = relationship("LoanProgram", back_populates="matrixRules")
    
    __table_args__ = (
        Index("idx_matrix_programId", "loanProgramId"),
        UniqueConstraint(
            "loanProgramId", "minLoanAmount", "maxLoanAmount", 
            "minFicoScore", "occupancyType", "loanPurpose", "dscrValue",
            name="uq_composite_rule"
        ),
    )

class Guideline(Base):
    __tablename__ = "guideline"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    loanProgramId = Column(String, ForeignKey("loan_program.id", ondelete="CASCADE"), nullable=False)
    category = Column(SAEnum(GuidelineCategory), nullable=False, index=True)
    content = Column(Text, nullable=False)
    sourceReference = Column(String, nullable=True)
    
    loanProgram = relationship("LoanProgram", back_populates="guidelines")
    __table_args__ = (Index("idx_guideline_programId", "loanProgramId"),)
