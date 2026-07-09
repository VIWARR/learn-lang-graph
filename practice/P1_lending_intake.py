from __future__ import annotations

import asyncio
import asyncio
import logging
from datetime import date
from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field, ValidationError, field_validator

from langgraph.graph import StateGraph, START, END

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("lending_graph")


# --------------------------------------------------------------------------
# 1. Доменные модели: используем Pydantic, т.к. данные приходят "снаружи"
#    и должны быть провалидированы на границе системы.
# --------------------------------------------------------------------------

class LoanApplicant(BaseModel):
    full_name: str = Field(min_length=2, max_length=120)
    birth_date: date
    requested_amount: float = Field(gt=0, le=5_000_000)
    monthly_income: float = Field(ge=0)
    national_id: str = Field(min_length=6, max_length=32)

    @field_validator("birth_date")
    @classmethod
    def applicant_must_be_adult(cls, value: date) -> date:
        age_years = (date.today() - value).days // 365
        if age_years < 18:
            raise ValueError("Заявитель должен быть совершеннолетним")
        return value
    

class ValidationVerdict(BaseModel):
    """Нормализованный результат первичной проверки."""

    is_valid: bool
    debt_to_income_hint: Optional[float] = None
    rejection_reason: Optional[str] = None


# --------------------------------------------------------------------------
# 2. Схема состояния графа.
#    Обратите внимание: State — это НЕ LoanApplicant. Это отдельная,
#    более широкая структура, которая будет расти по мере добавления узлов
#    (скоринг, решение андеррайтера и т.д. появятся в следующих уроках).
# --------------------------------------------------------------------------

class UnderwritingState(TypedDict):
    """Состояние графа андеррайтинга кредитной заявки."""

    raw_payload: dict                       # сырые данные с фронтенда, как пришли
    applicant: Optional[LoanApplicant]       # заполняется узлом intake_node
    verdict: Optional[ValidationVerdict]     # заполняется узлом intake_node
    status: Literal["pending", "validated", "rejected"]