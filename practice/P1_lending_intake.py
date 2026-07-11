from __future__ import annotations

import os
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


class RejectionHistoryEntry(BaseModel):
    """Запись в истории отказов кредитного конвейера."""
    attempt_number: int
    rejection_reason: str


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
    attempt_number: int
    rejection_history: list[RejectionHistoryEntry]


# --------------------------------------------------------------------------
# 3. Узел графа.
#    Контракт узла: принял срез State -> вернул ЧАСТИЧНЫЙ dict с апдейтом.
#    Узел НИКОГДА не пишет в state["..."] напрямую.
# --------------------------------------------------------------------------

async def intake_validation_node(state: UnderwritingState) -> dict:
    """Валидирует данные заявителя, контролирует лимит попыток и ведет историю."""
    logger.info("intake_validation_node: старт валидации заявки")

    current_attempt = state.get("attempt_number", 1)
    if state.get("status") == "rejected":
        current_attempt += 1

    logger.info("Текущая попытка подачи заявки: %d", current_attempt)

    current_history = list(state.get("rejection_history", []))

    if current_attempt > 3:
        logger.warning("Попытка %d заблокирована: превышен лимит", current_attempt)
        reason = "Превышено число попыток подачи заявки"
        current_history.append(
            RejectionHistoryEntry(attempt_number=current_attempt, rejection_reason=reason)
        )
        return {
            "attempt_number": current_attempt,
            "rejection_history": current_history,
            "verdict": ValidationVerdict(is_valid=False, rejection_reason=reason),
            "status": "rejected",
        }
    
    try:
        applicant = LoanApplicant.model_validate(state["raw_payload"])
    except ValidationError as exc:
        logger.warning("Заявка отклонена на этапе валидации: %s", exc)
        reason = f"Некорректные входные данные: {exc.error_count()} ошибок"
        current_history.append(
            RejectionHistoryEntry(attempt_number=current_attempt, rejection_reason=reason)
        )
        return {
            "attempt_number": current_attempt,
            "rejection_history": current_history,
            "verdict": ValidationVerdict(is_valid=False, rejection_reason=reason),
            "status": "rejected",
        } 
    
    annual_income = applicant.monthly_income * 12
    dti_hint = round(applicant.requested_amount / annual_income, 2) if annual_income > 0 else None

    if dti_hint is not None and dti_hint > 10:
        logger.info("Заявка отклонена по предварительному DTI-фильтру: %.2f", dti_hint)
        reason = "Запрошенная сумма несоразмерна доходу заявителя"
        current_history.append(
            RejectionHistoryEntry(attempt_number=current_attempt, rejection_reason=reason)
        )
        return {
            "attempt_number": current_attempt,
            "rejection_history": current_history,
            "applicant": applicant,
            "verdict": ValidationVerdict(
                is_valid=False, debt_to_income_hint=dti_hint, rejection_reason=reason
            ),
            "status": "rejected",
        }
    
    logger.info("Заявка %s прошла первичную валидацию", applicant.national_id)
    return {
        "attempt_number": current_attempt,
        "applicant": applicant,
        "verdict": ValidationVerdict(is_valid=True, debt_to_income_hint=dti_hint),
        "status": "validated",
    }


# --------------------------------------------------------------------------
# 4. Сборка графа
# --------------------------------------------------------------------------

def build_graph():
    builder = StateGraph(UnderwritingState)
    builder.add_node("intake_validation", intake_validation_node)
    builder.add_edge(START, "intake_validation")
    builder.add_edge("intake_validation", END)
    return builder.compile()


# --------------------------------------------------------------------------
# Демонстрация сценария повторных подач (re-submission flow)
# --------------------------------------------------------------------------
async def main() -> None:
    graph = build_graph()

    # Начальное состояние: некорректные данные (заявитель младше 18)
    state: UnderwritingState = {
        "raw_payload": {
            "full_name": "Иван Петров",
            "birth_date": "2015-04-12",  # Ошибка: несовершеннолетний
            "requested_amount": 300_000,
            "monthly_income": 150_000,
            "national_id": "AB1234567",
        },
        "applicant": None,
        "verdict": None,
        "status": "pending",
        "attempt_number": 1,
        "rejection_history": [],
    }

    print("\n--- ПОПЫТКА 1: Несовершеннолетний заявитель ---")
    state = await graph.ainvoke(state)
    print(f"Статус: {state['status']} | Попытка: {state['attempt_number']}")
    print(f"История: {state['rejection_history']}")

    print("\n--- ПОПЫТКА 2: Слишком большой DTI (Сумма 3 000 000 при доходе 15 000) ---")
    # Исправляем дату, но ломаем DTI соотношение
    state["raw_payload"]["birth_date"] = "1990-04-12"
    state["raw_payload"]["requested_amount"] = 3_000_000
    state["raw_payload"]["monthly_income"] = 15_000
    state = await graph.ainvoke(state)
    print(f"Статус: {state['status']} | Попытка: {state['attempt_number']}")
    print(f"История: {state['rejection_history']}")

    print("\n--- ПОПЫТКА 3: Снова ломаем валидацию (короткое имя) ---")
    state["raw_payload"]["full_name"] = "И"
    state = await graph.ainvoke(state)
    print(f"Статус: {state['status']} | Попытка: {state['attempt_number']}")
    print(f"История: {state['rejection_history']}")

    print("\n--- ПОПЫТКА 4: Превышение лимита (даже если данные идеальные) ---")
    state["raw_payload"]["full_name"] = "Иван Петров"
    state["raw_payload"]["requested_amount"] = 300_000
    state["raw_payload"]["monthly_income"] = 150_000
    state = await graph.ainvoke(state)
    print(f"Статус: {state['status']} | Попытка: {state['attempt_number']}")
    print(f"Итоговый вердикт: {state['verdict'].rejection_reason}")
    print(f"Финальная история (3+1 заблокированная): {state['rejection_history']}")


if __name__ == "__main__":
    asyncio.run(main())