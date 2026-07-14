from __future__ import annotations

import asyncio
import logging
import operator
import os
import uuid
from typing import Annotated, Literal, Optional, TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("retry_graph")

MAX_RETRIES = 2
DB_PATH = "lending_checkpoints.db"


# --------------------------------------------------------------------------
# Модели данных
# --------------------------------------------------------------------------

class BureauReport(BaseModel):
    report_id: str
    raw_score: int = Field(ge=300, le=850)


# --------------------------------------------------------------------------
# Кастомные редьюсеры
# --------------------------------------------------------------------------

def merge_risk_signals(
    left: dict[str, float] | None, right: dict[str, float] | None
) -> dict[str, float]:
    """
    Берёт максимум по каждому ключу — пессимистичная стратегия объединения сигналов.
    Передача None в качестве right — сигнал полной очистки словаря.
    """
    if right is None:
        return {}
    
    # Защита от None в left
    merged = dict(left) if left is not None else {}
    
    for key, value in right.items():
        merged[key] = max(merged.get(key, value), value)
    return merged


def resettable_list(left: list[str] | None, right: list[str] | None) -> list[str]:
    """
    Редьюсер для списков, поддерживающий сброс.
    Передача None в качестве right — сигнал очистки списка.
    """
    if right is None:
        return []
    base = left if left is not None else []
    return base + right


# --------------------------------------------------------------------------
# Состояние графа
# --------------------------------------------------------------------------

class UnderwritingState(TypedDict):
    applicant_national_id: str
    bureau_request_id: str
    status: Literal["pending", "approved", "rejected", "manual_review"]
    bureau_report: Optional[BureauReport]  # LWW по умолчанию (без Annotated)
    fraud_flags: Annotated[list[str], resettable_list]
    risk_signals: Annotated[dict[str, float] | None, merge_risk_signals]
    retry_count: int  # LWW по умолчанию — кастомный редьюсер не нужен!
    audit_log: Annotated[list[str], operator.add]


# --------------------------------------------------------------------------
# Имитация внешних API
# --------------------------------------------------------------------------

async def call_credit_bureau_api(national_id: str, request_id: str) -> BureauReport:
    await asyncio.sleep(0.5)
    if "FAIL" in national_id:
        raise RuntimeError("Бюро временно недоступно")
    return BureauReport(
        report_id=f"BUR-ID-{national_id}",
        raw_score=300 + (sum(map(ord, national_id)) % 551),
    )


async def call_fraud_screening_api(national_id: str) -> list[str]:
    await asyncio.sleep(0.2)
    if "FAIL" in national_id:
        raise RuntimeError("Сервис фрода временно недоступен")
    return ["device_fingerprint_mismatch"]


# --------------------------------------------------------------------------
# Узлы графа
# --------------------------------------------------------------------------

async def credit_bureau_check_node(state: UnderwritingState) -> dict:
    national_id = state["applicant_national_id"]
    request_id = state.get("bureau_request_id") or str(uuid.uuid5(uuid.NAMESPACE_DNS, national_id))
    try:
        report = await call_credit_bureau_api(national_id=national_id, request_id=request_id)
        normalized_risk = 1.0 - (report.raw_score - 300) / 550
        return {
            "bureau_request_id": request_id,
            "bureau_report": report,
            "risk_signals": {"credit_history_risk": round(normalized_risk, 2)},
        }
    except Exception as exc:
        logger.warning("credit_bureau_check: ошибка запроса — %s", exc)
        return {
            "bureau_request_id": request_id,
            "bureau_report": None,
            "risk_signals": {"bureau_unavailable_penalty": 0.4},
        }


async def fraud_screening_node(state: UnderwritingState) -> dict:
    national_id = state["applicant_national_id"]
    try:
        triggered_rules = await call_fraud_screening_api(national_id=national_id)
        return {
            "fraud_flags": triggered_rules,
            "risk_signals": {"fraud_risk": 0.2 if not triggered_rules else 0.6},
        }
    except Exception as exc:
        logger.warning("fraud_screening: ошибка запроса — %s", exc)
        return {
            "fraud_flags": ["ERROR_SCANNING"],
            "risk_signals": {"fraud_risk": 0.5},
        }


async def routing_decision_node(state: UnderwritingState) -> Command:
    report = state.get("bureau_report")
    fraud_flags = state.get("fraud_flags", [])
    retries = state.get("retry_count", 0)
    signals = state.get("risk_signals") or {}

    has_errors = (report is None) or ("ERROR_SCANNING" in fraud_flags)

    if has_errors:
        if retries < MAX_RETRIES:
            logger.info(
                "-> [Попытка %d/%d] Обнаружен сбой. Автоматический ретрай.",
                retries + 1,
                MAX_RETRIES,
            )
            return Command(
                update={
                    "retry_count": retries + 1,  # Прямой инкремент без редьюсера
                    "audit_log": [f"Ретрай после ошибки, круг {retries + 1}."],
                },
                goto=["credit_bureau_check", "fraud_screening"],
            )

        # Ретраи исчерпаны — передаём управление оператору
        logger.error("-> 🚨 Ретраи исчерпаны! Конвейер на ПАУЗЕ. Ожидаем вмешательства.")

        user_decision = interrupt(
            {
                "reason": "Превышен лимит попыток запросов во внешние сервисы Бюро/Фрода.",
                "current_state": {
                    "applicant_id": state["applicant_national_id"],
                    "bureau_request_id": state.get("bureau_request_id"),
                    "retry_count": retries,
                },
            }
        )

        logger.info("-> 🧑‍💻 Сигнал получен! Новые данные от оператора: %s", user_decision)

        new_id = user_decision["new_national_id"]

        # Прямой и безопасный сброс всех полей
        return Command(
            update={
                "applicant_national_id": new_id,
                "retry_count": 0,                 # Сброс LWW-поля в 0
                "bureau_report": None,            # Сброс LWW-поля в None
                "fraud_flags": None,              # Сигнал редьюсеру resettable_list
                "risk_signals": None,             # Сигнал редьюсеру merge_risk_signals
                "audit_log": [
                    f"Вмешательство оператора: ID изменён на {new_id}. Ретраи сброшены."
                ],
            },
            goto=["credit_bureau_check", "fraud_screening"],
        )

    # Ошибок нет — принимаем финальное решение
    final_risk_score = round(sum(signals.values()) / max(len(signals), 1), 3)
    logger.info("Финальный скор риска: %.3f", final_risk_score)

    if final_risk_score <= 0.3:
        return Command(update={"status": "approved"}, goto="auto_approve_node")
    elif final_risk_score >= 0.7:
        return Command(update={"status": "rejected"}, goto="auto_reject_node")
    else:
        return Command(update={"status": "manual_review"}, goto="manual_review_node")


async def auto_approve_node(state: UnderwritingState) -> dict:
    return {"audit_log": ["Финальный вердикт: Авто-одобрение"]}


async def manual_review_node(state: UnderwritingState) -> dict:
    return {"audit_log": ["Финальный вердикт: Ручной андеррайтинг"]}


async def auto_reject_node(state: UnderwritingState) -> dict:
    return {"audit_log": ["Финальный вердикт: Авто-отказ"]}


# --------------------------------------------------------------------------
# Сборка графа
# --------------------------------------------------------------------------

def build_graph(checkpointer: AsyncSqliteSaver) -> StateGraph:
    builder = StateGraph(UnderwritingState)

    builder.add_node("credit_bureau_check", credit_bureau_check_node)
    builder.add_node("fraud_screening", fraud_screening_node)
    builder.add_node("routing_decision", routing_decision_node)
    builder.add_node("auto_approve_node", auto_approve_node)
    builder.add_node("manual_review_node", manual_review_node)
    builder.add_node("auto_reject_node", auto_reject_node)

    builder.add_edge(START, "credit_bureau_check")
    builder.add_edge(START, "fraud_screening")

    builder.add_edge("credit_bureau_check", "routing_decision")
    builder.add_edge("fraud_screening", "routing_decision")

    builder.add_edge("auto_approve_node", END)
    builder.add_edge("manual_review_node", END)
    builder.add_edge("auto_reject_node", END)

    return builder.compile(checkpointer=checkpointer)


# --------------------------------------------------------------------------
# Вспомогательная функция вывода итогов
# --------------------------------------------------------------------------

def print_result(label: str, result: dict) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print(f"{'=' * 55}")
    print(f"  Статус заявки  : {result.get('status')}")
    print(f"  Ретраев (всего): {result.get('retry_count')}")
    print(f"  Отчёт бюро    : {'получен' if result.get('bureau_report') else 'отсутствует'}")
    print(f"  Флаги фрода   : {result.get('fraud_flags', [])}")
    print("  Аудит-лог:")
    for entry in result.get("audit_log", []):
        print(f"    • {entry}")


# --------------------------------------------------------------------------
# Тестирование сценария «Пауза → Пробуждение»
# --------------------------------------------------------------------------

async def main() -> None:
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as memory:
        graph = build_graph(memory)
        thread_config = {"configurable": {"thread_id": "session-customer-99"}}

        initial_state: UnderwritingState = {
            "applicant_national_id": "ID-FAIL-ONCE",
            "bureau_request_id": "",
            "status": "pending",
            "bureau_report": None,
            "fraud_flags": [],
            "risk_signals": {},
            "retry_count": 0,
            "audit_log": [],
        }

        print("\n=======================================================")
        print("🚀 ЗАПУСК 1: Старт процесса с проблемным ID")
        print("=======================================================")

        result_1 = await graph.ainvoke(initial_state, config=thread_config)
        print_result("СОСТОЯНИЕ ПОСЛЕ ПАУЗЫ", result_1)

        print("\n=======================================================")
        print("🔧 ЗАПУСК 2: Оператор передаёт исправленный ID")
        print("=======================================================")

        result_2 = await graph.ainvoke(
            Command(resume={"new_national_id": "ID-GOOD-777"}),
            config=thread_config,
        )
        print_result("ИТОГОВОЕ СОСТОЯНИЕ ПОСЛЕ ПРОБУЖДЕНИЯ", result_2)


if __name__ == "__main__":
    asyncio.run(main())