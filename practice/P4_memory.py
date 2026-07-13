from __future__ import annotations

import asyncio
import logging
import operator
from typing import Annotated, Literal, Optional, TypedDict

from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("retry_graph")

# Максимальное количество попыток достучаться до Бюро
MAX_RETRIES = 2

# --------------------------------------------------------------------------
# 1. Доменные модели
# --------------------------------------------------------------------------

class BureauReport(BaseModel):
    report_id: str
    raw_score: int = Field(ge=300, le=850)


# --------------------------------------------------------------------------
# 2. Кастомный редьюсер
# --------------------------------------------------------------------------

def merge_risk_signals(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    merged = dict(left)
    for key, value in right.items():
        merged[key] = max(merged.get(key, value), value)
    return merged


# --------------------------------------------------------------------------
# 3. Схема состояния (State)
# --------------------------------------------------------------------------

class UnderwritingState(TypedDict):
    applicant_national_id: str
    status: Literal["pending", "approved", "rejected", "manual_review"]
    bureau_report: Optional[BureauReport]
    fraud_flags: Annotated[list[str], operator.add]
    risk_signals: Annotated[dict[str, float], merge_risk_signals]
    retry_count: Annotated[int, operator.add]
    audit_log: Annotated[list[str], operator.add]


# --------------------------------------------------------------------------
# 4. Имитация внешних API
# --------------------------------------------------------------------------

async def call_credit_bureau_api(national_id: str) -> BureauReport:
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
# 5. Узлы графа
# --------------------------------------------------------------------------

async def credit_bureau_check_node(state: UnderwritingState) -> dict:
    national_id = state["applicant_national_id"]
    logger.info("Вызов credit_bureau_check_node для %s", national_id)
    try:
        report = await call_credit_bureau_api(national_id=national_id)
        normalized_risk = 1.0 - (report.raw_score - 300) / 550
        return {
            "bureau_report": report,
            "risk_signals": {"credit_history_risk": round(normalized_risk, 2)}
        }
    except Exception:
        logger.warning("Сбой при обращении к кредитному бюро")
        return {
            "bureau_report": None,
            "risk_signals": {"bureau_unavailable_penalty": 0.4},
        }

async def fraud_screening_node(state: UnderwritingState) -> dict:
    national_id = state["applicant_national_id"]
    logger.info("Вызов fraud_screening_node для %s", national_id)
    try:
        triggered_rules = await call_fraud_screening_api(national_id=national_id)
        return {
            "fraud_flags": triggered_rules,
            "risk_signals": {"fraud_risk": 0.2 if not triggered_rules else 0.6},
        }
    except Exception:
        logger.warning("Сбой при обращении к фрауду")
        return {
            "fraud_flags": ["ERROR_SCANNING"],
            "risk_signals": {"fraud_risk": 0.5},
        }


# --------------------------------------------------------------------------
# 6. Узел агрегации и управления потоком
# --------------------------------------------------------------------------

async def aggregate_risk_and_route_node(
        state: UnderwritingState
) -> Command[Literal["credit_bureau_check", "fraud_screening", "auto_approve_node", "manual_review_node", "auto_reject_node"]]:
    report = state.get("bureau_report")
    fraud_flags = state.get("fraud_flags", [])
    fraud_flag = fraud_flags[-1] if fraud_flags else None
    retries = state.get("retry_count", 0)
    signals = state.get("risk_signals", {})

    logger.info("aggregate_node: Анализ состояния. Всего попыток сделано: %d", retries)

    if report is None or fraud_flag == "ERROR_SCANNING":
        if retries < MAX_RETRIES:
            logger.info("-> Обнаружен сбой. Лимит попыток (%d/%d) не исчерпан. Идем на ретрай.", retries, MAX_RETRIES)
            return Command(
                update={
                    "retry_count": 1,
                    "audit_log": [f"Попытка {retries + 1} завершилась ошибкой. Ретрай."]
                },
                goto=["credit_bureau_check", "fraud_screening"]
            )
        else:
            logger.error("-> 🚨 Превышен лимит попыток (%d). Эскалация в ручную проверку.", retries)
            return Command(
                update={
                    "status": "manual_review",
                    "audit_log": ["Эскалация: Превышен лимит попыток запросов во внешние сервисы"]
                },
                goto="manual_review_node"
            )
        
    final_risk_score = round(sum(signals.values()) / max(len(signals), 1), 3)

    logger.info("Финальный скор риска: %f", final_risk_score)

    if final_risk_score <= 0.3:
        return Command(update={"status": "approved"}, goto="auto_approve_node")
    elif final_risk_score >= 0.7:
        return Command(update={"status": "rejected"}, goto="auto_reject_node")
    else:
        return Command(update={"status": "manual_review"}, goto="manual_review_node")
    
async def auto_approve_node(state: UnderwritingState) -> dict:
    logger.info("🎉 Узел Авто-Одобрения: Заявка одобрена.")
    return {"audit_log": ["Финальный вердикт: Авто-одобрение"]}

async def manual_review_node(state: UnderwritingState) -> dict:
    logger.info("🧑‍💻 Узел Ручной Проверки: Отправлено андеррайтеру.")
    return {"audit_log": ["Финальный вердикт: Ручной андеррайтинг"]}

async def auto_reject_node(state: UnderwritingState) -> dict:
    logger.info("❌ Узел Авто-Отказа: Заявка отклонена.")
    return {"audit_log": ["Финальный вердикт: Авто-отказ"]}

# --------------------------------------------------------------------------
# 7. Сборка графа через Command(goto=...)
# --------------------------------------------------------------------------

def build_graph_command():
    builder = StateGraph(UnderwritingState)

    builder.add_node("credit_bureau_check", credit_bureau_check_node)
    builder.add_node("fraud_screening", fraud_screening_node)
    builder.add_node("aggregate_risk_and_route", aggregate_risk_and_route_node)
    builder.add_node("auto_approve_node", auto_approve_node)
    builder.add_node("manual_review_node", manual_review_node)
    builder.add_node("auto_reject_node", auto_reject_node)

    builder.add_edge(START, "credit_bureau_check")
    builder.add_edge(START, "fraud_screening")

    builder.add_edge("credit_bureau_check", "aggregate_risk_and_route")
    builder.add_edge("fraud_screening", "aggregate_risk_and_route")

    builder.add_edge("auto_approve_node", END)
    builder.add_edge("manual_review_node", END)
    builder.add_edge("auto_reject_node", END)

    return builder.compile()


# --------------------------------------------------------------------------
# Тестирование сценариев
# --------------------------------------------------------------------------

async def run_test(national_id: str, description: str):
    graph = build_graph_command()
    initial_state: UnderwritingState = {
        "applicant_national_id": national_id,
        "status": "pending",
        "bureau_report": None,
        "fraud_flags": [],
        "risk_signals": {},
        "retry_count": 0,
        "audit_log": [],
    }
    print(f"\n=======================================================")
    print(f"🚀 СТАРТ ТЕСТА: {description}")
    print(f"=======================================================")
    
    result = await graph.ainvoke(initial_state)
    
    print(f"\n📊 ИТОГОВОЕ СОСТОЯНИЕ КОРОБКИ:")
    print(f"Итоговый статус: {result['status']}")
    print(f"Количество попыток ретрая: {result['retry_count']}")
    print(f"Наличие отчета: {result['bureau_report'] is not None}")
    print(f"Хроника аудита:")
    for log in result['audit_log']:
        print(f"  - {log}")


async def main():
    # Тест 1: Всё работает штатно (Финальный риск средний -> Ручная проверка)
    await run_test("ID-GOOD-777", "Штатный клиент (Успех с 1-й попытки)")

    # Тест 2: Имитируем ошибку (в ID передаем слово FAIL). 
    # Сервисы упадут, сработает ретрай, превысит MAX_RETRIES и уйдет в ручную проверку.
    await run_test("ID-FAIL-FATAL", "Сбой сервисов (Исчерпание лимита ретраев -> Эскалация)")

if __name__ == "__main__":
    asyncio.run(main())