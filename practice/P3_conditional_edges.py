from __future__ import annotations

import asyncio
import logging
import operator
from typing import Annotated, Literal, Optional, TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("retry_graph")

# Максимальное количество попыток достучаться до Бюро
MAX_BUREAU_RETRIES = 2

# --------------------------------------------------------------------------
# 1. Доменные модели
# --------------------------------------------------------------------------

class BureauReport(BaseModel):
    report_id: str
    raw_score: int = Field(ge=300, le=850)


# --------------------------------------------------------------------------
# 2. Схема состояния (State)
# --------------------------------------------------------------------------

class UnderwritingState(TypedDict):
    applicant_national_id: str
    status: Literal["pending", "approved", "rejected", "manual_review"]
    bureau_report: Optional[BureauReport]
    bureau_retry_count: Annotated[int, operator.add]
    audit_log: Annotated[list[str], operator.add]


# --------------------------------------------------------------------------
# 3. Узлы графа
# --------------------------------------------------------------------------

async def credit_bureau_check_node(state: UnderwritingState) -> dict:
    """Узел запроса к кредитному бюро.
    
    Имитирует нестабильное соединение на основе детерминированной функции от ID.
    """
    national_id = state["applicant_national_id"]
    current_retry = state.get("bureau_retry_count", 0)

    logger.info(
        "credit_bureau_check_node: Запрос к бюро для %s (Попытка №%d)", 
        national_id, current_retry + 1
    )

    if "FAIL_ONCE" in national_id and current_retry == 0:
        logger.warning("⚠️ Кредитное бюро временно недоступно (Временный сбой сети)")
        return {
            "bureau_report": None,
            "bureau_retry_count": 1,
            "audit_log": [f"Попытка #{current_retry + 1}: Бюро вернуло 503 Error"]
        }

    if "FAIL_FATAL" in national_id:
        logger.error("🚨 Кредитное бюро недоступно критически (Фатальный сбой)")
        return {
            "bureau_report": None,
            "bureau_retry_count": 1,
            "audit_log": [f"Попытка #{current_retry + 1}: Бюро стабильно не отвечает"]
        }
    
    logger.info("✅ Успешный ответ от Кредитного бюро получен.")
    return {
        "bureau_report": BureauReport(report_id=f"BUR-{national_id}", raw_score=710),
        "bureau_retry_count": 1,
        "audit_log": [f"Попытка #{current_retry + 1}: Успешно получен отчет Бюро"]
    }


async def aggregate_risk_and_route_node(
        state: UnderwritingState
) -> Command[Literal["credit_bureau_check", "auto_approve_node", "manual_review_node", "auto_reject_node"]]:
    """Управляющий узел-маршрутизатор (Оркестратор).
    
    Анализирует состояние и принимает решение о следующем шаге через Command(goto=...).
    """
    report = state.get("bureau_report")
    retries = state.get("bureau_retry_count", 0)

    logger.info("aggregate_node: Анализ состояния. Всего попыток сделано: %d", retries)

    if report is None:
        if retries < MAX_BUREAU_RETRIES:
            logger.info("-> Отчета нет, лимит попыток (%d/%d) не исчерпан. Идем на повторный круг.", retries, MAX_BUREAU_RETRIES)
            return Command(goto="credit_bureau_check")
        else:
            logger.error("-> 🚨 Превышен лимит попыток получения данных Бюро (%d). Эскалация в ручную проверку.", retries)
            return Command(
                update={
                    "status": "manual_review",
                    "audit_log": ["Эскалация: Превышен лимит попыток запроса в Бюро"]
                },
                goto="manual_review_node"
            )
        
    if report.raw_score >= 700:
        return Command(update={"status": "approved"}, goto="auto_approve_node")
    elif report.raw_score < 500:
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
# 4. Сборка графа через Command(goto=...)
# --------------------------------------------------------------------------

def build_graph_command():
    builder = StateGraph(UnderwritingState)

    builder.add_node("credit_bureau_check", credit_bureau_check_node)
    builder.add_node("aggregate_risk_and_route", aggregate_risk_and_route_node)
    builder.add_node("auto_approve_node", auto_approve_node)
    builder.add_node("manual_review_node", manual_review_node)
    builder.add_node("auto_reject_node", auto_reject_node)

    builder.add_edge(START, "credit_bureau_check")
    builder.add_edge("credit_bureau_check", "aggregate_risk_and_route")

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
        "bureau_retry_count": 0,
        "audit_log": [],
    }
    print(f"\n=======================================================")
    print(f"🚀 СТАРТ ТЕСТА: {description}")
    print(f"=======================================================")
    
    result = await graph.ainvoke(initial_state)
    
    print(f"\n📊 ИТОГОВОЕ СОСТОЯНИЕ КОРОБКИ:")
    print(f"Итоговый статус: {result['status']}")
    print(f"Количество попыток: {result['bureau_retry_count']}")
    print(f"Наличие отчета: {result['bureau_report'] is not None}")
    print(f"Хроника аудита:")
    for log in result['audit_log']:
        print(f"  - {log}")


async def main():
    # Тест 1: Все работает с первого раза
    await run_test("ID-GOOD-777", "Штатный клиент (Ответ Бюро с 1-й попытки)")

    # Тест 2: Первая попытка — сбой, вторая — успех (то, что вы описали)
    await run_test("ID-FAIL_ONCE", "Временный сбой Бюро (Успех со 2-й попытки)")

    # Тест 3: Обе попытки — сбой, срабатывает лимит MAX_BUREAU_RETRIES = 2
    # Граф принудительно перенаправит коробку в manual_review_node
    await run_test("ID-FAIL_FATAL", "Фатальный сбой Бюро (Исчерпание лимита ретраев -> Эскалация)")

if __name__ == "__main__":
    asyncio.run(main())