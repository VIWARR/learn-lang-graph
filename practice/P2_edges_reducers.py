from __future__ import annotations

import asyncio
import logging
import operator
from typing import Annotated, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("lending_graph")


# --------------------------------------------------------------------------
# 1. Доменные модели
# --------------------------------------------------------------------------

class BureauReport(BaseModel):
    """Ответ от внешнего кредитного бюро."""
    report_id: str
    raw_score: int = Field(ge=300, le=850)


class FraudServiceUnavailableError(Exception):
    """Технический сбой антифрод-сервиса (таймаут, 5xx и т.д.)."""


# --------------------------------------------------------------------------
# 2. Кастомный редьюсер
# --------------------------------------------------------------------------

def merge_risk_signals(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    """Сливает сигналы риска от разных источников, беря наихудший (максимум) по каждому ключу."""
    merged = dict(left)
    for key, value in right.items():
        merged[key] = max(merged.get(key, value), value)
    return merged


# --------------------------------------------------------------------------
# 3. Схема состояния (State)
# --------------------------------------------------------------------------

class UnderwritingState(TypedDict):
    raw_payload: dict
    applicant_national_id: str
    status: Literal["pending", "screened"]
    bureau_report: Optional[BureauReport]
    fraud_flags: Annotated[list[str], operator.add]
    checks_completed: Annotated[list[str], operator.add]
    risk_signals: Annotated[dict[str, float], merge_risk_signals]
    final_risk_score: Optional[float]


# --------------------------------------------------------------------------
# 4. Имитация внешних сервисов
# --------------------------------------------------------------------------

async def call_credit_bureau_api(national_id: str) -> BureauReport:
    await asyncio.sleep(0.3)  # имитация сетевой задержки
    pseudo_score = 300 + (sum(map(ord, national_id)) % 551)
    return BureauReport(report_id=f"BUR-{national_id}", raw_score=pseudo_score)


async def call_fraud_screening_api(national_id: str) -> list[str]:
    await asyncio.sleep(0.2)  # имитация сетевой задержки
    if sum(map(ord, national_id)) % 7 == 0:
        raise FraudServiceUnavailableError("Антифрод-сервис вернул 503")
    if sum(map(ord, national_id)) % 5 == 0:
        return ["device_fingerprint_mismatch"]
    return []

async def call_sanctions_api(national_id: str) -> bool:
    await asyncio.sleep(0.7)
    if sum(map(ord, national_id)) % 3 == 0:
        return True
    return False
        

# --------------------------------------------------------------------------
# 5. Узлы графа (Параллельный уровень)
# --------------------------------------------------------------------------

async def credit_bureau_check_node(state: UnderwritingState) -> dict:
    national_id = state["applicant_national_id"]
    logger.info("credit_bureau_check_node: запрос к бюро для %s", national_id)

    try:
        report = await call_credit_bureau_api(national_id)
    except Exception:
        logger.exception("Сбой при обращении к кредитному бюро")
        # Технический сбой — не роняем весь граф, деградируем в "средний риск"
        return {
            "risk_signals": {"bureau_unavailable_penalty": 0.4},
            "bureau_report": None,
        }

    # Нормализуем raw_score (300-850) в риск-метрику (0.0 = отлично, 1.0 = максимальный риск)
    normalized_risk = 1.0 - (report.raw_score - 300) / 550
    logger.info("Бюро вернуло score=%s (риск=%.2f)", report.raw_score, normalized_risk)

    return {
        "bureau_report": report,
        "risk_signals": {"credit_history_risk": round(normalized_risk, 2)},
    }


async def fraud_screening_node(state: UnderwritingState) -> dict:
    national_id = state["applicant_national_id"]
    logger.info("fraud_screening_node: запрос к антифрод-сервису для %s", national_id)

    try:
        triggered_rules = await call_fraud_screening_api(national_id)
    except FraudServiceUnavailableError:
        logger.warning("Антифрод-сервис недоступен, применяем консервативную оценку")
        return {
            "fraud_flags": ["fraud_service_unavailable"],
            "risk_signals": {"fraud_risk": 0.5},
        }

    fraud_risk = 0.8 if triggered_rules else 0.05
    logger.info("Антифрод-скрининг завершён, триггеров: %d", len(triggered_rules))

    return {
        "fraud_flags": triggered_rules,
        "risk_signals": {"fraud_risk": fraud_risk},
    }

async def sanctions_screening_node(state: UnderwritingState) -> dict:
    """Новый параллельный узел для проверки заявителя по санкционным спискам."""
    national_id = state["applicant_national_id"]
    logger.info("sanctions_screening_node: проверка санкций для %s", national_id)

    try:
        is_sanctioned = await call_sanctions_api(national_id=national_id)

        if is_sanctioned:
            logger.warning("🚨 НАЙДЕНО СОВПАДЕНИЕ В САНКЦИОННОМ СПИСКЕ для %s", national_id)
            risk_signals = {"sanctions_hit": 1.0}
        else:
            risk_signals = {"sanctions_hit": 0.0}
            
        return {
            "risk_signals": risk_signals,
            # Дописываем маркер выполнения в общий стейт
            "checks_completed": ["sanctions"]
        }
    except Exception:
        logger.exception("Сбой сервиса проверки санкций")
        # В случае сбоя санкций выставляем консервативный высокий риск для безопасности
        return {
            "risk_signals": {"sanctions_hit": 1.0},
            "checks_completed": ["sanctions"]
        }
    

# --------------------------------------------------------------------------
# 6. Узел агрегации (Fan-In)
# --------------------------------------------------------------------------

async def aggregate_risk_node(state: UnderwritingState) -> dict:
    """Fan-in узел: срабатывает только после слияния всех ТРЕХ параллельных потоков."""
    
    # Требование 2: Проверка целостности конвейера.
    # Так как редьюсер складывает списки, мы обязаны получить ровно 3 элемента.
    logger.info("Выполненные проверки: %s", state["checks_completed"])
    assert len(state["checks_completed"]) == 3, (
        f"Нарушение целостности: ожидалось 3 выполненных узла, получено {len(state['checks_completed'])}"
    )

    signals = state["risk_signals"]
    
    # Требование 3: Бизнес-логика "Жесткого правила" (Hard Rule).
    # Извлекаем sanctions_hit до расчета средних значений. Попадание в санкционный список
    # имеет абсолютный приоритет над любыми хорошими скорингами.
    if signals.get("sanctions_hit", 0.0) >= 1.0:
        logger.warning("Агрегатор: Активировано жесткое правило. Итоговый риск принудительно равен 1.0.")
        final_score = 1.0
    else:
        # Стандартная формула для штатных клиентов: берем среднее по всем сигналам,
        # исключая sanctions_hit (поскольку он равен 0.0 и размывал бы знаменатель)
        standard_signals = {k: v for k, v in signals.items() if k != "sanctions_hit"}
        final_score = round(sum(standard_signals.values()) / max(len(standard_signals), 1), 3)

    logger.info("aggregate_risk_node: итоговый риск-скор=%.3f", final_score)
    return {"final_risk_score": final_score, "status": "screened"}


# --------------------------------------------------------------------------
# 7. Сборка графа (Fan-Out на 3 узла)
# --------------------------------------------------------------------------

def build_graph():
    builder = StateGraph(UnderwritingState)

    # 1. Регистрируем все три параллельных станка и один финальный
    builder.add_node("credit_bureau_check", credit_bureau_check_node)
    builder.add_node("fraud_screening", fraud_screening_node)
    builder.add_node("sanctions_screening", sanctions_screening_node)
    builder.add_node("aggregate_risk", aggregate_risk_node)

    # 2. Разветвление (Fan-Out): СТАРТ одновременно толкает коробку на ВСЕ ТРИ узла
    builder.add_edge(START, "credit_bureau_check")
    builder.add_edge(START, "fraud_screening")
    builder.add_edge(START, "sanctions_screening")

    # 3. Слияние (Fan-In): Финальный узел ждет сигналы от ВСЕХ ТРЕХ рельс
    builder.add_edge("credit_bureau_check", "aggregate_risk")
    builder.add_edge("fraud_screening", "aggregate_risk")
    builder.add_edge("sanctions_screening", "aggregate_risk")

    builder.add_edge("aggregate_risk", END)
    return builder.compile()


# --------------------------------------------------------------------------
# Демонстрация сценариев
# --------------------------------------------------------------------------

async def run_scenario(national_id: str, description: str):
    graph = build_graph()
    initial_state: UnderwritingState = {
        "raw_payload": {},
        "applicant_national_id": national_id,
        "status": "pending",
        "bureau_report": None,
        "checks_completed": [],
        "fraud_flags": [],
        "risk_signals": {},
        "final_risk_score": None,
    }
    
    print(f"\n--- Запуск сценария: {description} ---")
    result = await graph.ainvoke(initial_state)
    print(f"[Результат] ID: {national_id} | Выполнено проверок: {result['checks_completed']}")
    print(f"[Результат] Все сигналы риска: {result['risk_signals']}")
    print(f"[Результат] Итоговый скор: {result['final_risk_score']}")


async def main() -> None:
    # Сценарий 1: Обычный чистый клиент (будет посчитано среднее)
    # Сумма кодов "AB1234567" % 3 != 0 и % 5 != 0
    await run_scenario("AB1234567", "Чистый клиент (нет санкций, нет фрода)")

    # Сценарий 2: Санкционный клиент (сработает Hard Rule)
    # Подбираем ID так, чтобы сумма кодов делилась на 3. Например "SANCTION1" -> сработает санкция
    await run_scenario("SANCTION1", "Клиент под санкциями (Итоговый риск должен быть строго 1.0)")


if __name__ == "__main__":
    asyncio.run(main())