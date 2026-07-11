# LangGraph Essentials

## Урок 3: Условные ребра (Conditional Edges)

---

### 🧠 1. Концептуальная база & Архитектура

До сих пор наш граф был предсказуем: `START → intake_validation → [параллельные проверки] → aggregate_risk → END`. Топология фиксирована на этапе `.compile()`. Но реальный андеррайтинг так не работает: после расчёта `final_risk_score` система должна **принять решение**, куда направить заявку — автоматически одобрить, отправить на ручную проверку андеррайтеру или автоматически отклонить. Это не "ещё один узел в цепочке" — это **точка ветвления**, где следующий шаг зависит от данных, посчитанных прямо сейчас, в этом узле.

LangGraph даёт для этого два принципиально разных, но взаимозаменяемых инструмента. Разберём оба до мелочей, потому что выбор между ними — это архитектурное решение, а не вопрос вкуса.

#### Подход A: `add_conditional_edges` — маршрутизация как отдельная сущность

```python
builder.add_conditional_edges(
    "aggregate_risk",           # узел-источник
    route_by_decision,          # функция маршрутизации: State -> str
    {                            # path_map: результат функции -> имя узла
        "approve": "auto_approve_node",
        "review": "manual_review_node",
        "reject": "auto_reject_node",
    },
)
```

Здесь `route_by_decision` — это **отдельная, чистая функция**, которая ничего не меняет в состоянии — она только **читает** его и возвращает строку-ключ. Эта строка ищется в `path_map`, который транслирует её в реальное имя узла. Под капотом `.compile()` использует `path_map` для построения полного графа переходов — именно поэтому визуализация графа (`graph.get_graph().draw_mermaid()`) при этом подходе всегда точна: все возможные направления объявлены декларативно, заранее, отдельно от бизнес-логики узла.

**Философия подхода:** узел `aggregate_risk_node` остаётся "глухим" к вопросу маршрутизации — он просто считает риск и возвращает `dict`. Решение "куда дальше" — забота графа, а не узла. Это классическое разделение ответственности (Separation of Concerns): бизнес-вычисление отдельно, диспетчеризация отдельно. Такую `route_by_decision`-функцию тривиально unit-тестировать в изоляции — на вход `State`-словарь, на выход строка, никаких моков LLM или API.

#### Подход B: `Command(goto=...)` — маршрутизация изнутри узла

```python
from langgraph.types import Command

async def aggregate_risk_and_route_node(state: State) -> Command[Literal["auto_approve_node", "manual_review_node", "auto_reject_node"]]:
    score = compute_score(state)
    goto = "auto_approve_node" if score < 0.3 else "manual_review_node"
    return Command(update={"final_risk_score": score}, goto=goto)
```

Здесь узел возвращает не `dict`, а объект `Command`, который **одновременно** несёт (1) обновление состояния (`update`) и (2) инструкцию маршрутизации (`goto`). LangGraph, получив `Command` вместо `dict`, интерпретирует его особым образом: сначала применяет `update` к каналам состояния (в точности как обычный `dict`-возврат — те же правила `LastValue`/редьюсеров из Урока 2), а затем **напрямую** активирует узел, указанный в `goto`, минуя необходимость в отдельном рёбер-объекте между `aggregate_risk` и тремя целевыми узлами.

**Важная деталь для валидации графа:** типовая аннотация `Command[Literal["auto_approve_node", ...]]` — это не просто для IDE. `.compile()` **читает** этот тип, чтобы понять, какие переходы вообще возможны из данного узла — без него граф всё равно будет работать в рантайме (маршрутизация управляется значением `goto`, а не типом), но статическая визуализация и валидация на этапе компиляции не смогут "увидеть" эти рёбра.

#### Когда что выбирать

| Критерий | `add_conditional_edges` | `Command(goto=...)` |
|---|---|---|
| Разделение логики/маршрутизации | Строгое | Слитное |
| Тестируемость routing-функции в изоляции | Высокая | Ниже (внутри узла) |
| Уместность | Бизнес-правила, DAG-подобная маршрутизация по вычисленным полям | Агентные паттерны, multi-agent handoff, когда LLM/узел сам "решает", кому передать управление |
| Количество сущностей в графе | Узел + функция + path_map | Один узел |
| Типичный пример использования | "Если риск > 0.6 → отклонить" | "Supervisor-агент решает, какому специализированному агенту передать запрос" |

**Мой честный совет как практика:** не выбирайте `Command` только потому, что он "современнее" и требует меньше кода. Для чисто детерминированной бизнес-маршрутизации (наш случай с риск-скором) `add_conditional_edges` почти всегда даёт более читаемый и тестируемый граф — маршрутизация видна как декларативная таблица, а не размазана по `if/elif` внутри узла. `Command` раскрывает свою силу там, где **сам факт вычисления** (например, ответ LLM "к какому агенту переключиться") неотделим от факта маршрутизации — то есть именно в мультиагентных хендоффах, к которым мы подойдём в более продвинутых модулях.

---

### 💻 2. Разбор кода (Production-ready пример)

Реализуем **оба подхода** для одной и той же бизнес-задачи, чтобы вы могли сравнить их построчно.

```python
"""
lending_routing.py

Маршрутизация решения по кредитной заявке двумя способами:
  A) add_conditional_edges + отдельная routing-функция
  B) Command(goto=...) прямо из узла

Оба варианта используют идентичную бизнес-логику принятия решения.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Literal, Optional, TypedDict

import operator

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("lending_graph")


# --------------------------------------------------------------------------
# 1. Состояние (final_risk_score считается заранее, как в Уроке 2)
# --------------------------------------------------------------------------

class UnderwritingState(TypedDict):
    applicant_national_id: str
    final_risk_score: Optional[float]
    decision: Optional[Literal["auto_approve", "manual_review", "auto_reject"]]
    status: str
    audit_log: Annotated[list[str], operator.add]


APPROVE_THRESHOLD = 0.3
REJECT_THRESHOLD = 0.7


def classify_risk(score: float) -> Literal["auto_approve", "manual_review", "auto_reject"]:
    """Единая бизнес-функция классификации, переиспользуется в обоих подходах."""
    if score < APPROVE_THRESHOLD:
        return "auto_approve"
    if score >= REJECT_THRESHOLD:
        return "auto_reject"
    return "manual_review"


# --------------------------------------------------------------------------
# 2. Терминальные узлы (общие для обоих графов)
# --------------------------------------------------------------------------

async def auto_approve_node(state: UnderwritingState) -> dict:
    logger.info("Заявка %s одобрена автоматически", state["applicant_national_id"])
    try:
        # В реальной системе — вызов сервиса выдачи кредита
        await asyncio.sleep(0.1)
    except Exception:
        logger.exception("Сбой при инициации выдачи кредита")
        raise
    return {"status": "approved", "audit_log": ["auto_approve_node: одобрено"]}


async def manual_review_node(state: UnderwritingState) -> dict:
    logger.info("Заявка %s направлена андеррайтеру", state["applicant_national_id"])
    try:
        # В реальной системе — постановка задачи в очередь ручной проверки
        await asyncio.sleep(0.1)
    except Exception:
        logger.exception("Сбой при постановке заявки в очередь ручной проверки")
        raise
    return {"status": "pending_review", "audit_log": ["manual_review_node: отправлено андеррайтеру"]}


async def auto_reject_node(state: UnderwritingState) -> dict:
    logger.info("Заявка %s отклонена автоматически", state["applicant_national_id"])
    try:
        # В реальной системе — отправка уведомления заявителю
        await asyncio.sleep(0.1)
    except Exception:
        logger.exception("Сбой при отправке уведомления об отказе")
        raise
    return {"status": "rejected", "audit_log": ["auto_reject_node: отклонено"]}


# --------------------------------------------------------------------------
# 3A. ПОДХОД A: add_conditional_edges
# --------------------------------------------------------------------------

async def aggregate_risk_node_conditional(state: UnderwritingState) -> dict:
    """Узел ТОЛЬКО считает решение, но не маршрутизирует."""
    score = state["final_risk_score"] or 0.0
    decision = classify_risk(score)
    logger.info("Подход A: score=%.2f -> decision=%s", score, decision)
    return {"decision": decision, "audit_log": [f"aggregate_risk_node_conditional: {decision}"]}


def route_by_decision(state: UnderwritingState) -> Literal["approve", "review", "reject"]:
    """Чистая routing-функция: читает state, ничего не меняет."""
    decision = state["decision"]
    if decision == "auto_approve":
        return "approve"
    if decision == "auto_reject":
        return "reject"
    return "review"


def build_graph_conditional_edges():
    builder = StateGraph(UnderwritingState)
    builder.add_node("aggregate_risk", aggregate_risk_node_conditional)
    builder.add_node("auto_approve_node", auto_approve_node)
    builder.add_node("manual_review_node", manual_review_node)
    builder.add_node("auto_reject_node", auto_reject_node)

    builder.add_edge(START, "aggregate_risk")
    builder.add_conditional_edges(
        "aggregate_risk",
        route_by_decision,
        {
            "approve": "auto_approve_node",
            "review": "manual_review_node",
            "reject": "auto_reject_node",
        },
    )
    builder.add_edge("auto_approve_node", END)
    builder.add_edge("manual_review_node", END)
    builder.add_edge("auto_reject_node", END)
    return builder.compile()


# --------------------------------------------------------------------------
# 3B. ПОДХОД B: Command(goto=...)
# --------------------------------------------------------------------------

async def aggregate_risk_and_route_node(
    state: UnderwritingState,
) -> Command[Literal["auto_approve_node", "manual_review_node", "auto_reject_node"]]:
    """Узел считает решение И маршрутизирует за один возврат."""
    score = state["final_risk_score"] or 0.0
    decision = classify_risk(score)
    goto = {
        "auto_approve": "auto_approve_node",
        "manual_review": "manual_review_node",
        "auto_reject": "auto_reject_node",
    }[decision]
    logger.info("Подход B: score=%.2f -> decision=%s -> goto=%s", score, decision, goto)

    return Command(
        update={"decision": decision, "audit_log": [f"aggregate_risk_and_route_node: {decision}"]},
        goto=goto,
    )


def build_graph_command():
    builder = StateGraph(UnderwritingState)
    builder.add_node("aggregate_risk", aggregate_risk_and_route_node)
    builder.add_node("auto_approve_node", auto_approve_node)
    builder.add_node("manual_review_node", manual_review_node)
    builder.add_node("auto_reject_node", auto_reject_node)

    builder.add_edge(START, "aggregate_risk")
    # Обратите внимание: НЕТ явных add_edge между aggregate_risk и тремя
    # терминальными узлами — маршрутизация целиком определяется значением
    # goto, возвращаемым узлом в рантайме.
    builder.add_edge("auto_approve_node", END)
    builder.add_edge("manual_review_node", END)
    builder.add_edge("auto_reject_node", END)
    return builder.compile()


async def main() -> None:
    sample_state: UnderwritingState = {
        "applicant_national_id": "AB1234567",
        "final_risk_score": 0.82,
        "decision": None,
        "status": "screened",
        "audit_log": [],
    }

    graph_a = build_graph_conditional_edges()
    graph_b = build_graph_command()

    result_a = await graph_a.ainvoke(sample_state)
    result_b = await graph_b.ainvoke(sample_state)

    logger.info("Подход A -> статус: %s, лог: %s", result_a["status"], result_a["audit_log"])
    logger.info("Подход B -> статус: %s, лог: %s", result_b["status"], result_b["audit_log"])


if __name__ == "__main__":
    asyncio.run(main())
```

---

### 📊 3. Трассировка Состояния (State Lifecycle)

Возьмём заявку с `final_risk_score = 0.82` (выше `REJECT_THRESHOLD = 0.7`) и сравним оба пути.

**Подход A (`add_conditional_edges`):**

| Шаг | Что происходит | Состояние после шага |
|---|---|---|
| 1 | `aggregate_risk_node_conditional` считает `decision="auto_reject"`, возвращает `dict` | `decision: "auto_reject"` записан через LastValue |
| 2 | Граф **отдельно** вызывает `route_by_decision(state)` над уже слитым состоянием | Функция читает `state["decision"]`, возвращает строку `"reject"` (это НЕ попадает в state — это просто ключ маршрута) |
| 3 | `"reject"` ищется в `path_map` → находится `"auto_reject_node"` | Активируется `auto_reject_node` |
| 4 | `auto_reject_node` выполняется, возвращает `dict` | `status: "rejected"` |

**Подход B (`Command(goto=...)`):**

| Шаг | Что происходит | Состояние после шага |
|---|---|---|
| 1 | `aggregate_risk_and_route_node` считает `decision="auto_reject"` **и** вычисляет `goto="auto_reject_node"` **в одном вызове** | `update` и `goto` формируются одновременно, до какого-либо взаимодействия с графом |
| 2 | LangGraph получает `Command`, применяет `update` к каналам состояния (как обычный `dict`) | `decision: "auto_reject"` записан |
| 3 | LangGraph немедленно активирует узел из `goto`, без промежуточного вызова внешней функции | Активируется `auto_reject_node` |
| 4 | `auto_reject_node` выполняется, возвращает `dict` | `status: "rejected"` |

**Итоговое состояние идентично в обоих случаях** — разница исключительно в **механизме**, а не в результате: подход A тратит один дополнительный "логический хоп" на вызов отдельной routing-функции над уже сохранённым состоянием, подход B принимает решение о маршруте *до* того, как состояние вообще было записано в канал.

---

### ⚠️ 4. Ошибка новичка (Bad Practice vs Good Practice)

Условные рёбра — это ровно то место в графе, где рождаются **циклы**. А цикл без условия останова — это не баг, который тихо испортит данные, это баг, который **гарантированно уронит весь запрос**, и это самая частая причина падений в проде на графах с условной маршрутизацией.

**❌ Bad Practice:**

Представим требование: "если бюро кредитных историй не вернуло данные (`bureau_report is None`), нужно повторно запросить обогащение перед тем, как принимать решение". Наивная реализация через `Command`:

```python
async def aggregate_risk_and_route_node(state: UnderwritingState) -> Command[Literal[...]]:
    if state.get("bureau_report") is None:
        logger.warning("Нет отчёта бюро, отправляем на повторное обогащение")
        # АНТИПАТТЕРН: возвращаемся к узлу, который сам ничего
        # не гарантирует изменить в bureau_report при повторном сбое
        return Command(goto="credit_bureau_check", update={})

    score = state["final_risk_score"] or 0.0
    decision = classify_risk(score)
    ...
```

**Почему это стреляет в проде:** если внешнее кредитное бюро легло надолго (не разовый сбой, а продолжительный инцидент на их стороне), `credit_bureau_check` будет **раз за разом** возвращать `bureau_report=None`, граф будет **раз за разом** отправлять заявку обратно на `aggregate_risk_and_route_node`, который снова увидит `None` и снова отправит на повтор. Ничто в этой логике не ограничивает число попыток. LangGraph такие графы не пускает выполняться бесконечно физически — есть защитный предохранитель `recursion_limit` (по умолчанию **25** супершагов), и вместо зависшего процесса вы получите:

```
langgraph.errors.GraphRecursionError: Recursion limit of 25 reached without hitting a stop condition.
You can increase the limit by setting the `recursion_limit` config key.
```

Это спасает процесс от бесконечного зависания, но **не спасает бизнес-процесс**: заявитель получает технический сбой вместо осмысленного ответа ("ваша заявка направлена на ручную проверку из-за временной недоступности бюро"), а вы получаете исключение без внятного бизнес-контекста в логах, если не залогировали количество попыток отдельно.

**✅ Good Practice:**

Вводим явный счётчик попыток и **детерминированную границу** выхода из цикла:

```python
class UnderwritingState(TypedDict):
    ...
    bureau_retry_count: Annotated[int, operator.add]  # редьюсер: суммируем попытки

MAX_BUREAU_RETRIES = 2

async def aggregate_risk_and_route_node(state: UnderwritingState) -> Command[Literal[...]]:
    if state.get("bureau_report") is None:
        if state["bureau_retry_count"] >= MAX_BUREAU_RETRIES:
            logger.error(
                "Бюро недоступно после %d попыток, эскалируем в ручную проверку",
                state["bureau_retry_count"],
            )
            return Command(
                update={"decision": "manual_review", "audit_log": ["escalated: bureau unavailable"]},
                goto="manual_review_node",
            )
        logger.warning("Попытка %d/%d обогащения от бюро", state["bureau_retry_count"] + 1, MAX_BUREAU_RETRIES)
        return Command(update={"bureau_retry_count": 1}, goto="credit_bureau_check")

    score = state["final_risk_score"] or 0.0
    decision = classify_risk(score)
    ...
```

**Правило, которое нужно применять к каждому циклическому ребру без исключений:** прежде чем писать `goto`/`add_conditional_edges`, ведущие обратно "назад" по графу, явно ответьте на вопрос — **"какое поле состояния гарантированно меняется на каждой итерации и какое конкретное значение этого поля остановит цикл?"**. Если у вас нет чёткого ответа — у вас нет цикла, у вас есть будущий `GraphRecursionError`. Параметр `recursion_limit` в конфиге — это защита от вашей собственной невнимательности, а не архитектурное решение проблемы; полагаться на него как на "план Б" — плохая инженерная практика.

---

### 🛠️ 5. Лабораторная мини-работа

**Задание:**

Возьмите `build_graph_command()` из раздела 2 и реализуйте описанный в разделе 4 сценарий повторного обогащения от бюро **целиком, с рабочим кодом**:

1. Добавьте в `UnderwritingState` поле `bureau_retry_count: Annotated[int, operator.add]` и узел `credit_bureau_check`, который с некоторой вероятностью (смоделируйте через детерминированную функцию от `applicant_national_id`, как в Уроке 2) не возвращает `bureau_report`.
2. Модифицируйте `aggregate_risk_and_route_node`, чтобы он реализовывал логику "Good Practice" из раздела 4: до `MAX_BUREAU_RETRIES = 2` повторов, затем — принудительная эскалация в `manual_review_node` с соответствующей записью в `audit_log`.
3. Добейтесь, чтобы граф **корректно компилировался** с учётом того, что теперь у `aggregate_risk_and_route_node` **четыре** возможных направления (`credit_bureau_check`, `auto_approve_node`, `manual_review_node`, `auto_reject_node`) — не забудьте про точность типовой аннотации `Command[Literal[...]]`.

**Вектор решения (Подсказка):**

Обратите внимание на редьюсер `operator.add` для `bureau_retry_count`. Это осознанный выбор, не случайность: вместо того чтобы внутри узла делать `state["bureau_retry_count"] + 1` и возвращать абсолютное новое значение (что потребовало бы точного знания текущего значения на момент записи), вы возвращаете **инкремент** (`{"bureau_retry_count": 1}`), а редьюсер сам суммирует его с накопленным. Это точно та же идея, с которой вы столкнулись в лабораторной Урока 1 (там пришлось вручную конкатенировать список) и которую мы формализовали в Уроке 2. Здесь она возвращается в новом контексте — счётчик попыток внутри управляющего цикла — чтобы закрепить: **редьюсеры существуют не только для параллельных fan-out веток, они точно так же корректно работают и для последовательных повторных проходов через один и тот же узел**, если вы возвращаете дельту, а не абсолютное значение.

---
