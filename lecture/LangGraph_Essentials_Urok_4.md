# LangGraph Essentials

## Урок 4: Память (Memory & Persistence)

---

### 🧠 1. Концептуальная база & Архитектура

Все графы, которые мы строили в Уроках 1–3, живут и умирают в рамках одного вызова `ainvoke()`. Закрылся процесс — состояние исчезло. Для демо-скрипта это нормально. Для системы кредитного андеррайтинга, которая может быть приостановлена на ручную проверку на несколько часов, упасть из-за сетевого сбоя посреди обращения к бюро, или просто обслуживать тысячи независимых заявителей одновременно — это неприемлемо. Нужен слой персистентности. В LangGraph за это отвечает **Checkpointer**.

#### Что такое чекпоинт на самом деле

Чекпоинт — это не "лог событий" и не "снимок для дебага". Это **полный, самодостаточный снимок всех каналов состояния** на границе супершага, достаточный, чтобы полностью восстановить и продолжить выполнение графа без какой-либо дополнительной информации. Важно усвоить архитектурный факт: **чекпоинт сохраняется на границах супершагов, а не в середине выполнения узла**. Это значит:

- Если узел успешно завершился и вернул `dict`/`Command` — LangGraph фиксирует новое состояние в чекпоинте.
- Если узел упал с исключением **посреди** выполнения (после того как он, например, уже успел вызвать внешний платёжный API, но до того как вернул `dict`) — на диске останется чекпоинт **до** этого узла, а не "наполовину выполненный" узел.
- При возобновлении граф не пытается угадать, "докуда" добрался упавший узел — он просто **запускает этот узел заново, с самого начала функции**, от последнего сохранённого состояния.

**Прямое следствие, которое обязан помнить каждый инженер, работающий с LangGraph:** любой узел, вызывающий системы с побочными эффектами (списание денег, отправка письма, запись строки в БД), должен быть написан **идемпотентно** — повторный запуск с теми же входными данными не должен приводить к дублирующему эффекту. Это не "хорошая практика вообще", это прямое архитектурное требование модели checkpoint-at-superstep-boundary.

#### Адресация: `thread_id`, `checkpoint_ns`, `checkpoint_id`

Каждый вызов графа привязывается к конфигу вида:

```python
config = {"configurable": {"thread_id": "applicant-AB1234567"}}
```

- **`thread_id`** — идентификатор независимого "потока выполнения" (в чат-системах это ID диалога, у нас — естественно, ID заявителя). Все чекпоинты одного `thread_id` образуют изолированную историю; чекпоинтер физически не смешивает данные разных `thread_id` — это и есть встроенная **многопользовательская изоляция**, без which вам пришлось бы вручную партиционировать БД по клиентам.
- **`checkpoint_ns`** — пространство имён, используется для вложенных подграфов (чтобы чекпоинты дочернего графа не путались с родительским).
- **`checkpoint_id`** — адрес конкретного исторического снимка внутри треда; если не указан, берётся последний. Указание конкретного `checkpoint_id` — это то, что открывает "путешествие во времени" (time travel), к которому мы подробнее вернёмся при разборе Human-in-the-loop.

#### Зоопарк реализаций `BaseCheckpointSaver`

| Реализация | Хранилище | Годится для | Ключевая оговорка |
|---|---|---|---|
| `InMemorySaver` | Оперативная память процесса | Разработка, юнит-тесты | Всё исчезает при перезапуске процесса |
| `SqliteSaver` / `AsyncSqliteSaver` | Файл SQLite | Прототипы, одиночный сервер, демо | Не рассчитан на высокую конкурентную запись — не для прод-нагрузки |
| `PostgresSaver` / `AsyncPostgresSaver` | PostgreSQL | Продакшен, множественные инстансы приложения | Требует явного вызова `.setup()` перед первым использованием (создание таблиц) |

**Синхронные и асинхронные версии — это не взаимозаменяемые альтернативы "на вкус".** Если ваш граф целиком построен на `async def`-узлах и вызывается через `ainvoke()`, вам категорически необходим **асинхронный** чекпоинтер (`AsyncSqliteSaver`, `AsyncPostgresSaver`) и асинхронные методы инспекции (`await graph.aget_state(...)`, а не `graph.get_state(...)`). Смешение синхронного и асинхронного стилей на уровне персистентности — тема раздела 4 этого урока, и это не теоретическая придирка: это реальный, задокументированный класс продакшен-инцидентов.

---

### 💻 2. Разбор кода (Production-ready пример)

Соберём отказоустойчивую версию мини-конвейера андеррайтинга: `intake_node → credit_bureau_check_node → finalize_node`, с персистентностью на файловом SQLite (`AsyncSqliteSaver`). Продемонстрируем: (1) восстановление после сбоя без повторного выполнения уже завершённых узлов, (2) изоляцию двух независимых заявителей через `thread_id`.

```python
"""
lending_persistence.py

Отказоустойчивый конвейер андеррайтинга с чекпоинтингом на SQLite.
Демонстрирует: восстановление после сбоя (crash recovery) и
изоляцию параллельных заявителей через thread_id.

Зависимость: pip install aiosqlite
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, START, END

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("lending_graph")


# --------------------------------------------------------------------------
# 1. Состояние
# --------------------------------------------------------------------------

class UnderwritingState(TypedDict):
    applicant_national_id: str
    bureau_score: Optional[int]
    status: str


# --------------------------------------------------------------------------
# 2. "Внешний" сервис
# --------------------------------------------------------------------------

async def call_credit_bureau_api(national_id: str) -> int:
    await asyncio.sleep(0.2)
    return 300 + (sum(map(ord, national_id)) % 551)


# --------------------------------------------------------------------------
# 3. Узлы
# --------------------------------------------------------------------------

async def intake_node(state: UnderwritingState) -> dict:
    logger.info("intake_node: приём заявки %s", state["applicant_national_id"])
    return {"status": "intake_complete"}


async def credit_bureau_check_node(state: UnderwritingState, config: RunnableConfig) -> dict:
    """
    Параметр simulate_outage приходит через runtime-config (configurable),
    а НЕ через State. Это осознанный выбор: флаги окружения, креды,
    feature-флаги не должны попадать в персистентное состояние графа —
    они относятся к конкретному вызову, а не к данным заявки.
    """
    national_id = state["applicant_national_id"]
    simulate_outage = config["configurable"].get("simulate_outage", False)

    if simulate_outage:
        logger.error("credit_bureau_check_node: имитация сбоя инфраструктуры бюро")
        raise RuntimeError("Сбой инфраструктуры: сервис кредитного бюро недоступен")

    try:
        score = await call_credit_bureau_api(national_id)
    except Exception:
        logger.exception("Непредвиденный сбой при обращении к бюро")
        raise

    logger.info("credit_bureau_check_node: получен score=%d для %s", score, national_id)
    return {"bureau_score": score, "status": "enriched"}


async def finalize_node(state: UnderwritingState) -> dict:
    logger.info("finalize_node: заявка %s завершена, score=%s", state["applicant_national_id"], state["bureau_score"])
    return {"status": "completed"}


def build_graph(checkpointer) -> "CompiledStateGraph":
    builder = StateGraph(UnderwritingState)
    builder.add_node("intake", intake_node)
    builder.add_node("credit_bureau_check", credit_bureau_check_node)
    builder.add_node("finalize", finalize_node)
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "credit_bureau_check")
    builder.add_edge("credit_bureau_check", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)


async def main() -> None:
    db_path = "lending_checkpoints.db"
    applicant_a = "AB1234567"
    applicant_b = "CD7654321"

    # --------------------------------------------------------------------
    # "День 1": инфраструктура бюро временно недоступна, граф падает
    # после успешного intake, но чекпоинт intake уже сохранён.
    # --------------------------------------------------------------------
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = build_graph(checkpointer)
        config_a = {"configurable": {"thread_id": applicant_a, "simulate_outage": True}}

        initial_state: UnderwritingState = {
            "applicant_national_id": applicant_a,
            "bureau_score": None,
            "status": "pending",
        }

        try:
            await graph.ainvoke(initial_state, config_a)
        except RuntimeError:
            logger.warning("День 1: граф упал на credit_bureau_check, но intake уже в чекпоинте")

        snapshot = await graph.aget_state(config_a)
        logger.info("День 1: следующий узел к выполнению -> %s", snapshot.next)

    # --------------------------------------------------------------------
    # "День 2": новый процесс (здесь — новый checkpointer/graph, как если
    # бы приложение перезапустили). Инфраструктура бюро восстановлена.
    # --------------------------------------------------------------------
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = build_graph(checkpointer)

        config_a_resume = {"configurable": {"thread_id": applicant_a, "simulate_outage": False}}
        # ВАЖНО: передаём None вместо initial_state — это официальный способ
        # сказать "продолжи с последнего чекпоинта для этого thread_id",
        # а не "начни выполнение заново".
        result_a = await graph.ainvoke(None, config_a_resume)
        logger.info("День 2: заявка %s завершена со статусом %s", applicant_a, result_a["status"])

        # ------------------------------------------------------------
        # Проверка изоляции: параллельно обслуживаем второго заявителя
        # с полностью независимым thread_id в том же чекпоинтере.
        # ------------------------------------------------------------
        config_b = {"configurable": {"thread_id": applicant_b, "simulate_outage": False}}
        initial_state_b: UnderwritingState = {
            "applicant_national_id": applicant_b,
            "bureau_score": None,
            "status": "pending",
        }
        result_b = await graph.ainvoke(initial_state_b, config_b)

        snapshot_a = await graph.aget_state(config_a_resume)
        snapshot_b = await graph.aget_state(config_b)
        logger.info("Изоляция: заявитель A -> %s | заявитель B -> %s", snapshot_a.values, snapshot_b.values)


if __name__ == "__main__":
    asyncio.run(main())
```

**На что обратить внимание:**

- `simulate_outage` живёт в `config["configurable"]`, а не в `UnderwritingState` — это runtime-параметр вызова, а не данные заявки. Смешивать эти два понятия — частая путаница у новичков: **State персистентен и версионирован чекпоинтером, `config` — нет** (за исключением `thread_id`/`checkpoint_id`, которые сами являются частью адресации, а не данными).
- Вызов `graph.ainvoke(None, config_a_resume)` — задокументированная идиома возобновления: передавая `None` вместо стартового состояния при уже существующем `thread_id`, вы говорите LangGraph "загрузи последний чекпоинт и продолжи", а не "начни новый прогон".

---

### 📊 3. Трассировка Состояния (State Lifecycle)

Проследим полный жизненный цикл для заявителя A через два "дня".

**День 1, супершаг 0 (`intake`):**

| Событие | Детали |
|---|---|
| Вход | `{applicant_national_id: "AB...", bureau_score: None, status: "pending"}` |
| Узел вернул | `{"status": "intake_complete"}` |
| **Чекпоинт сохранён** | ✅ `checkpoint_id = C1`, `next = ("credit_bureau_check",)` |

**День 1, супершаг 1 (`credit_bureau_check`) — АВАРИЙНОЕ ЗАВЕРШЕНИЕ:**

| Событие | Детали |
|---|---|
| Вход | Состояние из `C1` |
| Узел | Бросает `RuntimeError` до какого-либо `return` |
| **Чекпоинт** | ❌ Новый чекпоинт НЕ создаётся — состояние графа остаётся на `C1` |
| Что видно снаружи | `graph.aget_state(config_a).next == ("credit_bureau_check",)` — граф "знает", что следующий шаг не выполнен |

**День 2, супершаг 1 (повторный вход, `credit_bureau_check`):**

| Событие | Детали |
|---|---|
| Вход | Тот же снимок `C1` — `intake` **не выполняется повторно** |
| Узел | На этот раз `simulate_outage=False` → успешно вызывает бюро, возвращает `{"bureau_score": 517, "status": "enriched"}` |
| **Чекпоинт сохранён** | ✅ `checkpoint_id = C2` |

**День 2, супершаг 2 (`finalize`):** обычное выполнение, `checkpoint_id = C3`, `status: "completed"`.

**Ключевой вывод трассировки:** между "Днём 1" и "Днём 2" узел `intake` выполнился **ровно один раз за всю историю треда**, несмотря на то что весь процесс приложения был условно "перезапущен". Это прямое следствие того, что персистентность работает на уровне супершагов, а не на уровне "перезапустить всё с начала при любой ошибке".

---

### ⚠️ 4. Ошибка новичка (Bad Practice vs Good Practice)

Самая специфичная для этого урока ошибка — **смешивание синхронного и асинхронного API поверх асинхронного чекпоинтера**. Это не выбрасывает вменяемое исключение — это **тихо подвешивает процесс навсегда**, и отладка такого зависания без понимания первопричины может съесть у команды целый день.

**❌ Bad Practice:**

```python
async def main() -> None:
    async with AsyncSqliteSaver.from_conn_string("lending_checkpoints.db") as checkpointer:
        graph = build_graph(checkpointer)
        config = {"configurable": {"thread_id": "AB1234567"}}

        # АНТИПАТТЕРН: синхронный invoke() на графе,
        # скомпилированном с АСИНХРОННЫМ чекпоинтером
        result = graph.invoke(initial_state, config)  # <-- зависает здесь. Навсегда.
        print(result)
```

**Почему это происходит и почему это так коварно:** `AsyncSqliteSaver` реализует интерфейс `BaseCheckpointSaver` через асинхронные методы (`aget_tuple`, `aput`, ...), рассчитанные на вызов из уже работающего event loop. Синхронный метод `graph.invoke()` внутри себя пытается получить состояние через синхронный путь чекпоинтера. В части версий это приводило к явному `NotImplementedError`, но в текущих версиях библиотеки поведение хуже: вызов **зависает без единой строчки в логе об ошибке** — процесс просто не двигается дальше, и снаружи это неотличимо от "зависшего" внешнего API. Разработчик, не знающий про эту особенность, будет часами добавлять таймауты в HTTP-клиенты и проверять сеть, вместо того чтобы посмотреть на несовпадение sync/async на уровне персистентности.

**✅ Good Practice:**

```python
async def main() -> None:
    async with AsyncSqliteSaver.from_conn_string("lending_checkpoints.db") as checkpointer:
        graph = build_graph(checkpointer)
        config = {"configurable": {"thread_id": "AB1234567"}}

        # Асинхронный чекпоинтер -> асинхронный вызов графа, без исключений
        result = await graph.ainvoke(initial_state, config)
        snapshot = await graph.aget_state(config)  # тоже асинхронный метод
        print(result, snapshot.next)
```

**Правило для код-ревью:** как только в проекте появляется `Async*Saver` (SQLite или Postgres), **весь** код, взаимодействующий с этим графом — `ainvoke`, `astream`, `aget_state`, `aget_state_history`, `aupdate_state` — обязан быть асинхронным, без единого исключения "просто для этого одного вызова". Если по какой-то причине вам нужен синхронный интерфейс — используйте синхронный чекпоинтер (`SqliteSaver`, `PostgresSaver`), а не async-вариант с точечными обходами.

---

### 🛠️ 5. Лабораторная мини-работа

**Задание:**

Наш демонстрационный сбой в разделе 2 намеренно "удобный": исключение бросается **до** обращения к внешнему бюро, поэтому при повторном выполнении узла на "Дне 2" нет риска задвоить вызов. В реальности сбой может произойти **после** успешного (и, возможно, платного) обращения к бюро, но до того как узел успеет вернуть `dict` — например, из-за обрыва сети на обратном пути ответа. При возобновлении LangGraph всё равно выполнит `credit_bureau_check_node` **с самого начала**, что означает повторный вызов уже оплаченного API.

Модифицируйте граф так, чтобы это было безопасно:

1. Добавьте в `UnderwritingState` поле `bureau_request_id: Optional[str]`.
2. Перед вызовом `call_credit_bureau_api` сгенерируйте детерминированный идемпотентный ключ (например, на основе `thread_id` и `applicant_national_id` — но НЕ на основе текущего времени или случайных чисел, иначе ключ будет разным при каждой попытке) и сохраните его в `bureau_request_id` **до** совершения вызова.
3. Измените сигнатуру `call_credit_bureau_api`, чтобы она принимала этот идемпотентный ключ как параметр (в реальном бюро это соответствовало бы заголовку `Idempotency-Key`), и добавьте в узел логику: если `state["bureau_request_id"]` уже установлен при повторном входе в узел — не генерировать новый ключ, а переиспользовать существующий.

**Вектор решения (Подсказка):**

Ключевая ловушка здесь — соблазн сгенерировать идемпотентный ключ **внутри** той же функции, что делает сам вызов, каждый раз заново (например, через `uuid.uuid4()` в начале узла). Так вы получите новый, уникальный ключ при **каждом** повторном выполнении узла — что полностью убивает идею идемпотентности: внешний сервис бюро увидит два разных ключа для двух попыток одного и того же логического запроса и не сможет их дедуплицировать. Правильная последовательность: **проверить, есть ли уже сохранённый в state ключ → если нет, сгенерировать и немедленно вернуть его как частичное обновление ДО вызова API → если да, использовать существующий**. Это, по сути, распадается на два прохода через узел (первый — только генерация и сохранение ключа, второй — собственно вызов с этим ключом), что естественным образом подводит к архитектурному вопросу: не стоит ли разбить один узел на два отдельных шага графа, чтобы у каждого шага была ровно одна ответственность и одна точка отказа? Подумайте об этом при проектировании — это ровно тот тип решений, которые отличают "код, который работает на демо" от "код, который не создаст дублирующий платёж в проде при третьем перезапуске подряд".

---
