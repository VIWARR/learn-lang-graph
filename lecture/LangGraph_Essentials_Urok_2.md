# LangGraph Essentials

## Урок 2: Ребра и Параллельное выполнение (Edges & Reducers)

---

### 🧠 1. Концептуальная база & Архитектура

В лабораторной работе Урока 1 я намеренно подвёл вас к боли: чтобы список `rejection_history` аккумулировался, а не затирался, вам пришлось вручную читать старое значение из `state`, конкатенировать его с новым и возвращать целиком. Это работало — но не масштабируется. А главное, это **разваливается в тот момент, когда у вас появляется параллельное выполнение**. Разберёмся, почему.

#### Модель выполнения LangGraph: супершаги (Pregel-модель)

LangGraph построен на модели вычислений **Pregel** (та же идея, что лежит в основе распределённых графовых систем Google). Ключевая абстракция — **супершаг (superstep)**:

- На каждом супершаге LangGraph определяет, какие узлы **активны** (получили новое сообщение/состояние на входящем канале).
- Все активные узлы одного супершага запускаются **параллельно**.
- Узлы **одного и того же супершага не видят обновлений друг друга** — они все стартуют от одного и того же согласованного снимка состояния.
- Только когда все узлы супершага завершились, их обновления **сливаются одновременно**, и граф переходит к следующему супершагу.

**Практическое следствие:** если вы делаете fan-out — `add_edge(START, "node_a")` и `add_edge(START, "node_b")` — оба узла окажутся в одном супершаге. Это не "сначала A, потом B", это буквально конкурентное выполнение (в асинхронном рантайме — настоящие параллельные корутины, ожидающие I/O одновременно). И вот тут возникает вопрос, ради которого существует Урок 2: **если `node_a` и `node_b` оба хотят дописать что-то в один и тот же ключ состояния — что происходит?**

#### Что происходит без редьюсера

По умолчанию каждое поле `State` — это канал типа `LastValue`: "получил обновление — заменил старое значение новым". Но у `LastValue`-канала есть встроенная защита: **он физически не умеет принять два разных значения в рамках одного супершага**. Если `node_a` и `node_b`, работающие параллельно, оба вернут разные значения для ключа `foo`, LangGraph не будет молча выбирать одно из них (это было бы недетерминированное поведение, а недетерминизм в проде — это будущий инцидент). Вместо этого он **упадёт с ошибкой**:

```
langgraph.errors.InvalidUpdateError: At key 'foo': Can receive only one value per step.
Use an Annotated key to handle multiple values.
```

Это не баг и не сообщение "на всякий случай" — это **осознанный архитектурный контракт**: параллельная запись в один канал без явно указанной стратегии слияния запрещена на уровне рантайма. Именно это правило заставляет вас думать о конфликтах данных на этапе проектирования схемы, а не отлаживать race condition в проде в 3 часа ночи.

#### Редьюсеры: явный контракт слияния

**Редьюсер** — это чистая функция с сигнатурой `(left: T, right: T) -> T`, где `left` — это значение, уже накопленное в состоянии, а `right` — новое обновление от узла. Вы прикрепляете редьюсер к полю через `Annotated`:

```python
from typing import Annotated
import operator

class State(TypedDict):
    tags: Annotated[list[str], operator.add]
```

Когда поле аннотировано редьюсером, LangGraph строит для него не `LastValue`-канал, а **`BinaryOperatorAggregate`**-канал: вместо замены он вызывает вашу функцию `reducer(накопленное_значение, новое_обновление)` и сохраняет результат. Именно это снимает ограничение "одно значение за супершаг" — теперь несколько параллельных узлов **могут** писать в один ключ, потому что у канала есть чёткая инструкция, как эти записи скомбинировать.

**Критически важное инженерное требование к редьюсеру, которое почти никто не формулирует явно: он должен быть чистым и, в идеале, коммутативным.** Почему коммутативным? Потому что при настоящем параллельном (асинхронном, I/O-bound) выполнении вы **не контролируете**, какая из двух корутин физически завершится первой — это зависит от таймингов внешних систем (задержка сети до кредитного бюро против задержки сервиса антифрода). Если ваш редьюсер даёт разный результат в зависимости от порядка применения (`merge(A, B) != merge(B, A)`), то один и тот же граф с одними и теми же входными данными будет выдавать **разные** результаты от запуска к запуску. Это самый коварный класс багов — он не воспроизводится стабильно.

---

### 💻 2. Разбор кода (Production-ready пример)

Продолжаем граф кредитного скоринга из Урока 1. После `intake_validation` заявка параллельно уходит в **два независимых внешних сервиса**: проверку кредитной истории (бюро) и антифрод-скрининг. Оба узла пишут в общее состояние, а третий узел агрегирует результат.

> Ради концентрации на теме урока намеренно не используется условная маршрутизация (например, "не гонять параллельные проверки, если заявка уже отклонена на этапе intake") — этим мы вплотную займёмся в Уроке 3.

```python
"""
lending_parallel_screening.py

Параллельное обогащение заявки: кредитное бюро + антифрод-скрининг.
Демонстрирует fan-out/fan-in, редьюсеры (operator.add и кастомный merge)
и защиту от race condition при слиянии состояния.
"""

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
# 2. Кастомный редьюсер.
#    ВАЖНО: он коммутативен (max не зависит от порядка аргументов) —
#    это гарантия корректности при недетерминированном порядке завершения
#    параллельных async-узлов.
# --------------------------------------------------------------------------

def merge_risk_signals(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    """Сливает сигналы риска от разных источников, беря наихудший (максимум) по каждому ключу."""
    merged = dict(left)
    for key, value in right.items():
        merged[key] = max(merged.get(key, value), value)
    return merged


# --------------------------------------------------------------------------
# 3. Схема состояния (расширяем UnderwritingState из Урока 1)
# --------------------------------------------------------------------------

class UnderwritingState(TypedDict):
    raw_payload: dict
    applicant_national_id: str
    status: Literal["pending", "screened"]

    bureau_report: Optional[BureauReport]

    # operator.add: список конкатенируется при параллельной записи.
    # ассоциативен, но НЕ коммутативен — порядок элементов может плавать
    # между запусками. Для флагов это некритично (порядок не несёт смысла).
    fraud_flags: Annotated[list[str], operator.add]

    # кастомный коммутативный редьюсер: порядок применения не влияет на результат.
    risk_signals: Annotated[dict[str, float], merge_risk_signals]

    final_risk_score: Optional[float]


# --------------------------------------------------------------------------
# 4. "Внешние" сервисы (в реальной системе — httpx.AsyncClient к бюро/антифроду)
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


# --------------------------------------------------------------------------
# 5. Узлы графа — оба стартуют из одного и того же супершага
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


async def aggregate_risk_node(state: UnderwritingState) -> dict:
    """Fan-in узел: срабатывает только после завершения ОБОИХ параллельных узлов."""
    signals = state["risk_signals"]
    # Простая взвешенная сумма всех накопленных сигналов риска
    final_score = round(sum(signals.values()) / max(len(signals), 1), 3)
    logger.info("aggregate_risk_node: итоговый риск-скор=%.3f (из сигналов: %s)", final_score, signals)

    return {"final_risk_score": final_score, "status": "screened"}


# --------------------------------------------------------------------------
# 6. Сборка графа: fan-out из intake, fan-in в aggregate_risk
# --------------------------------------------------------------------------

def build_graph():
    builder = StateGraph(UnderwritingState)

    builder.add_node("credit_bureau_check", credit_bureau_check_node)
    builder.add_node("fraud_screening", fraud_screening_node)
    builder.add_node("aggregate_risk", aggregate_risk_node)

    # Fan-out: оба узла запускаются в одном супершаге сразу после START
    builder.add_edge(START, "credit_bureau_check")
    builder.add_edge(START, "fraud_screening")

    # Fan-in: aggregate_risk ждёт завершения ОБОИХ входящих рёбер
    builder.add_edge("credit_bureau_check", "aggregate_risk")
    builder.add_edge("fraud_screening", "aggregate_risk")

    builder.add_edge("aggregate_risk", END)
    return builder.compile()


async def main() -> None:
    graph = build_graph()

    initial_state: UnderwritingState = {
        "raw_payload": {},
        "applicant_national_id": "AB1234567",
        "status": "pending",
        "bureau_report": None,
        "fraud_flags": [],
        "risk_signals": {},
        "final_risk_score": None,
    }

    try:
        result = await graph.ainvoke(initial_state)
    except Exception:
        logger.exception("Непредвиденный сбой при выполнении графа")
        raise

    logger.info("Итоговый риск-скор: %s | флаги: %s", result["final_risk_score"], result["fraud_flags"])


if __name__ == "__main__":
    asyncio.run(main())
```

**На что обратить внимание:**

- `aggregate_risk_node` не нужно вручную "ждать" оба узла — LangGraph сам синхронизирует супершаг: узел с двумя входящими рёбрами (`credit_bureau_check → aggregate_risk` и `fraud_screening → aggregate_risk`) активируется только тогда, когда **оба** источника прислали сообщение.
- Оба внешних вызова обёрнуты в `try/except` с **разными стратегиями деградации**: сбой бюро не должен ронять весь граф, а должен консервативно повышать риск-скор — это осознанное бизнес-решение, а не техническая заглушка.

---

### 📊 3. Трассировка Состояния (State Lifecycle)

**Супершаг 0 (после `START`):** `credit_bureau_check` и `fraud_screening` активируются одновременно, оба получают идентичный снимок состояния:

| Ключ | Значение на входе в оба узла |
|---|---|
| `bureau_report` | `None` |
| `fraud_flags` | `[]` |
| `risk_signals` | `{}` |
| `final_risk_score` | `None` |

**Что возвращает каждый узел (частичные, независимые обновления):**

```python
# credit_bureau_check_node вернул:
{"bureau_report": BureauReport(...), "risk_signals": {"credit_history_risk": 0.31}}

# fraud_screening_node вернул (выполняется конкурентно, независимо):
{"fraud_flags": ["device_fingerprint_mismatch"], "risk_signals": {"fraud_risk": 0.8}}
```

**Слияние в конце супершага 0.** Так как `risk_signals` аннотирован кастомным редьюсером, LangGraph вызывает его для объединения обоих обновлений:

```python
merge_risk_signals(
    merge_risk_signals({}, {"credit_history_risk": 0.31}),
    {"fraud_risk": 0.8},
)
# => {"credit_history_risk": 0.31, "fraud_risk": 0.8}
```

Обратите внимание: поскольку ключи `credit_history_risk` и `fraud_risk` **не пересекаются**, порядок применения (сначала бюро, потом антифрод, или наоборот) **не влияет на итог** — именно этого мы и добивались коммутативностью редьюсера.

Для `fraud_flags` сработал `operator.add`: `[] + ["device_fingerprint_mismatch"] = ["device_fingerprint_mismatch"]`.

**Состояние после супершага 0 (вход в супершаг 1 — `aggregate_risk`):**

| Ключ | Значение | Механизм |
|---|---|---|
| `bureau_report` | `BureauReport(raw_score=...)` | LastValue (пишет только один узел) |
| `fraud_flags` | `["device_fingerprint_mismatch"]` | `operator.add` |
| `risk_signals` | `{"credit_history_risk": 0.31, "fraud_risk": 0.8}` | `merge_risk_signals` |
| `final_risk_score` | `None` | не тронут |

**Супершаг 1 (`aggregate_risk`):** узел читает уже полностью слитый `risk_signals`, считает среднее, возвращает `{"final_risk_score": 0.555, "status": "screened"}`. Поле `final_risk_score` не имеет редьюсера — здесь пишет только один узел, конфликтов нет, применяется обычный `LastValue`.

---

### ⚠️ 4. Ошибка новичка (Bad Practice vs Good Practice)

Самая частая ошибка на этом уроке — **забыть аннотировать редьюсером поле, в которое потенциально пишут несколько параллельных узлов**, понадеявшись, что "ну там же просто dict, как-нибудь сложится".

**❌ Bad Practice:**

```python
class UnderwritingState(TypedDict):
    bureau_report: Optional[BureauReport]
    fraud_flags: list[str]          # НЕТ РЕДЬЮСЕРА
    risk_signals: dict[str, float]  # НЕТ РЕДЬЮСЕРА
    final_risk_score: Optional[float]
```

При таком объявлении граф из раздела 2 **скомпилируется без единой ошибки** — Python не увидит здесь ничего подозрительного, это валидный `TypedDict`. Проблема проявится **только в рантайме**, причём не всегда: если один из узлов упадёт по исключению и вернёт `risk_signals` только от одного источника — граф отработает. Но в "счастливом" сценарии, когда оба узла успешно завершились в одном супершаге и оба попытались записать в `risk_signals`, вы получите:

```
langgraph.errors.InvalidUpdateError: At key 'risk_signals': Can receive only one value per step.
Use an Annotated key to handle multiple values.
```

**Почему это особенно коварно:** такой баг легко проходит юнит-тесты, если вы тестируете узлы **по отдельности** (стандартная практика), и даже интеграционный тест может "случайно" не поймать его, если в тестовом окружении сервисы отвечают настолько быстро, что вы не воспроизводите нужный тайминг параллельности в каждом прогоне CI. Это ровно тот класс дефектов, который добирается до продакшена и падает непредсказуемо под нагрузкой, когда оба внешних сервиса действительно отвечают "одновременно".

**✅ Good Practice:**

```python
class UnderwritingState(TypedDict):
    bureau_report: Optional[BureauReport]
    fraud_flags: Annotated[list[str], operator.add]
    risk_signals: Annotated[dict[str, float], merge_risk_signals]
    final_risk_score: Optional[float]
```

**Правило для чек-листа код-ревью:** прежде чем добавить узел, который пишет в поле состояния, спросите — "а может ли в это же поле в этом же супершаге писать ещё какой-то узел?". Если ответ "да" или "не уверен" — полю **обязателен** редьюсер, даже если сегодня в графе только один писатель. Графы имеют свойство обрастать параллельными ветками со временем, и поле, безопасное сегодня, завтра может стать источником `InvalidUpdateError` после, казалось бы, невинного добавления нового узла.

---

### 🛠️ 5. Лабораторная мини-работа

**Задание:**

Расширьте граф из раздела 2 третьим параллельным узлом — `sanctions_screening_node` (проверка по санкционным спискам, например аналог OFAC). Требования:

1. Узел должен запускаться **в том же супершаге**, что `credit_bureau_check` и `fraud_screening` (то есть тоже fan-out из `START`), и `aggregate_risk` должен ждать все три источника, а не два.
2. Добавьте в состояние новое поле `checks_completed: Annotated[list[str], operator.add]`, в которое каждый из трёх параллельных узлов дописывает своё имя (`"credit_bureau"`, `"fraud"`, `"sanctions"`) после завершения. В `aggregate_risk_node` добавьте `assert len(state["checks_completed"]) == 3` как защитную проверку целостности перед расчётом финального скора.
3. Санкционная проверка при срабатывании должна писать в `risk_signals` ключ `"sanctions_hit"` со значением `1.0` (максимальный риск), а `aggregate_risk_node` должен трактовать `sanctions_hit >= 1.0` как обязательный **автоматический** максимум итогового скора (`1.0`), независимо от значений остальных сигналов — попадание в санкционный список не может быть "усреднено" другими более мягкими метриками.

**Вектор решения (Подсказка):**

Пункт 3 — это ловушка для тех, кто скопирует текущую формулу `aggregate_risk_node` (среднее по `risk_signals.values()`) не задумываясь. Среднее арифметическое **размывает** критичный сигнал: если у заявителя `sanctions_hit = 1.0`, но остальные метрики низкие, среднее может оказаться комфортным "средним" риском — а по-хорошему это должно быть безусловным отказом. Вам нужно **явно выделить** `sanctions_hit` из общей агрегации как отдельное бизнес-правило ("hard rule"), а не как один из равноправных сигналов в общей формуле. Это прямая иллюстрация того, что не всякая бизнес-логика укладывается в "просто добавить ещё один ключ в редьюсер" — иногда агрегирующий узел должен явно различать *типы* сигналов, а не сваливать их в одну недифференцированную кучу.

---
