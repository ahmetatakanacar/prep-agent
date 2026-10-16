"""Demo mode: a scripted planner for tests/fixtures/customers.csv, so the UI can be tried without
an API key.

The script is built on `FakeLLM` and always tells the same story: a round of questions, a first
plan the validators reject, and a corrected plan. It behaves the way the real agent does: it
asks only what the rules cannot settle (which column is the target, whether to drop a date it
cannot use), offers free text only where the answer cannot be limited to options, and writes
in the language of the interface, as the real agent writes in the language of the goal. The
mistakes in the first plan are real: the real validators find them.
"""

from collections.abc import Callable

from pydantic import BaseModel

from prep_agent.llm.base import LLMError, LLMResult
from prep_agent.llm.fake import FakeLLM
from prep_agent.profiler import DatasetProfile

TARGET = "churned"
REQUIRED_COLUMNS = frozenset(
    {"customer_id", "age", "plan", "signup_date", "monthly_spend", "churned", "country"}
)
# The story has three calls; a fourth, if the plan is still rejected, repeats the last one.
LAST_CALL = 2
TARGET_TYPES = ("binary", "categorical", "numeric")
MAX_TARGET_OPTIONS = 4

TEXTS = {
    "en": {
        "drop": "Drop it",
        "keep": "Keep it",
        "target_q": "Which column should the model predict?",
        "target_why": (
            "The goal does not name a column and several could be predicted, so I do not guess. "
            "`churned` is a yes/no column, which fits a goal about customers leaving."
        ),
        "date_q": (
            "`signup_date` is stored as text and no step can turn it into date features. Drop it?"
        ),
        "date_why": (
            "None of the available steps can extract features from a date string, so keeping it "
            "would leave a text column in the data. Whether to keep it anyway is your call."
        ),
        "drop_id": "customer_id identifies a customer and says nothing about churn",
        "drop_country": "country holds a single value",
        "drop_date": "signup_date is a raw date string no step can use, and you chose to drop it",
        "drop_ghost": "tenure_months is derived from signup_date, so it goes too.",
        "impute_early": "age has missing values; fill them with the median of the training rows.",
        "impute_late": (
            "age has missing values. The median is learned from train only and then applied to "
            "test, so no test row shapes it."
        ),
        "dedupe": "Identical rows would land on both sides of the split and inflate the scores.",
        "target_missing": "A row without a churn label cannot teach or test anything.",
        "split": "Hold out a test set; stratified because churn is the rarer outcome.",
        "encode": "plan has three levels and no natural order, so one-hot, fit on train.",
        "scale_bad": "monthly_spend has a large outlier, so use a robust scaler.",
        "scale_good": (
            "monthly_spend has a large outlier, so a robust scaler. plan is categorical and is "
            "encoded instead of scaled."
        ),
        "limit": (
            "the demo script is written for the target 'churned'; answer with it to see the "
            "whole story"
        ),
    },
    "tr": {
        "drop": "Düşür",
        "keep": "Tut",
        "target_q": "Model hangi sütunu tahmin etsin?",
        "target_why": (
            "Hedef sütun belirtilmedi ve birden fazla sütun tahmin edilebilir, bu yüzden "
            "tahmin yürütmüyorum. `churned` evet/hayır sütunu ve müşterilerin ayrılmasıyla ilgili "
            "bir amaca uyuyor."
        ),
        "date_q": (
            "`signup_date` metin olarak duruyor ve hiçbir adım onu tarih özelliklerine "
            "dönüştüremiyor. Düşürülsün mü?"
        ),
        "date_why": (
            "Mevcut adımların hiçbiri tarih metninden özellik çıkaramıyor; tutarsak veride bir "
            "metin sütunu kalır. Yine de tutup tutmamak senin kararın."
        ),
        "drop_id": "customer_id bir müşteriyi tanımlar ve kayıp hakkında bir şey söylemez",
        "drop_country": "country tek bir değer taşıyor",
        "drop_date": (
            "signup_date ham bir tarih metni, hiçbir adım onu kullanamıyor ve düşürmeyi sen seçtin"
        ),
        "drop_ghost": "tenure_months signup_date'ten türetilmiş, o da gidiyor.",
        "impute_early": "age eksik değerler içeriyor; eğitim satırlarının medyanıyla doldur.",
        "impute_late": (
            "age eksik değerler içeriyor. Medyan yalnızca train'den öğrenilip teste uygulanıyor; "
            "hiçbir test satırı onu etkilemiyor."
        ),
        "dedupe": "Aynı satırlar split'in iki yanına da düşüp skorları şişirirdi.",
        "target_missing": "Churn etiketi olmayan bir satır hiçbir şey öğretemez ya da sınayamaz.",
        "split": "Test kümesini ayır; churn seyrek sonuç olduğu için tabakalı.",
        "encode": "plan üç düzeyli ve doğal bir sırası yok, bu yüzden one-hot, train'de öğrenilir.",
        "scale_bad": (
            "monthly_spend'de büyük bir aykırı değer var, bu yüzden dayanıklı ölçekleyici."
        ),
        "scale_good": (
            "monthly_spend'de büyük bir aykırı değer var, bu yüzden dayanıklı ölçekleyici. plan "
            "kategorik, ölçeklenmiyor, kodlanıyor."
        ),
        "limit": (
            "demo betiği 'churned' hedefi için yazıldı; hikâyenin tamamını görmek için onu seç"
        ),
    },
}


class DemoError(ValueError):
    """The demo script cannot serve this dataset or this form."""


class DemoScriptLimit(LLMError):
    """The user's answer leads somewhere the script has no plan for."""


class DemoLLM:
    """Answers the three planner calls of the story, one after the other."""

    def __init__(
        self,
        profile: DatasetProfile,
        target: str | None,
        model_family: str | None,
        answers: Callable[[], dict[str, str]],
        lang: str = "en",
    ) -> None:
        missing = REQUIRED_COLUMNS - {column.name for column in profile.columns}
        if missing:
            raise DemoError(
                "the demo script is written for tests/fixtures/customers.csv; this file lacks "
                f"the columns {', '.join(sorted(missing))}"
            )
        if target not in (None, TARGET):
            raise DemoError(f"the demo script is written for the target {TARGET!r}, not {target!r}")
        self._text = TEXTS.get(lang, TEXTS["en"])
        self._profile = profile
        self._target_given = target is not None
        self._family = model_family or "any"
        self._answers = answers
        self._calls = 0

    def generate[T: BaseModel](self, system: str, user: str, schema: type[T]) -> LLMResult[T]:
        index = min(self._calls, LAST_CALL)
        self._calls += 1
        if index == 0:
            response: dict = {"kind": "questions", "plan": None, "questions": self._questions()}
        else:
            response = {"kind": "plan", "plan": self._plan(rejected=index == 1), "questions": None}
        # No usage is reported: a script used no tokens, and a made-up count would pass for a
        # real one.
        return FakeLLM([response]).generate(system, user, schema)

    def _questions(self) -> list[dict]:
        text = self._text
        questions = []
        if not self._target_given:
            questions.append(
                {
                    "id": "target",
                    "question": text["target_q"],
                    "options": self._target_options(),
                    "recommended": TARGET,
                    "reason": text["target_why"],
                    # Any column of the file may be the target, and the options are only the
                    # likely ones.
                    "allow_custom": True,
                }
            )
        questions.append(
            {
                "id": "signup_date",
                "question": text["date_q"],
                "options": [text["drop"], text["keep"]],
                "recommended": text["drop"],
                "reason": text["date_why"],
                "allow_custom": False,
            }
        )
        return questions

    def _target_options(self) -> list[str]:
        others = [
            column.name
            for column in self._profile.columns
            if column.name != TARGET and column.inferred_type in TARGET_TYPES
        ]
        return [TARGET, *others][:MAX_TARGET_OPTIONS]

    def _plan(self, *, rejected: bool) -> dict:
        text = self._text
        answers = self._answers()
        if answers.get("target", TARGET).strip() != TARGET:
            raise DemoScriptLimit(text["limit"])
        keep_date = answers.get("signup_date", text["drop"]) == text["keep"]
        build = _rejected_steps if rejected else _corrected_steps
        return {
            "target": TARGET,
            "task_type": "classification",
            "model_family": self._family,
            "steps": build(text, drop_date=not keep_date, scale=self._family != "tree"),
        }


def _drop_step(text: dict, *, drop_date: bool, ghost: bool) -> dict:
    columns = ["customer_id", "country"]
    reasons = [text["drop_id"], text["drop_country"]]
    if drop_date:
        columns.insert(1, "signup_date")
        reasons.append(text["drop_date"])
    reason = "; ".join(reasons) + "."
    if ghost:
        columns.append("tenure_months")
        reason += " " + text["drop_ghost"]
    return {"op": "drop_columns", "columns": columns, "reason": reason}


def _impute_step(text: dict, *, early: bool) -> dict:
    return {
        "op": "impute",
        "columns": ["age"],
        "strategy": "median",
        "fill_value": None,
        "add_indicator": False,
        "reason": text["impute_early"] if early else text["impute_late"],
    }


def _fixed_steps(text: dict) -> dict[str, dict]:
    return {
        "dedupe": {"op": "dedupe", "subset": None, "keep": "first", "reason": text["dedupe"]},
        "missing": {"op": "drop_missing_target", "reason": text["target_missing"]},
        "split": {
            "op": "split",
            "test_size": 0.2,
            "stratify": True,
            "random_state": 42,
            "reason": text["split"],
        },
        "encode": {
            "op": "encode",
            "columns": ["plan"],
            "method": "onehot",
            "max_categories": None,
            "reason": text["encode"],
        },
    }


def _scale_step(columns: list[str], reason: str) -> dict:
    return {"op": "scale", "columns": columns, "method": "robust", "reason": reason}


def _rejected_steps(text: dict, *, drop_date: bool, scale: bool) -> list[dict]:
    fixed = _fixed_steps(text)
    steps = [
        _drop_step(text, drop_date=drop_date, ghost=True),
        _impute_step(text, early=True),
        fixed["dedupe"],
        fixed["missing"],
        fixed["split"],
    ]
    if scale:
        steps.append(_scale_step(["age", "monthly_spend", "plan"], text["scale_bad"]))
    steps.append(fixed["encode"])
    return steps


def _corrected_steps(text: dict, *, drop_date: bool, scale: bool) -> list[dict]:
    fixed = _fixed_steps(text)
    steps = [
        _drop_step(text, drop_date=drop_date, ghost=False),
        fixed["dedupe"],
        fixed["split"],
        _impute_step(text, early=False),
        fixed["encode"],
    ]
    if scale:
        steps.append(_scale_step(["age", "monthly_spend"], text["scale_good"]))
    return steps
