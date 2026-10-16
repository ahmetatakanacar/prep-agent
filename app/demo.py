"""Demo mode: a scripted planner for tests/fixtures/customers.csv, so the UI can be tried without
an API key.

The script is built on `FakeLLM` and always tells the same story: a round of questions, a first
plan the validators reject, and a corrected plan. It behaves the way the real agent does: it
asks only what the brief leaves open (which column is the target, which model family) and skips
the round when the brief answers both, drops a date without asking because no step can use it,
offers free text only where the answer cannot be limited to options, and writes in the language
of the interface, as the real agent writes in the language of the goal. The mistakes in the
first plan are real: the real validators find them.
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
TARGET_TYPES = ("binary", "categorical", "numeric")
MAX_TARGET_OPTIONS = 4

TEXTS = {
    "en": {
        "target_q": "Which column should the model predict?",
        "target_why": (
            "The goal does not name a column and several could be predicted, so I do not guess. "
            "`churned` is a yes/no column, which fits a goal about customers leaving."
        ),
        "family_q": "Which kind of model will you train on the prepared data?",
        # The options carry the labels of the family buttons of the form, and the answer is
        # mapped back to the family.
        "families": {"tree": "tree", "linear": "linear", "any": "any"},
        "family_why": (
            "The goal does not say, and the preparation depends on it: tree models need no "
            "scaling, linear models do. `any` keeps the plan safe for both."
        ),
        "drop_id": "customer_id identifies a customer and says nothing about churn",
        "drop_country": "country holds a single value",
        "drop_date": "signup_date is a raw date string and no step can turn it into model input",
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
        "target_q": "Model hangi sütunu tahmin etsin?",
        "target_why": (
            "Hedef sütun belirtilmedi ve birden fazla sütun tahmin edilebilir, bu yüzden "
            "tahmin yürütmüyorum. `churned` evet/hayır sütunu ve müşterilerin ayrılmasıyla ilgili "
            "bir amaca uyuyor."
        ),
        "family_q": "Hazırlanan veriyle hangi tür modeli eğiteceksin?",
        "families": {"tree": "ağaç", "linear": "doğrusal", "any": "fark etmez"},
        "family_why": (
            "Amaç bunu söylemiyor ve hazırlık buna bağlı: ağaç modelleri ölçeklemeye ihtiyaç "
            "duymaz, doğrusal modeller duyar. `fark etmez` planı ikisi için de güvenli tutar."
        ),
        "drop_id": "customer_id bir müşteriyi tanımlar ve kayıp hakkında bir şey söylemez",
        "drop_country": "country tek bir değer taşıyor",
        "drop_date": (
            "signup_date ham bir tarih metni ve hiçbir adım onu model girdisine çeviremiyor"
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
    """The demo script cannot serve this dataset or this form.

    `code` and `params` let the interface word the error in its own language.
    """

    def __init__(self, message: str, code: str, **params: str) -> None:
        super().__init__(message)
        self.code = code
        self.params = params


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
                f"the columns {', '.join(sorted(missing))}",
                "demo_unsupported_file",
                columns=", ".join(sorted(missing)),
            )
        if target not in (None, TARGET):
            raise DemoError(
                f"the demo script is written for the target {TARGET!r}, not {target!r}",
                "demo_unsupported",
                target=target,
                demo_target=TARGET,
            )
        self._text = TEXTS.get(lang, TEXTS["en"])
        self._profile = profile
        self._target_given = target is not None
        self._family_given = model_family is not None
        self._family = model_family
        self._answers = answers
        self._calls = 0

    def generate[T: BaseModel](self, system: str, user: str, schema: type[T]) -> LLMResult[T]:
        # Without open questions the story starts at the first plan. A further call, if the plan
        # is still rejected, repeats the last one.
        story = ["plan_rejected", "plan_corrected"]
        if not (self._target_given and self._family_given):
            story.insert(0, "questions")
        step = story[min(self._calls, len(story) - 1)]
        self._calls += 1
        if step == "questions":
            response: dict = {"kind": "questions", "plan": None, "questions": self._questions()}
        else:
            response = {
                "kind": "plan",
                "plan": self._plan(rejected=step == "plan_rejected"),
                "questions": None,
            }
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
        if not self._family_given:
            questions.append(
                {
                    "id": "model_family",
                    "question": text["family_q"],
                    "options": list(text["families"].values()),
                    "recommended": text["families"]["any"],
                    "reason": text["family_why"],
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
        by_label = {label: family for family, label in text["families"].items()}
        family = self._family or by_label.get(answers.get("model_family", ""), "any")
        build = _rejected_steps if rejected else _corrected_steps
        return {
            "target": TARGET,
            "task_type": "classification",
            "model_family": family,
            "steps": build(text, scale=family != "tree"),
        }


def _drop_step(text: dict, *, ghost: bool) -> dict:
    columns = ["customer_id", "signup_date", "country"]
    reasons = [text["drop_id"], text["drop_date"], text["drop_country"]]
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


def _rejected_steps(text: dict, *, scale: bool) -> list[dict]:
    fixed = _fixed_steps(text)
    steps = [
        _drop_step(text, ghost=True),
        _impute_step(text, early=True),
        fixed["dedupe"],
        fixed["missing"],
        fixed["split"],
    ]
    if scale:
        steps.append(_scale_step(["age", "monthly_spend", "plan"], text["scale_bad"]))
    steps.append(fixed["encode"])
    return steps


def _corrected_steps(text: dict, *, scale: bool) -> list[dict]:
    fixed = _fixed_steps(text)
    steps = [
        _drop_step(text, ghost=False),
        fixed["dedupe"],
        fixed["split"],
        _impute_step(text, early=False),
        fixed["encode"],
    ]
    if scale:
        steps.append(_scale_step(["age", "monthly_spend"], text["scale_good"]))
    return steps
