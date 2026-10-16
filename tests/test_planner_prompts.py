import json
from importlib import resources

import pandas as pd
import pytest

from prep_agent.ops import list_ops
from prep_agent.ops.base import StatelessOpSpec, StatementCode
from prep_agent.plan import DropMissingTargetStep, Plan
from prep_agent.planner import (
    QARecord,
    Question,
    Revision,
    build_system_prompt,
    build_user_prompt,
)
from prep_agent.profiler import CsvOptions, profile_dataframe
from prep_agent.validators import ValidationIssue


@pytest.fixture
def profile():
    df = pd.DataFrame(
        {
            "age": [20, 30, None, 50, 60, 70, 80, 90, 25, 35, 45, 55],
            "plan": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
            "churned": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    return profile_dataframe(df, target="churned")


def user_prompt(profile, **overrides) -> str:
    args = {
        "goal": "prepare for churn prediction",
        "profile": profile,
        "target": "churned",
        "model_family": None,
        "qa_history": [],
        "revision": None,
        "question_rounds_left": 2,
    }
    return build_user_prompt(**{**args, **overrides})


def sample_question(**overrides) -> Question:
    data = {
        "id": "family",
        "question": "Which model family will you use?",
        "options": ["tree", "linear", "any"],
        "recommended": "any",
        "reason": "scaling depends on it",
    }
    return Question.model_validate({**data, **overrides})


# --- system prompt ---


def test_templates_are_package_data():
    root = resources.files("prep_agent") / "prompts"

    assert (root / "system.j2").is_file()
    assert (root / "user.j2").is_file()


def test_system_prompt_lists_every_registered_op_with_kind_and_description():
    prompt = build_system_prompt()

    for spec in list_ops():
        assert f"`{spec.name}` ({spec.kind})" in prompt
        assert spec.description in prompt


def test_system_prompt_follows_the_registry(monkeypatch):
    extra = StatelessOpSpec(
        name="shuffle_rows",
        step_model=DropMissingTargetStep,
        description="Put the rows in random order.",
        apply=lambda df, step, ctx: df,
        codegen=lambda step: StatementCode(()),
    )
    monkeypatch.setattr("prep_agent.planner.list_ops", lambda: [*list_ops(), extra])

    prompt = build_system_prompt()

    assert "`shuffle_rows` (stateless): Put the rows in random order." in prompt


def test_system_prompt_states_the_rules():
    prompt = build_system_prompt()

    for expected in (
        "exactly one split",  # order and leakage
        "fit on the train set only",
        "model_family",
        "Do not guess",  # ask instead of guessing
        "`reason`",
        "Use only columns listed in the profile",
        "never change it",  # the user's target
        "same kind",  # merge consecutive steps
        "allow_custom",  # free-text answers
        "consistent with the profile",  # reasons must not claim what the profile lacks
        "Every option must be achievable with the operations listed above",
        "binary columns stored as text",
        "ask all of them in one answer",
        "only for new questions that follow from the user's earlier answers",
    ):
        assert expected in prompt, expected


def test_system_prompt_lists_cast_to_category_from_the_registry():
    [spec] = [s for s in list_ops() if s.name == "cast_to_category"]

    prompt = build_system_prompt()

    assert f"- `cast_to_category` (stateless): {spec.description}" in prompt


def test_system_prompt_says_when_to_cast_numbers_to_categories():
    prompt = build_system_prompt()

    for expected in (
        "stored as numbers",
        "an hour, a month, a weekday, a code or a type number",
        "`cast_to_category`",
        "then encode",
        "Do not scale a cast column",
    ):
        assert expected in prompt, expected


def test_system_prompt_lets_a_tree_keep_ordered_codes_but_not_unordered_ones():
    prompt = build_system_prompt()

    for expected in (
        "If the model family is tree, numeric codes with a natural order",
        "(an hour, a month, a weekday)",
        "may stay numbers and need no `cast_to_category`",
        "Codes with no order (a type or category code such as a building class number)",
        "cast them for every model family",
        "For linear and any, cast the ordered ones too",
    ):
        assert expected in prompt, expected


def test_system_prompt_presents_the_low_cardinality_warning_as_a_hint_to_judge():
    prompt = build_system_prompt()

    assert "`low_cardinality_numeric`" in prompt
    assert "only a hint" in prompt
    assert "name of the column and the target" in prompt
    assert "a count (rooms, children" in prompt
    assert "not sure and the choice would change the result, ask" in prompt


def test_system_prompt_forbids_casting_the_target():
    prompt = build_system_prompt()

    assert "(drop_columns, cast_to_category, impute, encode, scale)" in prompt


def test_system_prompt_forbids_steps_the_data_does_not_need():
    prompt = build_system_prompt()

    assert "Do not add steps the data does not need" in prompt
    assert "`n_duplicate_rows`" in prompt
    assert "the target has missing values" in prompt
    assert "says that the step is unnecessary, leave that step out" in prompt


def test_system_prompt_requires_recommended_to_repeat_an_option_verbatim():
    prompt = build_system_prompt()

    assert "`recommended` must be the exact text of one of the `options`" in prompt
    assert "character for character" in prompt
    assert "do not reword, shorten or extend it" in prompt


def test_system_prompt_is_stable():
    assert build_system_prompt() == build_system_prompt()


# --- user prompt ---


def test_user_prompt_carries_goal_and_user_settings(profile):
    prompt = user_prompt(profile, model_family="tree")

    assert "prepare for churn prediction" in prompt
    assert "Target column: churned" in prompt
    assert "Model family: tree" in prompt


def test_user_prompt_says_when_target_and_family_are_unspecified(profile):
    prompt = user_prompt(profile, target=None)

    assert "Target column: not specified" in prompt
    assert "Model family: not specified" in prompt


def test_the_csv_settings_of_the_file_are_not_part_of_the_prompt(profile):
    with_csv = profile.model_copy(
        update={"csv": CsvOptions(sep=";", decimal=",", encoding="cp1254")}
    )

    prompt = user_prompt(with_csv)

    assert "cp1254" not in prompt
    assert user_prompt(profile) == prompt


def test_profile_is_compact_json_without_none_fields(profile):
    prompt = user_prompt(profile)

    [line] = [ln for ln in prompt.splitlines() if ln.startswith('{"n_rows"')]
    assert line == profile.model_dump_json(exclude_none=True)
    assert "null" not in line
    assert json.loads(line)["target"] == "churned"


def test_pii_sample_values_do_not_reach_the_prompt():
    df = pd.DataFrame(
        {
            "email": [f"person{i}@example.com" for i in range(12)],
            "phone": [f"+90 555 123 45 {i:02d}" for i in range(12)],
            "age": list(range(20, 32)),
            "y": [0, 1] * 6,
        }
    )

    prompt = user_prompt(profile_dataframe(df, target="y"))

    assert "person3@example.com" not in prompt
    assert "555 123 45" not in prompt
    assert "possible_pii" in prompt


def test_answered_questions_appear_in_the_prompt(profile):
    history = [QARecord(question=sample_question(), answer="tree")]

    prompt = user_prompt(profile, qa_history=history)

    assert "[family] Which model family will you use?" in prompt
    assert "Answer: tree" in prompt


def test_no_question_section_without_history(profile):
    assert "already asked" not in user_prompt(profile)


def test_revision_shows_previous_plan_and_issues(profile):
    plan = Plan.model_validate(
        {
            "target": "churned",
            "task_type": "classification",
            "model_family": "any",
            "steps": [{"op": "dedupe", "keep": "first", "reason": "r"}],
        }
    )
    issues = [
        ValidationIssue(
            code="stateful_before_split",
            severity="error",
            step_index=0,
            column=None,
            message="move it after the split",
        ),
        ValidationIssue(code="unknown_column", severity="error", column="ghost", message="m"),
    ]

    prompt = user_prompt(profile, revision=Revision(plan=plan, issues=issues))

    assert plan.model_dump_json(exclude_none=True) in prompt
    assert "code=stateful_before_split step=0 column=none: move it after the split" in prompt
    assert "code=unknown_column step=none column=ghost: m" in prompt


def test_no_revision_section_on_the_first_call(profile):
    assert "rejected" not in user_prompt(profile)


def test_questions_are_allowed_while_rounds_are_left(profile):
    prompt = user_prompt(profile, question_rounds_left=2)

    assert "2 round(s) left" in prompt
    assert "Do not ask any more questions" not in prompt


def test_exhausted_question_rounds_forbid_questions(profile):
    prompt = user_prompt(profile, question_rounds_left=0)

    assert "Do not ask any more questions" in prompt
    assert "recommend" in prompt
    assert "round(s) left" not in prompt


def test_plan_only_notice_is_absent_by_default(profile):
    assert "no more questions are allowed" not in user_prompt(profile)


def test_plan_only_notice_says_the_previous_answer_was_refused(profile):
    prompt = user_prompt(profile, question_rounds_left=0, plan_only=True)

    assert "Your previous answer asked questions, but no more questions are allowed" in prompt
    assert 'Answer with kind "plan" now' in prompt


def test_schema_error_notice_is_absent_by_default(profile):
    assert "was rejected" not in user_prompt(profile).replace("Previous plan, rejected", "")


def test_schema_error_notice_quotes_the_validation_error(profile):
    prompt = user_prompt(profile, schema_error="questions.0: recommended must be one of options")

    assert "## Your previous answer was rejected" in prompt
    assert "did not match the response schema: questions.0: recommended must be one" in prompt
    assert prompt.index("was rejected") < prompt.index("## What to do now")


def revision_with(plan, *, warnings) -> Revision:
    error = ValidationIssue(code="unknown_column", severity="error", column="ghost", message="m1")
    return Revision(plan=plan, issues=[error], warnings=warnings)


def test_revision_lists_warnings_under_their_own_heading(profile):
    plan = Plan.model_validate(
        {
            "target": "churned",
            "task_type": "classification",
            "model_family": "any",
            "steps": [{"op": "dedupe", "keep": "first", "reason": "r"}],
        }
    )
    warning = ValidationIssue(
        code="stratify_disabled", severity="warning", step_index=3, message="turn it on"
    )

    prompt = user_prompt(profile, revision=revision_with(plan, warnings=[warning]))

    fix, also = prompt.split("Warnings: fix these too if you can")
    assert "code=unknown_column" in fix
    assert "code=stratify_disabled step=3 column=none: turn it on" in also
    assert "code=stratify_disabled" not in fix


def test_revision_without_warnings_has_no_warning_heading(profile):
    plan = Plan.model_validate(
        {
            "target": "churned",
            "task_type": "classification",
            "model_family": "any",
            "steps": [{"op": "dedupe", "keep": "first", "reason": "r"}],
        }
    )

    assert "Warnings: fix these too" not in user_prompt(
        profile, revision=revision_with(plan, warnings=[])
    )


def test_system_prompt_tells_any_to_use_onehot_because_ordinal_suits_only_trees():
    prompt = build_system_prompt()

    [line] = [ln for ln in prompt.splitlines() if ln.startswith("- any:")]
    assert "onehot" in line
    assert "ordinal" in line and "only" in line and "tree" in line


def test_system_prompt_tells_any_to_scale_because_the_model_may_be_linear():
    prompt = build_system_prompt()

    [line] = [ln.lower() for ln in prompt.splitlines() if ln.startswith("- any:")]
    assert "skip optional" not in line
    assert "scale numeric columns" in line
    assert "robust" in line and "outliers" in line
    assert "linear" in line
    assert "does not hurt tree models" in line


def test_system_prompt_recommends_dropping_a_column_that_may_give_the_target_away():
    prompt = build_system_prompt()

    for expected in (
        "give the target away",
        "may not be known at prediction time",
        "measured in the same period as the target",
        "derived from the target",
        "recorded after the outcome is known",
        "`recommended` is always the option that drops the column",
        "the user can change it if they know the column is available at prediction time",
    ):
        assert expected in prompt, expected


def test_system_prompt_always_drops_datetime_and_text_columns_without_asking():
    prompt = build_system_prompt()

    for expected in (
        "No operation can use a datetime or free-text column",
        "a text column left at the end of the plan is an error",
        "always drop them and never ask about them",
        "the `reason` of the drop step says why",
    ):
        assert expected in prompt, expected
    assert "or ask when they could matter" not in prompt
    assert "looks like an identifier, text or a date" not in prompt
    assert "- a column that looks like an identifier could carry signal the user cares about;" in (
        prompt
    )


def test_system_prompt_has_a_language_section_that_covers_questions_and_step_reasons():
    prompt = build_system_prompt()

    assert "\n## Language\n" in prompt
    section = prompt.split("\n## Language\n", 1)[1].split("\n## ", 1)[0]
    for expected in (
        "Every text you write is in the language the user's goal is written in",
        "`question`, `options` and `reason` of the questions",
        "`reason` of every plan step",
        "Only the goal decides the language",
        "column names, sample values, the user's answers or these instructions",
        "If you are not sure, write English",
    ):
        assert expected in section, expected
    # The rule lives in one place, not at the end of the question paragraph.
    assert prompt.count("in the language of the user's goal") == 0


@pytest.mark.parametrize("goal", ["preparar los datos para predecir la baja", "müşteri kaybı için"])
def test_answers_and_revisions_remind_the_planner_of_the_language_without_quoting_the_goal(
    profile, goal
):
    plan = Plan.model_validate(
        {
            "target": "churned",
            "task_type": "classification",
            "model_family": "any",
            "steps": [{"op": "dedupe", "keep": "first", "reason": "r"}],
        }
    )
    issues = [ValidationIssue(code="unknown_column", severity="error", column="ghost", message="m")]
    after_answers = user_prompt(
        profile, goal=goal, qa_history=[QARecord(question=sample_question(), answer="tree")]
    )
    after_revision = user_prompt(profile, goal=goal, revision=Revision(plan=plan, issues=issues))

    for prompt in (after_answers, after_revision):
        [line] = [
            line for line in prompt.splitlines() if line.startswith("Keep writing in the language")
        ]
        assert "the language of the goal" in line
        assert goal not in line


def test_the_first_message_has_no_language_reminder_because_the_system_prompt_has_the_rule(
    profile,
):
    assert "Keep writing in the language" not in user_prompt(profile)


def test_the_language_section_points_to_the_language_field_of_the_answer():
    prompt = build_system_prompt()

    section = prompt.split("\n## Language\n", 1)[1].split("\n## ", 1)[0]
    assert "`language` field" in section
    assert "first" in section
