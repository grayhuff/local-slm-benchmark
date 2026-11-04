from slmbench import suites


def test_builtin_suite_shape():
    items = suites.load_builtin_suite()
    assert 5 <= len(items) <= 15
    assert len({i.id for i in items}) == len(items)
    assert {"reasoning", "knowledge", "instruction"} <= {i.category for i in items}


def test_parse_jsonl_aliases_and_errors():
    text = "\n".join(
        [
            '{"question": "2+2?", "answer": 4}',
            "not json",
            '{"prompt": ""}',
            '{"prompt": "hi", "scorer": "nope"}',
            "// comments and blank lines are ignored",
            "",
            '{"id": "dup", "prompt": "a"}',
            '{"id": "dup", "prompt": "b"}',
        ]
    )
    items, errors = suites.parse_jsonl(text, source="t")
    assert [i.prompt for i in items] == ["2+2?", "a", "b"]
    assert items[0].expected == 4 and items[0].id == "t-1"
    assert len({i.id for i in items}) == 3  # duplicate ids are made unique
    assert len(errors) == 3 and errors[0].startswith("line 2")


def test_custom_prompt_item():
    it = suites.custom_prompt_item("  Say hi  ", expected="  ", scorer=None)
    assert it.prompt == "Say hi" and it.expected is None


def test_example_file_parses():
    from slmbench.config import PROJECT_ROOT

    items, errors = suites.parse_jsonl((PROJECT_ROOT / "data" / "example_prompts.jsonl").read_text())
    assert items and not errors
