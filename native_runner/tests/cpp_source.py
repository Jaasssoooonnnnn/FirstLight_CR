"""Inspect C++ statements without depending on clang-format whitespace."""

import re


_TOKEN = re.compile(
    r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    r'|\w+|::|->|&&|\|\||==|!=|<=|>=|\+\+|--|<<|>>|[^\s]',
    re.S,
)


def contains_cpp(source: str, statement: str) -> bool:
    def tokens(text: str) -> str:
        return " " + " ".join(
            match[0] for match in _TOKEN.finditer(text)
            if not match[0].startswith(("//", "/*"))
        ) + " "

    return tokens(statement) in tokens(source)


def cpp_function(source: str, name: str) -> str:
    match = re.search(
        r"(?m)^[^\n;{}]*\b" + re.escape(name) + r"\s*\([^;{}]*\)\s*\{", source
    )
    assert match is not None, f"Missing C++ definition: {name}"
    depth = 1
    for token in _TOKEN.finditer(source, match.end()):
        if token[0] == "{":
            depth += 1
        elif token[0] == "}":
            depth -= 1
            if depth == 0:
                return source[match.start():token.end()]
    raise AssertionError(f"Unclosed C++ definition: {name}")
