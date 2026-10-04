import json
import os
import shutil
import subprocess
import sys

import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_parse_env_value_handles_comments_and_quotes():
    parse = config.parse_env_value
    assert parse("  abc   # 行尾注释") == "abc"
    assert parse("") == ""
    assert parse("   # 只有注释") == ""
    assert parse('"a # b"  # c') == "a # b"
    assert parse("'x y'") == "x y"
    assert parse("http://h/p#frag") == "http://h/p#frag"


def test_load_dotenv_does_not_override_existing(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text(
        "# c\nexport A_KEY=1   # note\nB_KEY=\nC_KEY=\"q#1\"\nKEEP=file\nnot a line\n",
        encoding="utf-8",
    )
    for k in ("A_KEY", "B_KEY", "C_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("KEEP", "env")
    config.load_dotenv(str(f))
    try:
        assert os.environ["A_KEY"] == "1"
        assert os.environ["B_KEY"] == ""
        assert os.environ["C_KEY"] == "q#1"
        assert os.environ["KEEP"] == "env"
    finally:
        for k in ("A_KEY", "B_KEY", "C_KEY"):
            os.environ.pop(k, None)


def test_endpoints_come_from_dotenv_in_clean_process(tmp_path):
    (tmp_path / "eval").mkdir()
    shutil.copy(os.path.join(ROOT, "eval", "config.py"), tmp_path / "eval" / "config.py")
    (tmp_path / ".env").write_text("ASR_PLATFORM_URL=http://dotenv.test:1  # c\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("ASR_")}
    out = subprocess.run(
        [sys.executable, "-c", "import json,config;print(json.dumps(config.ENDPOINTS))"],
        cwd=tmp_path / "eval", env=env, capture_output=True, text=True, check=True,
    ).stdout
    assert json.loads(out)["platform"] == "http://dotenv.test:1"
