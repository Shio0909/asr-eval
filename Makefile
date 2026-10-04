.PHONY: test requirements run

test:
	uv run --with pytest pytest tests/ -q

# 从 uv.lock 导出 Docker 镜像用的 requirements.txt
requirements:
	uv export --frozen --no-dev --no-hashes --no-emit-project -o requirements.txt

run:
	uv run python dashboard/server.py
