FROM python:3.12-slim-bookworm

# pip 源可在构建时覆盖。默认用清华镜像；境外 CI（如 GitHub Actions）应传
# --build-arg PIP_INDEX_URL=https://pypi.org/simple。
# 注意：必须用 ARG 声明后再赋给 ENV，否则 --build-arg 不会生效。
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    BASELINE_CPU_THREADS=1 \
    CONTEST_WORK=/workspace \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}

WORKDIR /build

# LightGBM 的二进制扩展在导入时就 dlopen libgomp.so.1（OpenMP 运行库）。
# python:3.12-slim-bookworm 默认不带它，缺了会导致 `import lightgbm` 抛 OSError
# 而不是 ImportError。装在依赖安装之前，保证运行库先于 wheel 就位。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /build/requirements.txt
COPY sdk_src /build/sdk_src
RUN python3 -m pip install --no-cache-dir --retries 10 --timeout 120 \
    -r /build/requirements.txt \
    && python3 -m pip install --no-cache-dir --retries 10 --timeout 120 /build/sdk_src

COPY app/harness.py /app/harness.py
COPY app/contest_submission.py /app/contest_submission.py
COPY app/rsi_engine /app/rsi_engine

# uid 65534（nobody）运行：确保 /app 全部源码可读可遍历
RUN chmod -R a+rX /app

RUN mkdir -p /workspace && chown -R 65534:65534 /workspace
USER 65534:65534
WORKDIR /workspace

ENTRYPOINT []
CMD ["python3", "/app/harness.py", "describe"]
