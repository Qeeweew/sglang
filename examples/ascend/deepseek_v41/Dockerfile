ARG BASE_IMAGE=quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.12@sha256:cfa76a9bcd84dd81c487d6d26edb8b95e2140525891009ccb6009e5cd2544d8e
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG APT_MIRROR=https://mirrors.ustc.edu.cn
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV CC=/usr/bin/gcc-13 \
    CXX=/usr/bin/g++-13 \
    ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 \
    DSV41_PYTHON=/usr/local/python3.12.13/bin/python3 \
    DSV41_NATIVE_DIR=/opt/sglang/build/dsv41 \
    DSV41_JIT_CACHE_DIR=/var/cache/dsv41/deepjit \
    PIP_CACHE_DIR=/var/cache/pip \
    PYTHONUNBUFFERED=1

RUN --mount=type=bind,from=dsv41_deps,source=ubuntu-toolchain-r-test.asc,target=/tmp/toolchain.asc \
    set -eu; \
    sed -i "s|http://ports.ubuntu.com|${APT_MIRROR%/}|g" /etc/apt/sources.list; \
    install -m 0644 /tmp/toolchain.asc /usr/share/keyrings/ubuntu-toolchain-r-test.asc; \
    echo 'deb [arch=arm64 signed-by=/usr/share/keyrings/ubuntu-toolchain-r-test.asc] https://ppa.launchpadcontent.net/ubuntu-toolchain-r/test/ubuntu jammy main' \
        > /etc/apt/sources.list.d/ubuntu-toolchain-r-test.list; \
    apt-get -o Acquire::Retries=3 update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake curl gcc-13 g++-13 libdw-dev \
        libgl1 libglib2.0-0 libnuma-dev libsndfile1 pkg-config unzip; \
    rm -rf /var/lib/apt/lists/*

COPY examples/ascend/deepseek_v41/constraints.txt /opt/dsv41/constraints.txt
RUN --mount=type=cache,target=/var/cache/pip \
    python3 -m pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.10.0+cpu torchvision==0.25.0+cpu torchaudio==2.10.0+cpu
RUN --mount=type=cache,target=/var/cache/pip \
    python3 -m pip install -c /opt/dsv41/constraints.txt torch-npu==2.10.0.post6

COPY --chmod=0755 examples/ascend/deepseek_v41/entrypoint.sh /usr/local/bin/dsv41-entrypoint
RUN --mount=type=bind,from=dsv41_deps,target=/opt/deps \
    --mount=type=cache,target=/var/cache/pip \
    set -e; \
    source "$ASCEND_HOME_PATH/set_env.sh"; \
    unpacked=$(mktemp -d); \
    cd "$unpacked"; \
    for archive in /opt/deps/*.zip; do unzip -q "$archive"; done; \
    bash ./CANN-custom_ops-none-linux.aarch64.run --install-path="$ASCEND_HOME_PATH/opp"; \
    bash ./cann-ops-transformer-custom_linux-aarch64.run --install-path="$ASCEND_HOME_PATH/opp"; \
    dsv41-entrypoint python3 -m pip install -c /opt/dsv41/constraints.txt \
        /opt/deps/triton_ascend-*.whl ./custom_ops-*.whl ./deep_ep-*.whl \
        ./sgl_kernel_npu-*.whl ./torch_memory_saver-*.whl ./attentions-*.whl \
        memfabric-hybrid==1.2.1 memcache-hybrid==1.2.1 memfabric-zbal==1.1.3 \
        sglang-router==0.3.2; \
    cd /; \
    rm -rf "$unpacked"; \
    python3 -c 'import sysconfig; from pathlib import Path; site = Path(sysconfig.get_path("purelib")); ext, = (site / "deep_ep").glob("deep_ep_cpp*.so"); (site / ext.name).symlink_to(ext.relative_to(site))'

WORKDIR /opt/sglang
COPY . ./

ARG SGLANG_REVISION
ARG NATIVE_JOBS=8
LABEL org.opencontainers.image.revision="${SGLANG_REVISION}" \
      org.opencontainers.image.title="SGLang DeepSeek V4.1 for Ascend 910B"

# Match the Responses API types used by the SGLang server.
RUN --mount=type=cache,target=/var/cache/pip \
    set -e; \
    test -n "$SGLANG_REVISION"; \
    sed 's/"openai==2\.6\.1"/"openai==2.8.0"/' python/pyproject_npu.toml > python/pyproject.toml; \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="0.0.0+g${SGLANG_REVISION:0:12}" \
        dsv41-entrypoint python3 -m pip install -c /opt/dsv41/constraints.txt \
        -e './python[all_npu,tracing]' apache-tvm-ffi==0.1.15.post1 safetensors==0.8.0

RUN set -e; \
    mkdir -p "$DSV41_JIT_CACHE_DIR"; \
    dsv41-entrypoint python3 -m sglang.srt.hardware_backend.npu.dsv41.kernels.build \
        --backend native --output "$DSV41_NATIVE_DIR" --jobs "$NATIVE_JOBS" \
        || { cat "$DSV41_NATIVE_DIR"/{configure,build}.log 2>/dev/null; exit 1; }; \
    rm -rf "$DSV41_NATIVE_DIR/cmake"

ENTRYPOINT ["/usr/local/bin/dsv41-entrypoint"]
CMD ["bash", "examples/ascend/deepseek_v41/serve.sh"]
