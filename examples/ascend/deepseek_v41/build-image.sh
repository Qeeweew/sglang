#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
revision=$(git rev-parse HEAD)
cache=${DSV41_DEPS_DIR:-$PWD/build/dsv41-deps}
mkdir -p -- "$cache"
stage=$(mktemp -d "$cache/.build.XXXXXX")
trap 'rm -rf -- "$stage"' EXIT

fetch() {
    local url=$1 name=$2 digest=$3
    if [[ ! -f "$cache/$name" ]]; then
        curl --fail --location --retry 3 --silent --show-error "$url" --output "$stage/$name"
        printf '%s  %s\n' "$digest" "$stage/$name" | sha256sum --check --quiet
        mv -- "$stage/$name" "$cache/$name"
    fi
    ln -- "$cache/$name" "$stage/$name"
    printf '%s  %s\n' "$digest" "$stage/$name" | sha256sum --check --quiet
}

npu_release=https://github.com/sgl-project/sgl-kernel-npu/releases/download/2026.9.0.post9
for package in \
    'custom-ops-2026.9.0.post9-torch2.10.0-cann9.1.0-910b-aarch64.zip 7f40759e934324bca257114daa5659c9857ca5b5cfa533cf91b1c9d006e6df5b' \
    'ops-transformer-2026.9.0.post9-torch2.10.0-cann9.1.0-910b-aarch64.zip e99a001638f24c5097f2a1909bca46d7a62583d69e080bbb7e7f840f2aaae6d1' \
    'sgl-kernel-npu-2026.9.0.post9-torch2.10.0-py312-cann9.1.0-910b-aarch64.zip 21f8e3737c103a437f6bf552555237752992c6b771bdcb87b745e7c8dcc67d0b'; do
    read -r name digest <<< "$package"
    fetch "$npu_release/$name" "$name" "$digest"
done

triton=triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
fetch "https://github.com/triton-lang/triton-ascend/releases/download/v3.2.2/$triton" \
    "$triton" 259ef27091a11648b4813d5d590bb3118ba395418254b78a93f522bdf0468346
fetch 'https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC8EC952E2A0E1FBDC5090F6A2C277A0A352154E5' \
    ubuntu-toolchain-r-test.asc aa6927526a6e522ddeac8b82920f72098e558677255f86ce7278aa68d9df7e47

git archive --format=tar "$revision" | docker buildx build --load --platform linux/arm64 \
    --file examples/ascend/deepseek_v41/Dockerfile \
    --build-context "dsv41_deps=$stage" \
    --build-arg "SGLANG_REVISION=$revision" \
    --tag "${DSV41_IMAGE:-sglang-dsv41:910b}" "$@" -
