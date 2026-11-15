# DeepJIT in SGLang

The source identity and local changes are recorded in `SOURCE.json`. The
upstream MIT license is preserved in `LICENSE`.

This pinned header tree compiles and loads the Ascend C kernels in
`python/sglang/srt/hardware_backend/npu/dsv41/kernels/native`. CMake consumes it
through `DSV41_DEEPJIT_DIR`; the Python builder supplies the repository path.
SGLang's wheel/sdist hooks copy the same tree into package data for installed
builds. There is no independent Python package or download step at runtime.

Local changes retain the original project's 910B initialization, actor-specific
SDK PCH cache, compile locks, and draining of failed parallel preparations.
