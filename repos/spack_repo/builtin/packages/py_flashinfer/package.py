# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack_repo.builtin.build_systems.python import PythonPackage

from spack.package import *


class PyFlashinfer(PythonPackage):
    """FlashInfer: Kernel Library for LLM Serving.

    Kernels for attention, sampling, quantization and GEMM. The package is
    pure python: every CUDA kernel is JIT-compiled on first use with nvcc and
    ninja, and cached under ~/.cache/flashinfer (override with
    FLASHINFER_WORKSPACE_BASE).
    """

    homepage = "https://github.com/flashinfer-ai/flashinfer"
    # Install the pure-python wheel instead of the sdist: the sdist uses a
    # custom PEP 517 build backend that requires apache-tvm-ffi at build time
    # and best-effort-builds NVSHMEM/NIXL expert-parallelism libraries. The
    # wheel ships the complete JIT payload (csrc/, include/ and vendored
    # cutlass/cccl/spdlog sources under flashinfer/data/), so it is
    # functionally identical. NOTE: PyPI file URLs are not stable across
    # versions, so future versions must carry their own url= kwarg.
    url = (
        "https://files.pythonhosted.org/packages/9d/dc/"
        "5367cb601fc9190cc75b4c141f5f7e9a224d1c26a1e983151823e02a25b3/"
        "flashinfer_python-0.6.16.post3-py3-none-any.whl"
    )
    list_url = "https://pypi.org/project/flashinfer-python"

    maintainers("thomas-bouvier")

    license("Apache-2.0")

    version(
        "0.6.16.post3",
        sha256="caf686b9b079abe1c9d65ab505698bd325e8072de40afd822f2c74f2ac3bc601",
    )

    depends_on("python@3.10:", type=("build", "run"))
    depends_on("py-setuptools", type="build")

    # Modules imported by `import flashinfer` (flashinfer/__init__.py,
    # flashinfer/jit/core.py, flashinfer/utils.py). The remaining upstream
    # dependencies (cuda-python, nccl4py, nvidia-cutlass-dsl,
    # nvidia-cudnn-frontend, cuda-tile) have no Spack packages; flashinfer
    # guards those imports, so the corresponding features (CuTe DSL kernels,
    # cuDNN attention, MoE expert parallelism) are unavailable instead of
    # broken.
    depends_on("py-apache-tvm-ffi@0.1.6:0.2", type="run")
    # https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post3/requirements.txt
    conflicts("^py-apache-tvm-ffi@0.1.8")
    conflicts("^py-apache-tvm-ffi@0.1.8.post0")
    depends_on("py-torch", type="run")
    depends_on("py-nvidia-ml-py", type="run")
    depends_on("py-numpy", type="run")
    depends_on("py-click", type="run")  # `flashinfer` CLI
    depends_on("py-tabulate", type="run")  # `flashinfer` CLI
    depends_on("py-einops", type="run")
    depends_on("py-filelock", type="run")  # JIT workspace locking
    depends_on("py-jinja2", type="run")  # jit/gemm kernel templates
    depends_on("py-packaging@24.2:", type="run")
    depends_on("py-requests", type="run")  # cubin downloads
    depends_on("py-tqdm", type="run")

    # The JIT toolchain is a *runtime* requirement: kernels are compiled when
    # first used. The driver (flashinfer/jit/cpp_ext.py) finds nvcc through
    # CUDA_HOME/CUDA_PATH and invokes ninja from PATH.
    depends_on("cuda", type="run")
    depends_on("ninja", type="run")

    def setup_run_environment(self, env: EnvironmentModifications) -> None:
        # JIT: nvcc is found via CUDA_HOME. Exposing nvcc on PATH is required
        # by consumers such as vLLM, which gate their flashinfer support on
        # `shutil.which("nvcc")` (vllm/utils/flashinfer.py).
        env.set("CUDA_HOME", self.spec["cuda"].prefix)
        env.prepend_path("PATH", join_path(str(self.spec["cuda"].prefix), "bin"))
        # The JIT driver invokes ninja from PATH.
        env.prepend_path("PATH", self.spec["ninja"].prefix.bin)
