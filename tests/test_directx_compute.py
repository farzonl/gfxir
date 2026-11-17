"""Tests for persistent DirectX 12 compute resources and runtime contract."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

from mlir_gpu import (
    Buffer,
    ComputeContext,
    ComputePipeline,
    DirectXComputeContext,
    Dispatch,
    ParameterBlock,
    PersistentBuffer,
    PersistentComputeError,
    PersistentPipeline,
    Target,
    Type,
    compile_pipeline,
    compute,
    global_invocation_id,
    load_compute_program,
    open_compute_context,
)
from mlir_gpu.viewer import ComputeProgram, Shader
from test_support import TOOLS


def _dummy_program(
    *,
    target: str = "directx",
    buffers: tuple[dict, ...] = (),
    parameters: dict | None = None,
    workgroup_size: tuple[int, int, int] = (64, 1, 1),
    grid: tuple[int, int, int] = (1, 1, 1),
) -> ComputeProgram:
    return ComputeProgram(
        compute=Shader(entry_point="main", code=b"DXBC_DUMMY"),
        target=target,
        workgroup_size=workgroup_size,
        buffers=buffers,
        dispatch=Dispatch(grid=grid),
        parameters=parameters,
    )


class _FakeBridgeLib:
    """Minimal in-memory C-ABI stub for testing DirectXComputeContext validation on any host."""

    def __init__(self, *, is_software: bool = True, device_name: str = "Microsoft Basic Render Driver") -> None:
        self._is_software = is_software
        self._device_name = device_name.encode("utf-8")
        self._next_handle = 100
        self._buffers: dict[int, int] = {}
        self._buffer_data: dict[int, bytearray] = {}
        self._pipelines: set[int] = set()
        self.dispatches = 0

    def _alloc_handle(self) -> int:
        self._next_handle += 1
        return self._next_handle

    def gfx_dx12_create_device(self, use_warp, enable_val, err_buf, err_len):
        return self._alloc_handle()

    def gfx_dx12_device_name(self, dev):
        return self._device_name

    def gfx_dx12_device_is_software(self, dev):
        return 1 if self._is_software else 0

    def gfx_dx12_last_error(self, dev):
        return b"stub error"

    def gfx_dx12_destroy_device(self, dev):
        pass

    def gfx_dx12_create_buffer(self, dev, size_bytes):
        val = int(getattr(size_bytes, "value", size_bytes))
        h = self._alloc_handle()
        self._buffers[h] = val
        self._buffer_data[h] = bytearray(val)
        return h

    def gfx_dx12_buffer_size(self, buf):
        return self._buffers.get(int(buf), 0)

    def gfx_dx12_upload_buffer(self, dev, buf, data, size_bytes, offset):
        h = int(getattr(buf, "value", buf))
        n = int(getattr(size_bytes, "value", size_bytes))
        off = int(getattr(offset, "value", offset))
        ptr = int(getattr(data, "value", data) or 0)
        if h in self._buffer_data and n > 0 and ptr:
            self._buffer_data[h][off : off + n] = ctypes.string_at(ptr, n)
        return 0

    def gfx_dx12_download_buffer(self, dev, buf, out_data, size_bytes, offset):
        h = int(getattr(buf, "value", buf))
        n = int(getattr(size_bytes, "value", size_bytes))
        off = int(getattr(offset, "value", offset))
        ptr = int(getattr(out_data, "value", out_data) or 0)
        if h in self._buffer_data and n > 0 and ptr:
            ctypes.memmove(ptr, bytes(self._buffer_data[h][off : off + n]), n)
        return 0

    def gfx_dx12_destroy_buffer(self, buf):
        h = int(getattr(buf, "value", buf))
        self._buffers.pop(h, None)
        self._buffer_data.pop(h, None)

    def gfx_dx12_create_pipeline(self, dev, dxil, dxil_size):
        h = self._alloc_handle()
        self._pipelines.add(h)
        return h

    def gfx_dx12_pipeline_root_constants_bytes(self, pipe):
        return 0

    def gfx_dx12_destroy_pipeline(self, pipe):
        self._pipelines.discard(int(getattr(pipe, "value", pipe)))

    def gfx_dx12_dispatch(self, dev, pipe, gx, gy, gz, rc_ptr, rc_len, bindings, num_bindings):
        self.dispatches += 1
        return 0


class TestDirectXComputeValidation(unittest.TestCase):
    """Host-only contract, lifetime, ownership, and error-propagation tests."""

    def test_nacelle_provider_identity_and_mismatch(self) -> None:
        for platform in ("linux", "darwin"):
            for name in ("Nacelle", "Microsoft Basic Render Driver"):
                fake = _FakeBridgeLib(device_name=name)
                with self.subTest(platform=platform, name=name), \
                     mock.patch("mlir_gpu.directx_compute.sys.platform", platform), \
                     mock.patch("mlir_gpu.directx_compute._load_bridge", return_value=fake):
                    if name != "Nacelle":
                        with self.assertRaisesRegex(PersistentComputeError, "fallback rejected"):
                            DirectXComputeContext(provider="nacelle")
                    else:
                        with DirectXComputeContext(provider="nacelle") as context:
                            self.assertEqual(context.provider, "nacelle")
                            self.assertTrue(context.is_software)

    def test_non_windows_rejected_without_cpu_fallback(self) -> None:
        with mock.patch("mlir_gpu.directx_compute.sys.platform", "linux"):
            with self.assertRaises(PersistentComputeError) as ctx:
                DirectXComputeContext()
            self.assertIn("requires Windows", str(ctx.exception))

    def test_invalid_provider_rejected(self) -> None:
        with mock.patch("mlir_gpu.directx_compute.sys.platform", "win32"):
            with self.assertRaises(PersistentComputeError):
                DirectXComputeContext(provider="swiftshader")
            with self.assertRaises(PersistentComputeError):
                DirectXComputeContext(provider="unknown")

    def test_missing_bridge_tools_raises_persistent_compute_error(self) -> None:
        with mock.patch("mlir_gpu.directx_compute.sys.platform", "win32"), mock.patch(
            "mlir_gpu.directx_compute._BRIDGE_LIB", None
        ), mock.patch(
            "mlir_gpu.directx_compute._find_or_build_bridge_dll",
            side_effect=PersistentComputeError("Neither clang++ nor MSVC cl.exe was found."),
        ):
            with self.assertRaises(PersistentComputeError) as ctx:
                DirectXComputeContext()
            self.assertIn("clang++", str(ctx.exception))

    def test_provider_mismatch_rejected(self) -> None:
        fake_hw = _FakeBridgeLib(is_software=False, device_name="NVIDIA GeForce RTX")
        with mock.patch("mlir_gpu.directx_compute.sys.platform", "win32"), mock.patch(
            "mlir_gpu.directx_compute._load_bridge", return_value=fake_hw
        ):
            with self.assertRaises(PersistentComputeError) as ctx:
                DirectXComputeContext(provider="warp")
            self.assertIn("Accidental hardware/driver fallback rejected", str(ctx.exception))

        fake_sw = _FakeBridgeLib(is_software=True, device_name="Microsoft Basic Render Driver")
        with mock.patch("mlir_gpu.directx_compute.sys.platform", "win32"), mock.patch(
            "mlir_gpu.directx_compute._load_bridge", return_value=fake_sw
        ):
            with self.assertRaises(PersistentComputeError) as ctx:
                DirectXComputeContext(provider="hardware")
            self.assertIn("Accidental hardware/driver fallback rejected", str(ctx.exception))

    def test_buffer_sizes_use_after_close_and_cross_device_validation(self) -> None:
        fake = _FakeBridgeLib()
        with mock.patch("mlir_gpu.directx_compute.sys.platform", "win32"), mock.patch(
            "mlir_gpu.directx_compute._load_bridge", return_value=fake
        ):
            ctx_a = DirectXComputeContext(provider="warp")
            ctx_b = DirectXComputeContext(provider="warp")
            try:
                self.assertIsInstance(ctx_a, ComputeContext)
                for bad_size in (0, -16, True, "16", None):
                    with self.assertRaises(PersistentComputeError):
                        ctx_a.create_buffer(bad_size)  # type: ignore[arg-type]

                buf_a = ctx_a.create_buffer(16)
                buf_b = ctx_b.create_buffer(16)
                self.assertIsInstance(buf_a, PersistentBuffer)

                with self.assertRaises(PersistentComputeError):
                    buf_a.upload(b"x" * 20)
                with self.assertRaises(PersistentComputeError):
                    buf_a.upload(b"x" * 8, offset=12)
                with self.assertRaises(PersistentComputeError):
                    buf_a.download(size_bytes=8, offset=12)

                prog = _dummy_program(
                    buffers=(
                        {"name": "src", "set": 0, "binding": 0, "access": "read_only"},
                        {"name": "dst", "set": 0, "binding": 1, "access": "write_only"},
                    )
                )
                pipe_a = ctx_a.compile_pipeline(prog)
                self.assertIsInstance(pipe_a, PersistentPipeline)

                # Cross-device buffer binding rejected
                with self.assertRaises(PersistentComputeError):
                    pipe_a.dispatch(buffers={"src": buf_b, "dst": buf_a})
                # Same buffer bound as both read_only and writable rejected
                with self.assertRaises(PersistentComputeError):
                    pipe_a.dispatch(buffers={"src": buf_a, "dst": buf_a})
                # Missing/extra buffer names rejected
                with self.assertRaises(PersistentComputeError):
                    pipe_a.dispatch(buffers={"src": buf_a})
                with self.assertRaises(PersistentComputeError):
                    pipe_a.dispatch(buffers={"src": buf_a, "dst": buf_a, "extra": buf_a})
                # Invalid grid rejected
                buf_a2 = ctx_a.create_buffer(16)
                with self.assertRaises(PersistentComputeError):
                    pipe_a.dispatch(buffers={"src": buf_a, "dst": buf_a2}, grid=(0, 1, 1))
                with self.assertRaises(PersistentComputeError):
                    pipe_a.dispatch(buffers={"src": buf_a, "dst": buf_a2}, grid=(65536, 1, 1))

                # Use-after-close on buffer
                buf_a.close()
                with self.assertRaises(PersistentComputeError):
                    buf_a.upload(b"1234")
                with self.assertRaises(PersistentComputeError):
                    buf_a.download()
                with self.assertRaises(PersistentComputeError):
                    pipe_a.dispatch(buffers={"src": buf_a, "dst": buf_a2})

                # Use-after-close on pipeline
                pipe_a.close()
                with self.assertRaises(PersistentComputeError):
                    pipe_a.dispatch(buffers={"src": buf_a2, "dst": buf_a2})
            finally:
                ctx_a.close()
                ctx_b.close()

            # Use-after-close on context
            with self.assertRaises(PersistentComputeError):
                ctx_a.create_buffer(16)
            with self.assertRaises(PersistentComputeError):
                ctx_a.compile_pipeline(_dummy_program())
            with self.assertRaises(PersistentComputeError):
                ctx_a.synchronize()

    def test_wrong_target_program_rejected_and_open_compute_context(self) -> None:
        fake = _FakeBridgeLib()
        with mock.patch("mlir_gpu.directx_compute.sys.platform", "win32"), mock.patch(
            "mlir_gpu.directx_compute._load_bridge", return_value=fake
        ):
            with open_compute_context("directx", provider="warp") as ctx:
                self.assertEqual(ctx.target, "directx")
                with self.assertRaises(PersistentComputeError):
                    ctx.compile_pipeline(_dummy_program(target="vulkan"))
            self.assertTrue(ctx.closed)

    def test_download_into_direct_and_validation(self) -> None:
        fake = _FakeBridgeLib()
        with mock.patch("mlir_gpu.directx_compute.sys.platform", "win32"), mock.patch(
            "mlir_gpu.directx_compute._load_bridge", return_value=fake
        ):
            with DirectXComputeContext(provider="warp") as ctx:
                buf = ctx.create_buffer(8, initial_data=b"\x01\x02\x03\x04\x05\x06\x07\x08")
                dst = bytearray(8)
                self.assertEqual(buf.download_into(dst), 8)
                self.assertEqual(bytes(dst), b"\x01\x02\x03\x04\x05\x06\x07\x08")

                sliced = bytearray(4)
                self.assertEqual(buf.download_into(memoryview(sliced), size_bytes=4, offset=2), 4)
                self.assertEqual(bytes(sliced), b"\x03\x04\x05\x06")

                strided = bytearray(8)
                strided_view = memoryview(strided)[::2]
                self.assertEqual(buf.download_into(strided_view, size_bytes=4, offset=4), 4)
                self.assertEqual(bytes(strided_view), b"\x05\x06\x07\x08")

                with self.assertRaises(PersistentComputeError):
                    buf.download_into(memoryview(b"12345678"))
                with self.assertRaises(PersistentComputeError):
                    buf.download_into(bytearray(4))
                buf.close()


_DispatchParams = ParameterBlock(
    "dispatch_params",
    (
        ("length", Type.U32),
        ("scale", Type.F32),
        ("bias", Type.F32),
    ),
)


@unittest.skipUnless(
    (sys.platform == "win32" and os.environ.get("MLIR_GPU_TEST_DIRECTX") == "1")
    or os.environ.get("MLIR_GPU_TEST_NACELLE") == "1",
    "Opt-in DirectX live test (set MLIR_GPU_TEST_DIRECTX=1 or MLIR_GPU_TEST_NACELLE=1)",
)
class TestDirectXComputeLive(unittest.TestCase):
    """Live DirectX 12 persistent compute execution tests."""

    def _compile_dxil_program(self, pipeline_obj: ComputePipeline, work_dir: Path, name: str) -> ComputeProgram:
        manifest = compile_pipeline(
            pipeline_obj,
            Target.DIRECTX,
            work_dir / name,
            TOOLS,
        )
        return load_compute_program(manifest)

    def test_two_dependent_dispatches_without_intermediate_readback_non_aligned(self) -> None:
        length = 137  # Intentionally not aligned to workgroup_size=64
        workgroup_size = (64, 1, 1)
        grid = ((length + 63) // 64, 1, 1)

        @compute(workgroup_size=workgroup_size)
        def kernel_mul_add(
            a: Buffer[Type.F32, "read_only"],
            b: Buffer[Type.F32, "read_only"],
            tmp: Buffer[Type.F32, "write_only"],
            params: _DispatchParams,
        ):
            gid = global_invocation_id().x
            if gid < params.length:
                tmp[gid] = a[gid] * b[gid] + params.bias

        @compute(workgroup_size=workgroup_size)
        def kernel_scale_add(
            tmp: Buffer[Type.F32, "read_only"],
            c: Buffer[Type.F32, "read_only"],
            out: Buffer[Type.F32, "write_only"],
            params: _DispatchParams,
        ):
            gid = global_invocation_id().x
            if gid < params.length:
                out[gid] = tmp[gid] * params.scale + c[gid]

        mul_add_pipe = ComputePipeline(compute=kernel_mul_add)
        scale_add_pipe = ComputePipeline(compute=kernel_scale_add)

        a_vals = [float(i) * 0.25 - 5.0 for i in range(length)]
        b_vals = [float((i % 7) + 1) * 0.5 for i in range(length)]
        c_vals = [float(i % 11) * -0.75 for i in range(length)]
        bias = 1.5
        scale = 2.0

        expected = [((a * b + bias) * scale + c) for a, b, c in zip(a_vals, b_vals, c_vals)]

        requested_provider = (
            "nacelle" if os.environ.get("MLIR_GPU_TEST_NACELLE") == "1" else
            "warp" if os.environ.get("MLIR_GPU_TEST_WARP") == "1" else None
        )
        validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1" and requested_provider != "nacelle"

        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            prog1 = self._compile_dxil_program(mul_add_pipe, work_dir, "mul_add")
            prog2 = self._compile_dxil_program(scale_add_pipe, work_dir, "scale_add")

            with open_compute_context("directx", provider=requested_provider, validation=validation) as ctx:
                assert isinstance(ctx, DirectXComputeContext)
                print(
                    f"[DirectXComputeLive] provider={ctx.provider} "
                    f"is_software={ctx.is_software} device={ctx.device_name!r}"
                )
                if requested_provider in ("warp", "nacelle"):
                    self.assertEqual(ctx.provider, requested_provider)
                    self.assertTrue(ctx.is_software)

                nbytes = length * 4
                buf_a = ctx.create_buffer(nbytes, initial_data=struct.pack(f"<{length}f", *a_vals))
                buf_b = ctx.create_buffer(nbytes, initial_data=struct.pack(f"<{length}f", *b_vals))
                buf_c = ctx.create_buffer(nbytes, initial_data=struct.pack(f"<{length}f", *c_vals))
                buf_tmp = ctx.create_buffer(nbytes)
                buf_out = ctx.create_buffer(nbytes)

                pipe1 = ctx.compile_pipeline(prog1)
                pipe2 = ctx.compile_pipeline(prog2)

                # Dispatch 1 writes buf_tmp on GPU; no host readback of buf_tmp.
                pipe1.dispatch(
                    buffers={"a": buf_a, "b": buf_b, "tmp": buf_tmp},
                    parameters={"length": length, "scale": 1.0, "bias": bias},
                    grid=grid,
                )

                # Dispatch 2 consumes buf_tmp directly on GPU and writes buf_out.
                pipe2.dispatch(
                    buffers={"tmp": buf_tmp, "c": buf_c, "out": buf_out},
                    parameters={"length": length, "scale": scale, "bias": 0.0},
                    grid=grid,
                )

                ctx.synchronize()
                raw_out = buf_out.download()
                actual = list(struct.unpack(f"<{length}f", raw_out))
                for idx, (got, exp) in enumerate(zip(actual, expected)):
                    self.assertAlmostEqual(
                        got,
                        exp,
                        places=5,
                        msg=f"Mismatch at index {idx}: got {got}, expected {exp}",
                    )

                # Reuse compiled pipeline with a second dispatch on different inputs.
                buf_a.upload(struct.pack(f"<{length}f", *[x + 1.0 for x in a_vals]))
                pipe1.dispatch(
                    buffers={"a": buf_a, "b": buf_b, "tmp": buf_tmp},
                    parameters={"length": length, "scale": 1.0, "bias": bias},
                    grid=grid,
                )
                pipe2.dispatch(
                    buffers={"tmp": buf_tmp, "c": buf_c, "out": buf_out},
                    parameters={"length": length, "scale": scale, "bias": 0.0},
                    grid=grid,
                )
                expected2 = [(((a + 1.0) * b + bias) * scale + c) for a, b, c in zip(a_vals, b_vals, c_vals)]
                actual2 = list(struct.unpack(f"<{length}f", buf_out.download()))
                for idx, (got, exp) in enumerate(zip(actual2, expected2)):
                    self.assertAlmostEqual(got, exp, places=5)

                # Verify partial-range download with offset.
                tail_bytes = buf_out.download(size_bytes=8, offset=(length - 2) * 4)
                tail_vals = list(struct.unpack("<2f", tail_bytes))
                self.assertAlmostEqual(tail_vals[0], expected2[-2], places=5)
                self.assertAlmostEqual(tail_vals[1], expected2[-1], places=5)
