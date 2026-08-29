from types import SimpleNamespace

import pytest

from scripts import check_cuda_devices


class _ProbeTensor:
    def add_(self, value):
        assert value == 1
        return self


class _FakeCuda:
    def __init__(self, count: int, *, failing_index: int | None = None) -> None:
        self.count = count
        self.failing_index = failing_index
        self.selected = []
        self.synchronized = []
        self.cache_cleared = False

    def device_count(self) -> int:
        return self.count

    def set_device(self, index: int) -> None:
        self.selected.append(index)
        if index == self.failing_index:
            raise RuntimeError("device busy or unavailable")

    def get_device_properties(self, index: int):
        return SimpleNamespace(name=f"fake-{index}", total_memory=1024)

    def synchronize(self, index: int) -> None:
        self.synchronized.append(index)

    def empty_cache(self) -> None:
        self.cache_cleared = True


def test_cuda_preflight_queries_pci_inventory(monkeypatch) -> None:
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        return SimpleNamespace(
            stdout="0, 00000000:65:00.0, GPU-test, NVIDIA A100\n"
        )

    monkeypatch.setattr(check_cuda_devices.subprocess, "run", fake_run)

    rows = check_cuda_devices.query_nvidia_smi_devices()

    assert rows == ["0, 00000000:65:00.0, GPU-test, NVIDIA A100"]
    command, kwargs = commands[0]
    assert command == [
        "nvidia-smi",
        "--query-gpu=index,pci.bus_id,uuid,name",
        "--format=csv,noheader",
    ]
    assert kwargs == {"check": True, "capture_output": True, "text": True}


def test_cuda_preflight_initializes_every_visible_device(monkeypatch) -> None:
    fake_cuda = _FakeCuda(count=2)
    allocations = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(check_cuda_devices.torch, "cuda", fake_cuda)
    monkeypatch.setattr(
        check_cuda_devices.torch,
        "ones",
        lambda size, device: allocations.append((size, device)) or _ProbeTensor(),
    )
    monkeypatch.setattr(
        check_cuda_devices,
        "query_nvidia_smi_devices",
        lambda: [
            "0, 00000000:17:00.0, GPU-good-0, NVIDIA A100",
            "1, 00000000:CA:00.0, GPU-good-1, NVIDIA A100",
        ],
    )

    check_cuda_devices.probe_cuda_devices(expected=2)

    assert fake_cuda.selected == [0, 1]
    assert fake_cuda.synchronized == [0, 1]
    assert allocations == [(1, "cuda:0"), (1, "cuda:1")]
    assert fake_cuda.cache_cleared is True


def test_cuda_preflight_rejects_an_incomplete_allocation(monkeypatch) -> None:
    monkeypatch.setattr(check_cuda_devices.torch, "cuda", _FakeCuda(count=1))

    with pytest.raises(RuntimeError, match="Expected 2 visible CUDA devices"):
        check_cuda_devices.probe_cuda_devices(expected=2)


def test_cuda_preflight_identifies_the_failing_pci_device(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
    monkeypatch.setattr(
        check_cuda_devices.torch,
        "cuda",
        _FakeCuda(count=2, failing_index=1),
    )
    monkeypatch.setattr(
        check_cuda_devices,
        "query_nvidia_smi_devices",
        lambda: [
            "0, 00000000:17:00.0, GPU-unused-0, NVIDIA A100",
            "1, 00000000:31:00.0, GPU-good, NVIDIA A100",
            "2, 00000000:4B:00.0, GPU-unused-2, NVIDIA A100",
            "3, 00000000:65:00.0, GPU-bad, NVIDIA A100",
        ],
    )
    monkeypatch.setattr(
        check_cuda_devices.torch,
        "ones",
        lambda size, device: _ProbeTensor(),
    )

    with pytest.raises(RuntimeError, match=r"cuda:1.*00000000:65:00.0"):
        check_cuda_devices.probe_cuda_devices(expected=2)


def test_cuda_preflight_maps_noncontiguous_visible_devices_by_physical_index(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
    inventory = [
        "0, 00000000:17:00.0, GPU-zero, NVIDIA A100",
        "1, 00000000:31:00.0, GPU-one, NVIDIA A100",
        "2, 00000000:4B:00.0, GPU-two, NVIDIA A100",
        "3, 00000000:65:00.0, GPU-three, NVIDIA A100",
    ]

    mapped = check_cuda_devices.map_visible_device_inventory(inventory, 2)

    assert "token=1" in mapped[0]
    assert "pci_bus_id=00000000:31:00.0" in mapped[0]
    assert "token=3" in mapped[1]
    assert "pci_bus_id=00000000:65:00.0" in mapped[1]


def test_cuda_preflight_does_not_guess_physical_rows_without_cvd(
    monkeypatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    inventory = ["0, 00000000:17:00.0, GPU-zero, NVIDIA A100"]

    mapped = check_cuda_devices.map_visible_device_inventory(inventory, 1)

    assert mapped == [
        "physical mapping unavailable (CUDA_VISIBLE_DEVICES is unset)"
    ]
    assert "17:00.0" not in mapped[0]
