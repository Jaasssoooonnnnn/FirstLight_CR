import pytest

from native_runner import cr_native_env as native


def test_transport_error_keeps_cause_and_omits_large_command_payload(monkeypatch):
    def timeout(*args, **kwargs):
        raise TimeoutError("timed out")
    monkeypatch.setattr(native.socket, "create_connection", timeout)
    client = native.NativeClashEnv()
    with pytest.raises(native.RunnerTransportError) as captured:
        client._request('configure {"battle":"' + 'x' * 10_000 + '"}')
    assert isinstance(captured.value.__cause__, TimeoutError)
    assert "'configure'" in str(captured.value)
    assert "timed out" in str(captured.value)
    assert len(str(captured.value)) < 200


def test_unsupported_persistent_transport_does_not_retry_forever(monkeypatch):
    calls = []
    def timeout(*args, **kwargs):
        calls.append(1)
        raise TimeoutError("timed out")
    monkeypatch.setattr(native.socket, "create_connection", timeout)
    client = native.NativeClashEnv()
    client._persistent_transport_requested = True
    client._persistent_transport_supported = False
    with pytest.raises(native.RunnerTransportError):
        client.status()
    assert len(calls) == 3

