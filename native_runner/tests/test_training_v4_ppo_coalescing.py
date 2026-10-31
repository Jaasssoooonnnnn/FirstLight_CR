from concurrent.futures import Future
from threading import Timer

from native_runner.training.v4.async_cluster_self_play import _await_inference_batch_v4


def test_coalescing_keeps_late_and_failed_frames_for_the_normal_receiver():
    first, second, pending = Future(), Future(), Future()
    first.set_result('first')
    timer = Timer(0.002, lambda: second.set_exception(RuntimeError('worker failure')))
    timer.start()
    try:
        ready = _await_inference_batch_v4([first, second, pending], wait_seconds=0.02, gather_all=True)
    finally:
        timer.join()
    assert ready == {first, second}
    assert isinstance(second.exception(), RuntimeError)
    assert not pending.done()
    pending.set_result('next')
    assert _await_inference_batch_v4([pending], wait_seconds=0.02, gather_all=True) == {pending}


def test_full_batch_does_not_wait_for_the_deadline():
    first, second = Future(), Future()
    first.set_result(1)
    second.set_result(2)
    assert _await_inference_batch_v4([first, second], wait_seconds=0.0, gather_all=True) == {first, second}
