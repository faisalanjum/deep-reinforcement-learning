"""Spawn-compatible local workers, including factories defined in notebooks."""
import time
import traceback
import cloudpickle
import numpy as np
import torch.multiprocessing as multiprocessing


class SerializableCallable:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def __getstate__(self):
        return cloudpickle.dumps(self.function)

    def __setstate__(self, payload):
        self.function = cloudpickle.loads(payload)


def join_workers(workers, timeout):
    deadline = time.monotonic() + timeout
    try:
        while any(worker.is_alive() for worker in workers):
            failed = [(w.pid, w.exitcode) for w in workers if w.exitcode not in (None, 0)]
            if failed:
                raise RuntimeError(f"Training worker failed: {failed}")
            if time.monotonic() > deadline:
                raise TimeoutError("Training workers exceeded the configured time limit")
            for worker in workers:
                worker.join(timeout=0.05)
        failed = [(w.pid, w.exitcode) for w in workers if w.exitcode != 0]
        if failed:
            raise RuntimeError(f"Training worker failed: {failed}")
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)


def _environment_worker(factory, kwargs, seed, connection):
    env = None
    try:
        env = factory(**kwargs, seed=seed)
        connection.send((True, None))
        while True:
            command, arguments = connection.recv()
            if command == "close":
                env.close(**arguments)
                env = None
                connection.send((True, None))
                return
            if command == "reset":
                result = env.reset(**arguments)
            elif command == "step":
                result = env.step(**arguments)
            elif command == "_past_limit":
                result = env._elapsed_steps >= env._max_episode_steps
            else:
                raise ValueError(f"Unknown worker command: {command}")
            connection.send((True, result))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            connection.send((False, traceback.format_exc()))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


class SpawnEnvironments:
    """Same reset/step interface as the book, with standalone spawned workers."""
    column_rewards = False

    def __init__(self, make_env_fn, make_env_kargs, seed, n_workers):
        self.n_workers = n_workers
        self.workers = []
        self.connections = []
        self.closed = False
        context = multiprocessing.get_context("spawn")
        try:
            for rank in range(n_workers):
                parent, child = context.Pipe()
                worker = context.Process(
                    target=_environment_worker,
                    args=(SerializableCallable(make_env_fn), make_env_kargs, seed + rank, child),
                    daemon=True,
                )
                worker.start()
                child.close()
                self.workers.append(worker)
                self.connections.append(parent)
            for rank in range(n_workers):
                self._receive(rank)
        except BaseException:
            self.close()
            raise

    def _receive(self, rank):
        connection = self.connections[rank]
        if not connection.poll(120):
            raise TimeoutError(f"Environment worker {rank} did not respond")
        try:
            succeeded, result = connection.recv()
        except EOFError as error:
            raise RuntimeError(f"Environment worker {rank} exited unexpectedly") from error
        if not succeeded:
            raise RuntimeError(result)
        return result

    def reset(self, rank=None, ranks=None, **kwargs):
        selected = [rank] if rank is not None else list(range(self.n_workers)) if ranks is None else list(ranks)
        for index in selected:
            self.connections[index].send(("reset", kwargs))
        observations = [self._receive(index) for index in selected]
        return observations[0] if rank is not None else np.stack(observations)

    def step(self, actions):
        if len(actions) != self.n_workers:
            raise ValueError("One action per worker is required")
        for rank, action in enumerate(actions):
            self.connections[rank].send(("step", {"action": action}))
        observations, rewards, dones, infos = zip(*(self._receive(rank) for rank in range(self.n_workers)))
        rewards = np.asarray(rewards, dtype=float)
        dones = np.asarray(dones, dtype=float)
        infos = np.asarray(infos, dtype=object)
        if self.column_rewards:
            rewards, dones, infos = rewards[:, None], dones[:, None], infos[:, None]
        return np.stack(observations), rewards, dones, infos

    def _past_limit(self):
        for connection in self.connections:
            connection.send(("_past_limit", {}))
        return np.asarray([self._receive(rank) for rank in range(self.n_workers)])[:, None]

    def close(self, **kwargs):
        if self.closed:
            return
        self.closed = True
        for connection, worker in zip(self.connections, self.workers):
            try:
                if worker.is_alive():
                    connection.send(("close", kwargs))
            except (EOFError, BrokenPipeError, OSError):
                pass
        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        for connection in self.connections:
            connection.close()


class ColumnSpawnEnvironments(SpawnEnvironments):
    column_rewards = True
