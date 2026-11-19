"""Persistent q/frequency-tile workers for exact g-only current actions.

Workers own disjoint q/frequency tiles. All grid points are retained;
max_workers controls concurrency and does not change the physical quadrature.
Only O(M) temperature/direction vectors cross the process pipes.
"""
from __future__ import annotations
import multiprocessing as mp
from numbers import Real
import resource
import time
import traceback


class GOnlyWorkerFailure(RuntimeError):
    def __init__(self, record):
        super().__init__(record.get('error', 'G-only worker failure'))
        self.failure_record = record


def _weights(omega, temperatures, omega0, hbar, derivative=False):
    from . import probe_core as core
    function = core.bose_energy_temperature_derivative if derivative else core.bose_energy_weight
    return function(omega, temperatures, omega0=omega0, hbar=hbar)


def partition_frequency_tiles(nq, n_frequency, tile_size, max_workers):
    """Assign every q/positive-frequency point exactly once, without sampling."""
    if any(type(value) is not int or value < 1
           for value in (nq, n_frequency, tile_size, max_workers)):
        raise ValueError('Tile dimensions and worker count must be positive integers')
    tasks = tuple((q, start, min(start+tile_size, n_frequency))
                  for q in range(nq) for start in range(0, n_frequency, tile_size))
    count = min(max_workers, len(tasks))
    return tuple(tasks[worker::count] for worker in range(count))


def _worker(connection, tasks, data, centers, deadline, cache_cap, startup_temperatures):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from .g_only_point import GOnlyFrozenPoint
    begun = time.monotonic()
    records, harmonic_records, q_tiles = [], [], []
    harmonic = np.zeros(centers+2)
    calls = 0
    work = dict(harmonic_tiles_attempted=0, harmonic_tiles_completed=0,
                interacting_tiles_attempted=0, interacting_tiles_completed=0,
                current_tile_actions_attempted=0, current_tile_actions_completed=0)
    try:
        with threadpool_limits(limits=1, user_api='blas'):
            temperatures = np.r_[startup_temperatures[:2],
                                 np.full(centers, startup_temperatures[2])]
            q_slices = {}
            for q, start, stop in tasks:
                q_slices.setdefault(q, []).append((start, stop))
            for q, slices in q_slices.items():
                q_start = time.monotonic()
                lead = (data.harmonic.onsite[q], data.harmonic.forward[q],
                        data.harmonic.forward[q].conj().T)
                kernel = data.kernel.retarded[q, data.grid.positive_indices]
                tiles, harmonic_tiles = [], []
                for start, stop in slices:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('G-only factorization deadline reached')
                    sl = slice(start, stop)
                    arguments = dict(harmonic=lead, active_dofs=data.kernel.active_dofs,
                                     contacts=tuple(a[q, sl] for a in data.contact_retarded),
                                     center_layers=centers, frequencies=data.frequencies[sl])
                    work['harmonic_tiles_attempted'] += 1
                    h = GOnlyFrozenPoint(sigma_short=np.zeros_like(kernel[sl]), **arguments)
                    energy = _weights(data.frequencies[sl, None], temperatures[None, :],
                                      data.scale.omega0, data.scale.hbar)
                    harmonic += np.einsum('f,fb->b', data.weights[sl], h.currents(energy))
                    harmonic_tiles.append(dict(stats=h.stats(), diagnostics=h.diagnostics))
                    work['harmonic_tiles_completed'] += 1
                    del h
                    work['interacting_tiles_attempted'] += 1
                    point = GOnlyFrozenPoint(sigma_short=kernel[sl], **arguments)
                    if (point.stats()['g_only'] is not True
                            or point.stats()['exact_coupling_proof']['passed'] is not True):
                        raise ValueError('G-only qualified cache representation is missing')
                    tiles.append((sl, point))
                    owned = sum(p.stats()['stored_numeric_bytes'] for _, ts in q_tiles for _, p in ts)
                    owned += sum(p.stats()['stored_numeric_bytes'] for _, p in tiles)
                    if owned > cache_cap:
                        raise MemoryError('G-only retained numeric payload cap exceeded')
                    work['interacting_tiles_completed'] += 1
                q_tiles.append((q, tiles))
                records.append(dict(q=q, frequency_slices=[list(pair) for pair in slices],
                                    seconds=time.monotonic()-q_start,
                                    tiles=[p.stats() for _, p in tiles],
                                    diagnostics=[p.diagnostics for _, p in tiles]))
                harmonic_records.append(dict(q=q, frequency_slices=[list(pair) for pair in slices],
                                             tiles=harmonic_tiles))
            connection.send(dict(ok=True, phase='ready', q_indices=list(q_slices),
                                 frequency_tasks=[list(task) for task in tasks],
                                 harmonic=harmonic, seconds=time.monotonic()-begun,
                                 q_records=records, harmonic_records=harmonic_records,
                                 work_counts=dict(work),
                                 worker_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            while True:
                message = connection.recv()
                action = message.get('action')
                if action == 'stop':
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('G-only transport deadline reached')
                if action == 'stats':
                    connection.send(dict(ok=True, q_indices=list(q_slices), calls=calls,
                                         frequency_tasks=[list(task) for task in tasks],
                                         q_records=[dict(q=q, tiles=[p.stats() for _, p in ts])
                                                    for q, ts in q_tiles],
                                         work_counts=dict(work),
                                         worker_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
                    continue
                if action not in ('currents', 'jacobian'):
                    raise ValueError('Unknown g-only worker action')
                temperature = np.asarray(message['temperatures'], float)
                if (temperature.shape != (centers+2,) or not np.isfinite(temperature).all()
                        or np.any(temperature <= 0)):
                    raise ValueError('Invalid temperatures')
                if action == 'jacobian':
                    direction = np.asarray(message['direction'], float)
                    if direction.shape != temperature.shape or not np.isfinite(direction).all():
                        raise ValueError('Invalid tangent')
                current = np.zeros(centers+2)
                for q, tiles in q_tiles:
                    for sl, point in tiles:
                        if time.monotonic() >= deadline:
                            raise TimeoutError('G-only current/Jv tile deadline reached')
                        energy = _weights(data.frequencies[sl, None], temperature[None, :],
                                          data.scale.omega0, data.scale.hbar,
                                          derivative=action == 'jacobian')
                        if action == 'jacobian':
                            energy *= direction[None, :]
                        work['current_tile_actions_attempted'] += 1
                        current += np.einsum('f,fb->b', data.weights[sl], point.currents(energy))
                        work['current_tile_actions_completed'] += 1
                calls += 1
                connection.send(dict(ok=True, current=current))
    except BaseException:
        try:
            connection.send(dict(ok=False, q_indices=sorted({task[0] for task in tasks}),
                                 frequency_tasks=[list(task) for task in tasks],
                                 error=traceback.format_exc(),
                                 work_counts=dict(work), completed_callbacks=calls,
                                 other_worker_counts_unknown=True))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        connection.close()


class GOnlyParallelCurrents:
    def __init__(self, data, centers, *, deadline, tile_size=8,
                 max_cache_bytes=128*1024**3, max_workers=16, emit=None,
                 left_temperature=300.5, right_temperature=299.5,
                 equilibrium_temperature=300.):
        import numpy as np
        self.connections, self.processes = [], []
        self.emit = emit or (lambda *args: None)
        if (isinstance(centers, (bool, np.bool_)) or not isinstance(centers, (int, np.integer))
                or centers < 1 or type(tile_size) is not int or tile_size < 1
                or type(max_workers) is not int or max_workers < 1
                or type(max_cache_bytes) is not int or max_cache_bytes < 1
                or not np.isfinite(deadline) or deadline <= time.monotonic()):
            raise ValueError('Invalid g-only engine dimensions, cache cap or deadline')
        self.data, self.centers, self.deadline = data, int(centers), float(deadline)
        self.nq = int(data.kernel.retarded.shape[0])
        if self.nq < 1:
            raise ValueError('Empty q mesh')
        startup_temperatures = (left_temperature, right_temperature, equilibrium_temperature)
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
               or not np.isfinite(value) or value <= 0
               for value in startup_temperatures):
            raise ValueError('Startup Kelvin temperatures must be finite and positive')
        self.startup_temperatures = tuple(float(value) for value in startup_temperatures)
        self.task_groups = partition_frequency_tiles(
            self.nq, len(data.frequencies), tile_size, max_workers)
        self.worker_count = len(self.task_groups)
        self.q_groups = [sorted({task[0] for task in group}) for group in self.task_groups]
        self.calls, self.startup = {'currents': 0, 'jacobian': 0}, []
        self.started = time.monotonic()
        self.harmonic_current = np.zeros(self.centers+2)
        context = mp.get_context('fork')
        try:
            for group in self.task_groups:
                parent, child = context.Pipe()
                assigned_points = sum(stop-start for _, start, stop in group)
                cache_cap = max_cache_bytes*assigned_points//(self.nq*len(data.frequencies))
                process = context.Process(target=_worker, args=(child, group, data, self.centers,
                                          self.deadline, cache_cap, self.startup_temperatures))
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            for connection in self.connections:
                item = self._receive(connection)
                self.harmonic_current += item.pop('harmonic')/self.nq
                self.startup.append(item)
                self.emit('g_only_tile_group_ready', dict(centers=centers, q_indices=item['q_indices'],
                                                       frequency_tasks=item['frequency_tasks'],
                                                       seconds=item['seconds']))
            self.harmonic_current *= data.scale.hbar*data.scale.omega0**2
            self.setup_seconds = time.monotonic()-self.started
        except BaseException:
            self.close()
            raise

    def _receive(self, connection):
        process = self.processes[self.connections.index(connection)]
        while not connection.poll(min(1., max(0., self.deadline-time.monotonic()))):
            if time.monotonic() >= self.deadline:
                raise TimeoutError('G-only worker response deadline reached')
            if not process.is_alive():
                raise RuntimeError('G-only worker died before result; exit='+str(process.exitcode))
        try:
            result = connection.recv()
        except EOFError as error:
            raise RuntimeError('G-only worker exited without result') from error
        if not result.get('ok'):
            raise GOnlyWorkerFailure(result)
        return result

    def _action(self, action, temperatures, direction=None):
        import numpy as np
        message = dict(action=action, temperatures=np.asarray(temperatures, float))
        if direction is not None:
            message['direction'] = np.asarray(direction, float)
        for connection in self.connections:
            connection.send(message)
        total = np.zeros(self.centers+2)
        for connection in self.connections:
            total += self._receive(connection)['current']/self.nq
        self.calls[action] += 1
        self.emit('matrix_free_callback_complete', dict(centers=self.centers, action=action,
                  calls=dict(self.calls), elapsed_seconds=time.monotonic()-self.started))
        return total*self.data.scale.hbar*self.data.scale.omega0**2

    def currents(self, temperatures):
        return self._action('currents', temperatures)

    def jacobian_action(self, temperatures, direction):
        return self._action('jacobian', temperatures, direction)

    def stats(self):
        for connection in self.connections:
            connection.send(dict(action='stats'))
        return dict(callbacks=dict(self.calls), workers=[self._receive(c) for c in self.connections],
                    startup=self.startup, setup_seconds=self.setup_seconds,
                    workers_count=self.worker_count, q_points=self.nq,
                    q_groups=self.q_groups, blas_threads=1, g_only=True,
                    frequency_task_groups=[[list(task) for task in group] for group in self.task_groups],
                    positive_frequency_points=len(self.data.frequencies),
                    startup_temperatures_kelvin=list(self.startup_temperatures),
                    point_partition='disjoint q/frequency tiles; global transverse average 1/Nq',
                    harmonic_startup_work_recorded_separately=True)

    def close(self):
        for connection in self.connections:
            try:
                connection.send(dict(action='stop'))
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=.2)
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=2.)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.)
        for connection in self.connections:
            connection.close()
        self.connections, self.processes = [], []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
