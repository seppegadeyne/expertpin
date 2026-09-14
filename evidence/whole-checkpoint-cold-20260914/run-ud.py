"""Local physical run: existing guarded harness, UD/NCMOE40, client cap4G.
No production defaults changed. Run ONLY via run-gpu.sh's outer client scope.
"""
import importlib.util
import json
import os
from pathlib import Path
import time

ROOT = Path('/home/seppe/Projects/expertpin')
OUT = ROOT / '.hermes-work/daily-20260914'
CLIENT_UNIT = 'expertpin-daily-client-20260914.scope'
spec = importlib.util.spec_from_file_location('daily_cold', ROOT / 'evidence/coldcache-ab-20260911/run-guarded.py')
assert spec is not None and spec.loader is not None
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def client_point():
    cgroup = next(s[3:] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'))
    if not cgroup.endswith('/' + CLIENT_UNIT):
        raise RuntimeError('runner must be inside the dedicated 4G client scope')
    base = Path('/sys/fs/cgroup') / cgroup.lstrip('/')
    point = {k: (base / k).read_text().strip() for k in
             ('memory.current', 'memory.peak', 'memory.max', 'memory.swap.current', 'memory.events')}
    if int(point['memory.max']) != 4 * 1024**3:
        raise RuntimeError('client MemoryMax must be exactly 4G')
    if not 0 <= int(point['memory.current']) <= int(point['memory.peak']) <= 4 * 1024**3:
        raise RuntimeError('client RAM cap exceeded')
    events = dict(line.split() for line in point['memory.events'].splitlines())
    if any(int(events.get(k, 0)) for k in ('max', 'oom', 'oom_kill', 'oom_group_kill')):
        raise RuntimeError('client cap pressure/OOM event')
    point['cgroup'] = cgroup
    return point


original_env = h.clean_environment

def environment(*args, **kwargs):
    env = original_env(*args, **kwargs)
    env['NCMOE'] = '40'
    # Throughput probe retains its standard reasoning/greedy prompt, not the
    # reasoning-off Hermes code gate. Do not call this an E2E measurement.
    env['REASONING'] = 'auto'
    return env


original_report = h.verified_cold_report

def report(path, **kwargs):
    # All target shards and the shared drafter are separately measured at
    # a strict zero-resident-pages threshold, after the inherited idle guard.
    draft = original_report(h.clean_environment(os.environ, 'unused', 4, 'ud-q4kxl')['DRAFT_MODEL'], max_resident_ratio=0.0)
    (OUT / 'draft-cold-report.json').write_text(json.dumps(draft, indent=2) + '\n')
    if not draft['verified_cold']:
        raise RuntimeError('drafter not fully cold; refusing load')
    return original_report(path, max_resident_ratio=0.0)


class Run(h.Run):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        (OUT / 'server-unit.txt').write_text(self.scope + '\n')
        (OUT / 'run-path.txt').write_text(str(self.out) + '\n')

    def command(self, args, *rest, **kwargs):
        if args[:2] == ['systemctl', '--user'] and 'qli.service' in args and any(a in args for a in ('start', 'restart', 'stop')):
            raise RuntimeError('qli pause: no service mutations permitted')
        return super().command(args, *rest, **kwargs)

    def sample(self):
        super().sample()
        client = client_point()
        assert self.cgroup_path is not None
        server = {k: (self.cgroup_path / k).read_text().strip() for k in ('memory.current', 'memory.peak', 'memory.max')}
        if int(server['memory.max']) != 36 * 1024**3:
            raise RuntimeError('server cap must be 36G')
        upper = int(server['memory.peak']) + int(client['memory.peak'])
        record = dict(elapsed_s=time.monotonic() - self.start, client=client, server=server,
                      sum_of_kernel_peaks_bytes=upper)
        with (self.out / 'inclusive-ram.jsonl').open('a') as output:
            output.write(json.dumps(record) + '\n')
        if upper > 40 * 1024**3:
            raise RuntimeError('combined RAM cap exceeded')


if __name__ == '__main__':
    client_point()  # refuse raw launch, before touching services
    h.clean_environment = environment
    h.verified_cold_report = report
    h.Run = Run
    try:
        rc = h.main(['--checkpoint', 'ud-q4kxl', '--startup-n-max', '4', '--cold-cache'])
    finally:
        (OUT / 'client-final.json').write_text(json.dumps(client_point(), indent=2) + '\n')
    raise SystemExit(rc)
