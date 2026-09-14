"""Audit the real physical-run artifacts; calculate, never synthesize, metrics."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path('/home/seppe/Projects/expertpin')
OUT = ROOT / '.hermes-work/daily-20260914'
run = Path((OUT / 'run-path.txt').read_text().strip())
summary = json.loads((run / 'summary.json').read_text())
env = json.loads((run / 'environment.json').read_text())
samples = [json.loads(s) for s in (run / 'samples.jsonl').read_text().splitlines()]
ram = [json.loads(s) for s in (run / 'inclusive-ram.jsonl').read_text().splitlines()]
client = json.loads((OUT / 'client-final.json').read_text())
draft = json.loads((OUT / 'draft-cold-report.json').read_text())
assert summary['status'] == 'completed' and summary['scope_stopped_verified'] and not summary['cleanup_errors']
assert summary['checkpoint'] == 'ud-q4kxl' and len(summary['requests']) == 2
assert env['NCMOE'] == '40' and env['RAM_BUDGET_GIB'] == '36' and env['DRAFT'] == '1'
assert env['GGML_CUDA_NO_PINNED'] == '1' and env['CTX'] == '8192'
assert len(summary['residency']['files']) == 4
for report in (summary['cold_cache_report'], draft):
    assert report['verified_cold'] and report['max_resident_ratio'] == 0
    assert report['snapshot']['resident_pages'] == 0
    assert report['snapshot']['pages'] == sum(s['pages'] for s in report['snapshot']['files'])
    assert report['drop']['bytes'] == sum(s['bytes'] for s in report['drop']['files'])
    assert all(s['resident_pages'] == 0 and s['error'] is None for s in report['snapshot']['files'])
for point in samples:
    assert int(point['memory.max']) == 36 * 1024**3
    assert int(point['memory.peak']) <= 40 * 1024**3
    assert point['vram_used_mib'] < 28 * 1024
    events = dict(s.split() for s in point['memory.events'].splitlines())
    assert all(int(events[k]) == 0 for k in ('max', 'oom', 'oom_kill', 'oom_group_kill'))
assert len(samples) == len(ram)
for point in ram:
    assert int(point['client']['memory.max']) == 4 * 1024**3
    assert point['sum_of_kernel_peaks_bytes'] == int(point['client']['memory.peak']) + int(point['server']['memory.peak'])
    assert point['sum_of_kernel_peaks_bytes'] <= 40 * 1024**3
for item in summary['requests']:
    body = json.loads((run / (item['label'] + '-response.json')).read_text())
    request = json.loads((run / (item['label'] + '-request.json')).read_text())
    assert request['max_tokens'] == 256 and request['temperature'] == 0
    assert body['timings'] == item['timings']
    assert body['usage']['completion_tokens'] == item['timings']['predicted_n'] == 256
    assert body['usage']['prompt_tokens'] == 68

server_peak = max(int(s['memory.peak']) for s in samples)
client_peak = max([int(client['memory.peak'])] + [int(s['client']['memory.peak']) for s in ram])
journal = subprocess.run(['journalctl', '--user', '-u', summary['scope'], '--no-pager', '-o', 'short-iso'],
                         capture_output=True, text=True, timeout=30)
(OUT / 'scope-journal.log').write_text(journal.stdout + journal.stderr)
full = (OUT / 'full.log').read_text()
failed, total = map(int, re.search(r'(\d+) tests failed out of (\d+)', full).groups())
result = dict(
    run=str(run.relative_to(ROOT)), code_commit=(run / 'commit.log').read_text().strip(),
    target_cold_pages=summary['residency']['pages'], target_resident_pages=0,
    draft_cold_pages=draft['snapshot']['pages'], draft_resident_pages=0,
    target_bytes=summary['cache_drop']['bytes'],
    target_first_shard_fraction=summary['cache_drop']['files'][0]['bytes']/summary['cache_drop']['bytes'],
    decode_tok_s=[r['timings']['predicted_per_second'] for r in summary['requests']],
    prefill_tok_s=[r['timings']['prompt_per_second'] for r in summary['requests']],
    acceptance=[r['acceptance_fraction'] for r in summary['requests']],
    goal_tok_s=20, decode_goal_met=[r['timings']['predicted_per_second'] >=20 for r in summary['requests']],
    server_kernel_peak_gib=server_peak/1024**3, client_kernel_peak_gib=client_peak/1024**3,
    sum_kernel_peaks_upper_gib=(server_peak+client_peak)/1024**3, ram_cap_gib=40,
    vram_sampled_peak_gib=max(s['vram_used_mib'] for s in samples)/1024, vram_cap_gib=28,
    server_swap_sampled_peak_gib=max(int(s['memory.swap.current']) for s in samples)/1024**3,
    client_swap_final_gib=int(client['memory.swap.current'])/1024**3,
    samples=len(samples), max_sample_gap_seconds=max(b['elapsed_s']-a['elapsed_s'] for a,b in zip(samples,samples[1:])),
    first_sample_seconds=samples[0]['elapsed_s'], last_sample_seconds=samples[-1]['elapsed_s'],
    elapsed_seconds=summary['elapsed_seconds'], mem_available_before_gib=summary['guard']['mem_available_bytes']/1024**3,
    cleanup_errors=summary['cleanup_errors'], full_ctest_passed=total-failed, full_ctest_total=total,
    limitations=['n=1 prompt; 2x256 tokens; reasoning auto; not the reasoning-off Hermes code gate',
                 'Sequential pre-load mincore snapshots, not atomic; concurrent readers not continuously excluded',
                 'RAM sums server/client cgroup kernel peaks; unrelated scope readers/shared page charges remain an attribution caveat',
                 'VRAM sampled, no continuous peak guarantee; swap reported separately',
                 'No new long-context, toolcalling, API-contract or Hermes E2E validation'])
(OUT / 'outcome.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
