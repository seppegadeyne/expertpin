import importlib.util
from pathlib import Path
from unittest.mock import patch
import subprocess

spec = importlib.util.spec_from_file_location('physical', Path(__file__).with_name('run-physical.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
for states, fail_stop, expected_kill, fail_final in [(['inactive'],False,False,False),(['active','inactive'],True,True,False),(['active','active'],True,True,True)]:
    calls=[]
    def command(args,name,**kwargs):
        calls.append(name)
        if name=='cleanup-cgroup': return ''
        if name=='scope-stop' and fail_stop: raise subprocess.TimeoutExpired(args,20)
        if name=='scope-state': return states.pop(0)
        return ''
    with patch.object(m,'cmd',side_effect=command):
        try:
            assert m.stop_scope('unit.scope')=='inactive'
            assert not fail_final
        except RuntimeError:
            assert fail_final
    assert ('scope-kill' in calls)==expected_kill
print('cleanup: 3 mocked termination scenarios passed; no service/GPU actions')
