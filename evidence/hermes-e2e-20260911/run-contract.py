#!/usr/bin/env python3
"""Clean repeated MTP measurement. Import/--help are CPU-only.

Run separately at startup depths 4, 8, 16: recurrent checkpoints are startup-bounded.
Each invocation repeats the same 256-token request twice at its startup depth.
Depth 0 disables the draft model entirely for target-only correctness isolation.
Lifecycle derived from ../expert-offset-trace/run-guarded.py; no tracing/dmon.
"""
import argparse
from datetime import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from typing import Any
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[2]
PREP = '/home/seppe/.hermes/profiles/expertpin/scripts/gpu-host-prep.sh'
DEPTHS = (4, 8, 16)
TOKENS = 256
WORK_SECONDS = 660  # Reserve >3 minutes of the 15-minute bound for bounded cleanup.
REQUEST_SECONDS = 180
RAM_BUDGET_GIB = 36
PROMPT = 'Explain how a bounded expert cache handles RAM and NVMe misses in detail.'
TRACE_PREFIXES = ('GGML_MOE_TRACE', 'GGML_MOE_GPU_TRACE', 'GGML_CUDA_TRANSFER_TRACE')
GRAPH_DISABLE = ('GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_NO_GRAPHS')
# Checkpoint aliases for A/B: shared drafter, identical serving configuration;
# only MODEL_DIR/MODEL differ. The reference is the 197 GiB AD-4.27bpw Q4_K_M
# shard set; ps-iq2xxs is PeasantSmith's community 75.2 GiB IQ2_XXS GGUF
# (SHA-256 verified 2026-09-11 against the HF model card before first load).
CHECKPOINTS = {
    'reference': ('/home/seppe/Models/qwen3.8-flash-next/AD-4.27bpw-Q4_K_M-M64',
                  'Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf'),
    'ps-iq2xxs': ('/home/seppe/Models/qwen3.8-flash-next-ps-iq2xxs',
                  'Qwen3.8-Flash-Next-IQ2_XXS.gguf'),
}
DEFAULT_CHECKPOINT = 'reference'
# Tool-call JSON quality gate (state-file "VOLGENDE" item 2, first half):
# does the checkpoint still produce schema-exact tool calls and a clean
# follow-up answer after a tool round? Target-only (DRAFT=0) so gate quality
# measures checkpoint quantization, not drafter interaction. Both checkpoints
# run the identical scenario; the verdict is per-checkpoint PASS/FAIL.
GATES = ('throughput', 'toolcall', 'needle', 'contract', 'errors')
TOOLCALL_TOKENS = 512
TOOLCALL_TOKENS_ROUND2 = 512
# Needle-recall gate (quality gate second half): one code needle at
# start/middle/end of a deterministic filler haystack sized ~NEEDLE_TARGET
# tokens via the /tokenize endpoint. Positions run end->middle->start so each
# request shares the previous prefix and cache_prompt=True can reuse it.
NEEDLE_PARAGRAPH = ('Project Aurora maintains a private access code for internal audits. '
                    'The code is 7391. It is listed here for completeness of the records.')
NEEDLE_QUESTION = ('According to the reference document above, what is the private access code '
                   'of Project Aurora? Answer with the four-digit code only.')
NEEDLE_CODE = '7391'
NEEDLE_TOKENS = 512
NEEDLE_TARGET_TOKENS = 2048
NEEDLE_TOKEN_TOLERANCE = 0.10
NEEDLE_POSITIONS = {'start': 0.1, 'middle': 0.5, 'end': 0.9}
NEEDLE_REQUEST_SECONDS = 300
NEEDLE_WORK_SECONDS = 960
TOOL_SCHEMA = {
    'type': 'object',
    'properties': {
        'city': {'type': 'string', 'enum': ['Ghent', 'Brussels', 'Antwerp']},
        'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']},
    },
    'required': ['city', 'unit'],
}
TOOL_RESULT = {'city': 'Ghent', 'unit': 'celsius', 'temperature_c': 17.5, 'condition': 'cloudy'}


def validate_plan(startup_nmax, checkpoint=DEFAULT_CHECKPOINT, gate='throughput'):
    if gate not in GATES:
        raise ValueError('unknown gate: ' + repr(gate))
    if gate in ('toolcall', 'needle', 'contract', 'errors'):
        if startup_nmax != 0:
            raise ValueError(gate + ' gate requires --startup-n-max 0 (target-only)')
        if checkpoint not in CHECKPOINTS:
            raise ValueError('unknown checkpoint: ' + repr(checkpoint))
        return
    if type(startup_nmax) is not int or startup_nmax not in (0, 4, 8, 16):
        raise ValueError('startup n_max must be 0 (target-only), 4, 8 or 16')
    if checkpoint not in CHECKPOINTS:
        raise ValueError('unknown checkpoint: ' + repr(checkpoint))


def clean_environment(inherited, scope, startup_nmax=4, checkpoint=DEFAULT_CHECKPOINT):
    env = {k: v for k, v in inherited.items()
           if not k.startswith(TRACE_PREFIXES) and k not in GRAPH_DISABLE and k != 'EXPERTPIN_VERIFIER_TRACE'
           and not k.startswith('LLAMA_ARG_')}
    validate_plan(startup_nmax, checkpoint)
    model_dir, model_name = CHECKPOINTS[checkpoint]
    env.update(GGML_CUDA_NO_PINNED='1', DRAFT=str(int(startup_nmax != 0)), DRAFT_NMAX=str(startup_nmax),
               DRAFT_MODEL='/home/seppe/Models/qwen3.8-flash-next/mtp-drafter/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf',
               MODEL_DIR=model_dir, MODEL=model_dir + '/' + model_name,
               CTX='8192', NCMOE='36', NGL='99', THREADS='16', KVT='q8_0',
               RAM_BUDGET_GIB='36', GPU_NEED_GIB='24', CACHE_RAM_MIB='512',
               PORT='8102', BIN_DIR=str(ROOT / 'build-sm120/bin'), FORCE='0', PINNED='0',
               EXPERT_CACHE_SIM_MIB='0', EXPERT_STATS_FILE='', MANIFEST='', RESIDENT='0',
               EXPERTPIN_SCOPE_UNIT=scope, DRY='0')
    return env


def payload(nmax):
    validate_plan(nmax)
    result = {'messages': [{'role': 'user', 'content': PROMPT}], 'max_tokens': TOKENS,
            'temperature': 0.0, 'seed': 42, 'stream': False, 'cache_prompt': False,
            'ignore_eos': True}
    if nmax:
        result['speculative.n_max'] = nmax
    return result


def output_fields(data):
    message = data['choices'][0]['message']
    result = {k: message.get(k) for k in ('content', 'reasoning_content')}
    if any(v is not None and not isinstance(v, str) for v in result.values()):
        raise ValueError('non-text response field')
    return result


def toolcall_payload():
    """Forced single-tool call: exact name, schema-valid arguments, tool_call id.

    tool_choice is the STRING "required", not the OpenAI object form: this
    server build parses tool_choice via json_value(..., std::string) and would
    silently degrade an object to "auto" (server-common.h json_value +
    server-common.cpp:651). With exactly one tool bound, "required" is
    equivalent to forcing get_weather (grammar min_calls=1)."""
    prompt = ('You have access to the get_weather tool. Do not answer from memory. '
              'Use the get_weather tool to fetch the current weather in Ghent. '
              'Call get_weather with city "Ghent" and unit "celsius". '
              'After you receive the tool result, answer the user and include the '
              'numeric temperature value in your final answer.')
    return {'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': TOOLCALL_TOKENS, 'temperature': 0.0, 'seed': 42,
            'stream': False, 'cache_prompt': False,
            'tools': [{'type': 'function', 'function': {
                'name': 'get_weather', 'description': 'Get the current weather for a city.',
                'parameters': TOOL_SCHEMA}}],
            'tool_choice': 'required'}


def toolcall_round2_payload(round1_payload, tool_call_id, arguments, tool_result_json):
    """Feed the tool result back and ask for a plain final answer (no tools bound)."""
    if not isinstance(tool_call_id, str) or not tool_call_id.strip():
        raise ValueError('empty tool_call_id')
    parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
    if not isinstance(parsed, dict) or sorted(parsed) != sorted(TOOL_SCHEMA['required']):
        raise ValueError('round-1 arguments do not satisfy the tool schema keys')
    if parsed.get('city') != tool_result_json.get('city'):
        raise ValueError('round-1 arguments city does not match the tool result')
    messages = [dict(m) for m in round1_payload['messages']]
    messages.append({'role': 'assistant', 'content': None, 'tool_calls': [{
        'id': tool_call_id, 'type': 'function',
        'function': {'name': 'get_weather', 'arguments': arguments}}]})
    messages.append({'role': 'tool', 'tool_call_id': tool_call_id,
                     'content': json.dumps(tool_result_json)})
    return {'messages': messages, 'max_tokens': TOOLCALL_TOKENS_ROUND2,
            'temperature': 0.0, 'seed': 42, 'stream': False, 'cache_prompt': False}


def _validated_choice(data, expected_finish):
    if not isinstance(data, dict) or 'error' in data:
        raise RuntimeError('invalid/error completion')
    usage, choices, timings = data.get('usage'), data.get('choices'), data.get('timings')
    if (not isinstance(usage, dict) or type(usage.get('completion_tokens')) is not int
            or usage['completion_tokens'] <= 0 or not isinstance(choices, list)
            or len(choices) != 1 or not isinstance(choices[0], dict)
            or choices[0].get('finish_reason') != expected_finish):
        raise RuntimeError(f'completion is not a single {expected_finish!r} choice')
    if not isinstance(timings, dict):
        raise RuntimeError('missing timings')
    for key in ('prompt_ms', 'predicted_ms', 'predicted_per_second'):
        value = timings.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise RuntimeError('invalid timing: ' + key)
    message = choices[0].get('message')
    if not isinstance(message, dict):
        raise RuntimeError('missing message')
    return message


def validate_toolcall(data):
    """Round 1: exactly one get_weather call, arguments exact per TOOL_SCHEMA."""
    message = _validated_choice(data, 'tool_calls')
    calls = message.get('tool_calls')
    if not isinstance(calls, list) or len(calls) != 1:
        raise RuntimeError('expected exactly one tool call')
    call = calls[0]
    function = call.get('function') if isinstance(call, dict) else None
    if not isinstance(function, dict) or function.get('name') != 'get_weather':
        raise RuntimeError('tool call is not get_weather')
    call_id = call.get('id')
    if not isinstance(call_id, str) or not call_id.strip():
        raise RuntimeError('missing tool_call id')
    if call.get('type') not in (None, 'function'):
        raise RuntimeError('unexpected tool call type')
    try:
        arguments = json.loads(function.get('arguments', ''))
    except (json.JSONDecodeError, TypeError):
        raise RuntimeError('tool call arguments are not valid JSON') from None
    if not isinstance(arguments, dict):
        raise RuntimeError('tool call arguments are not a JSON object')
    if sorted(arguments) != sorted(TOOL_SCHEMA['required']):
        raise RuntimeError('tool call arguments keys do not match the schema exactly')
    if arguments['city'] != 'Ghent' or arguments['unit'] != 'celsius':
        raise RuntimeError('tool call argument values are wrong')
    return {'tool_name': function['name'], 'arguments': arguments, 'tool_call_id': call_id}


def validate_round2(data):
    """Round 2: clean stop with non-empty text; no new tool calls."""
    message = _validated_choice(data, 'stop')
    content = message.get('content')
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError('round-2 answer is empty')
    if message.get('tool_calls') is not None:
        raise RuntimeError('round-2 answer contains unexpected tool calls')
    return {'content': content}


def gate_verdict(results):
    """Per-checkpoint PASS requires both rounds valid with the exact scenario values."""
    failures = []
    for name, result in sorted(results.items()):
        if not isinstance(result, dict) or 'round1' not in result or 'round2' not in result:
            failures.append(f'{name}: incomplete rounds')
            continue
        first = result['round1']
        if (first.get('tool_name') != 'get_weather'
                or first.get('arguments') != {'city': 'Ghent', 'unit': 'celsius'}
                or not isinstance(first.get('tool_call_id'), str)):
            failures.append(f'{name}: round-1 tool call not exact')
        if not isinstance(result['round2'].get('content'), str) or not result['round2']['content'].strip():
            failures.append(f'{name}: round-2 answer empty')
        elif '17' not in result['round2']['content']:
            failures.append(f'{name}: round-2 answer does not reflect the tool result')
    return {'gate': 'PASS' if not failures else 'FAIL', 'failures': failures,
            'per_checkpoint': {k: dict(v, verdict='PASS') if 'round1' in v and 'round2' in v
                               and v['round1'].get('tool_name') == 'get_weather'
                               and v['round1'].get('arguments') == {'city': 'Ghent', 'unit': 'celsius'}
                               and isinstance(v['round2'].get('content'), str)
                               and v['round2']['content'].strip()
                               and '17' in v['round2']['content']
                               else dict(v, verdict='FAIL') for k, v in sorted(results.items())}}


_HAYSTACK_TOPICS = ('inventory rotation', 'shelf labeling', 'loading dock scheduling', 'climate control',
                    'delivery routes', 'pallet repair', 'safety inspections', 'visitor badging',
                    'furniture placement', 'cleaning rosters', 'signage updates', 'door maintenance')
_HAYSTACK_ACTIONS = ('is reviewed quarterly', 'was audited last month', 'follows the 2025 checklist',
                     'requires two signatures', 'is handled by the facilities group', 'was paused in spring',
                     'resumed in summer', 'is documented separately', 'has its own binder',
                     'was discussed in the morning meeting')


def build_haystack(paragraph_count):
    """Deterministic, code-free filler paragraphs (no digits, no Aurora)."""
    if type(paragraph_count) is not int or paragraph_count <= 0:
        raise ValueError('paragraph_count must be a positive int')
    paragraphs = []
    for index in range(paragraph_count):
        topic = _HAYSTACK_TOPICS[index % len(_HAYSTACK_TOPICS)]
        action = _HAYSTACK_ACTIONS[(index // len(_HAYSTACK_TOPICS)) % len(_HAYSTACK_ACTIONS)]
        paragraphs.append(f'Facility note {index + 1}: the {topic} {action}.')
    return paragraphs


def insert_needle(paragraphs, position):
    """Return (paragraphs_with_needle, index); exactly one needle paragraph."""
    if position not in NEEDLE_POSITIONS:
        raise ValueError('unknown needle position: ' + repr(position))
    index = round(len(paragraphs) * NEEDLE_POSITIONS[position])
    index = min(max(index, 0), len(paragraphs))
    with_needle = list(paragraphs)
    with_needle.insert(index, NEEDLE_PARAGRAPH)
    return with_needle, index


def size_haystack(post, target_tokens=NEEDLE_TARGET_TOKENS):
    """Grow the haystack until tokenize() reports the target token count.

    `post(endpoint, body)` must return {'tokens': [...]} for /tokenize with
    {'content': text, 'add_special': False}. Bounded to 6 calls."""
    if type(target_tokens) is not int or target_tokens <= 0:
        raise ValueError('target_tokens must be a positive int')
    count = max(1, target_tokens // 40)
    for _ in range(6):
        paragraphs = build_haystack(count)
        text = '\n\n'.join(paragraphs)
        result = post('tokenize', {'content': text, 'add_special': False})
        tokens = result.get('tokens') if isinstance(result, dict) else None
        if not isinstance(tokens, list):
            raise RuntimeError('tokenize returned no token list')
        if abs(len(tokens) - target_tokens) <= target_tokens * NEEDLE_TOKEN_TOLERANCE:
            return paragraphs
        count = max(1, round(count * target_tokens / max(len(tokens), 1)))
    raise RuntimeError('haystack sizing did not converge')


def needle_payload(paragraphs):
    content = '\n\n'.join(paragraphs) + '\n\n' + NEEDLE_QUESTION
    return {'messages': [{'role': 'user', 'content': content}],
            'max_tokens': NEEDLE_TOKENS, 'temperature': 0.0, 'seed': 42,
            'stream': False, 'cache_prompt': True}


def validate_needle(data):
    """Clean stop, non-empty text, and the exact code present in the answer."""
    message = _validated_choice(data, 'stop')
    content = message.get('content')
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError('needle answer is empty')
    if message.get('tool_calls') is not None:
        raise RuntimeError('needle answer contains unexpected tool calls')
    return {'recalled': NEEDLE_CODE in content, 'content': content}


def needle_verdict(results):
    """PASS requires recall at start, middle and end."""
    failures = []
    for position, fraction in sorted(NEEDLE_POSITIONS.items()):
        entry = results.get(position)
        if not isinstance(entry, dict) or not entry.get('recalled'):
            failures.append(f'{position}: code not recalled')
    return {'gate': 'PASS' if not failures else 'FAIL', 'failures': failures,
            'positions': dict(results)}


# ---------------------------------------------------------------------------
# Hermes E2E contract gate (mandate 2026-09-09 items 5-6): OpenAI-API contract
# tests against the live local endpoint: model discovery, non-streaming and
# streaming chat completions, tool definitions/JSON schema, tool-call id
# linkage, tool result feed-back, finish reasons, multiple tool rounds.
# Target-only (DRAFT=0) like the other quality gates.
# ---------------------------------------------------------------------------
CONTRACT_MODEL_HINT = 'expertpin'


def contract_check_models(models_payload):
    """GET /v1/models returns an OpenAI-style list with >=1 id."""
    if not isinstance(models_payload, dict):
        raise RuntimeError('models payload is not an object')
    data = models_payload.get('data')
    if not isinstance(data, list) or not data:
        raise RuntimeError('models list is empty')
    for entry in data:
        if not isinstance(entry, dict) or not isinstance(entry.get('id'), str) or not entry['id']:
            raise RuntimeError('model entry missing string id')
    return {'model_ids': [entry['id'] for entry in data]}


def contract_check_nonstreaming(completion, expect_tool_calls):
    """POST /v1/chat/completions (stream=false) shape and finish reason."""
    if not isinstance(completion, dict) or 'error' in completion:
        raise RuntimeError('invalid/error completion')
    if completion.get('object') != 'chat.completion':
        raise RuntimeError('object is not chat.completion')
    choices = completion.get('choices')
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError('expected exactly one choice')
    choice = choices[0]
    if choice.get('index') != 0:
        raise RuntimeError('choice index is not 0')
    message = choice.get('message')
    if not isinstance(message, dict) or message.get('role') != 'assistant':
        raise RuntimeError('message is not assistant')
    finish = choice.get('finish_reason')
    calls = message.get('tool_calls')
    if expect_tool_calls:
        if finish != 'tool_calls':
            raise RuntimeError(f'expected finish_reason tool_calls, got {finish!r}')
        if not isinstance(calls, list) or not calls:
            raise RuntimeError('expected tool calls in message')
    else:
        if finish != 'stop':
            raise RuntimeError(f'expected finish_reason stop, got {finish!r}')
        if calls is not None:
            raise RuntimeError('unexpected tool calls in plain answer')
        if not isinstance(message.get('content'), str) or not message['content'].strip():
            raise RuntimeError('plain answer content missing/empty')
    usage = completion.get('usage')
    if not isinstance(usage, dict) or type(usage.get('completion_tokens')) is not int:
        raise RuntimeError('usage.completion_tokens missing/not int')
    return {'finish_reason': finish}


def contract_check_streaming(events, expect_tool_calls):
    """POST stream=true: SSE chunks assemble a coherent completion.

    events: parsed SSE payload objects (dicts) in order, WITHOUT the [DONE]
    sentinel. Validates object/choices shape, delta roles, content or
    tool_call deltas with stable index, and a terminal finish_reason."""
    if not events:
        raise RuntimeError('no stream events')
    roles = set()
    finishes = []
    content_parts = []
    tool_delta_indexes = []
    for event in events:
        if not isinstance(event, dict) or event.get('object') != 'chat.completion.chunk':
            raise RuntimeError('stream event is not a chat.completion.chunk')
        choices = event.get('choices')
        if not isinstance(choices, list) or len(choices) != 1:
            raise RuntimeError('stream chunk without exactly one choice')
        delta = choices[0].get('delta')
        if not isinstance(delta, dict):
            raise RuntimeError('stream chunk without delta object')
        role = delta.get('role')
        if role is not None:
            roles.add(role)
        if isinstance(delta.get('content'), str):
            content_parts.append(delta['content'])
        calls = delta.get('tool_calls')
        if calls is not None:
            if not isinstance(calls, list):
                raise RuntimeError('delta.tool_calls is not a list')
            for call in calls:
                if isinstance(call, dict) and isinstance(call.get('index'), int):
                    tool_delta_indexes.append(call['index'])
                elif isinstance(call, dict) and 'function' in call and 'index' not in call:
                    raise RuntimeError('streamed tool call delta missing index')
        finish = choices[0].get('finish_reason')
        if finish is not None:
            finishes.append(finish)
    if roles - {'assistant'}:
        raise RuntimeError(f'unexpected delta roles: {sorted(roles)}')
    if not finishes or finishes[-1] not in ('stop', 'tool_calls', 'length'):
        raise RuntimeError('stream ended without a terminal finish_reason')
    if len([f for f in finishes if f is not None]) > 1:
        raise RuntimeError('multiple finish_reasons in stream')
    if expect_tool_calls:
        if finishes[-1] != 'tool_calls' or not tool_delta_indexes:
            raise RuntimeError('expected streamed tool calls with finish tool_calls')
        return {'finish_reason': finishes[-1], 'streamed_tool_calls': len(tool_delta_indexes)}
    if finishes[-1] != 'stop' or not ''.join(content_parts).strip():
        raise RuntimeError('expected a plain streamed answer with finish stop')
    return {'finish_reason': finishes[-1]}


def contract_check_tool_round(trip):
    """One full tool round: forced call -> result feed-back -> plain answer."""
    if not isinstance(trip, dict):
        raise RuntimeError('tool round record invalid')
    first = contract_check_nonstreaming(trip['round1'], expect_tool_calls=True)
    message = trip['round1']['choices'][0]['message']
    call = message['tool_calls'][0]
    call_id = call.get('id')
    if not isinstance(call_id, str) or not call_id.strip():
        raise RuntimeError('tool call id missing')
    function = call.get('function')
    if not isinstance(function, dict) or function.get('name') != 'get_weather':
        raise RuntimeError('forced tool is not get_weather')
    import json as _json
    arguments = _json.loads(function.get('arguments') or '{}')
    if arguments.get('city') != 'Ghent':
        raise RuntimeError('streamed tool arguments city is not Ghent')
    feedback = trip['round2_request']['messages']
    roles = [m.get('role') for m in feedback]
    if roles != ['user', 'assistant', 'tool']:
        raise RuntimeError('round-2 message roles wrong: ' + repr(roles))
    tool_message = feedback[2]
    if tool_message.get('tool_call_id') != call_id:
        raise RuntimeError('round-2 tool message does not reference the emitted call id')
    if tool_message.get('role') != 'tool':
        raise RuntimeError('round-2 tool message role is not tool')
    second = contract_check_nonstreaming(trip['round2'], expect_tool_calls=False)
    return {'call_id': call_id, 'round1_finish': first['finish_reason'],
            'round2_finish': second['finish_reason']}


# Error-path contract (mandate item 5: predictable error handling). The server
# wraps every failure as {"error": {code, message, ...}} with the HTTP status
# taken from error.code (examples/server/server.cpp:442-446 res_err).
ERROR_OVERFLOW_WORDS = 12000  # > 8192 tokens after tokenization at ~1.5-2 tok/word


def overflow_payload():
    """A prompt that exceeds the default CTX=8192 window plus output."""
    filler = 'overflow ' * ERROR_OVERFLOW_WORDS
    return {'messages': [{'role': 'user', 'content': filler.strip()}],
            'max_tokens': 64, 'temperature': 0.0, 'seed': 42,
            'stream': False, 'cache_prompt': False}


def bad_tool_choice_payload():
    """An invalid tool_choice value the server must reject predictably:
    an unknown string throws std::invalid_argument (common/chat.cpp:263)
    and surfaces as a 4xx error envelope. (The OpenAI OBJECT form is NOT
    an error in this build — it silently degrades to 'auto'; documented in
    the tool-call gate review.)"""
    return {'messages': [{'role': 'user', 'content': 'Check the weather in Ghent.'}],
            'max_tokens': 64, 'temperature': 0.0, 'seed': 42, 'stream': False,
            'cache_prompt': False,
            'tools': [{'type': 'function', 'function': {
                'name': 'get_weather', 'description': 'Get weather.',
                'parameters': TOOL_SCHEMA}}],
            'tool_choice': 'bogus-choice'}


def missing_messages_payload():
    """Chat completion without the required messages array."""
    return {'max_tokens': 32, 'temperature': 0.0, 'seed': 42, 'stream': False}


def contract_check_http_error(status, body):
    """A failed request must return a JSON error envelope, not a 2xx body.

    Status must be a client/server error (4xx or 5xx). Note: this server
    classifies context-overflow as ERROR_TYPE_SERVER -> HTTP 500
    (examples/server/server-context.cpp:3983), while invalid tool_choice
    and missing messages surface as 4xx — predictable envelope either way."""
    if not isinstance(body, dict) or not isinstance(body.get('error'), dict):
        raise RuntimeError('error response is not an {"error": {...}} envelope')
    if not 400 <= status < 600:
        raise RuntimeError(f'expected a 4xx/5xx error status, got {status}')
    error = body['error']
    message = error.get('message')
    if not isinstance(message, str) or not message.strip():
        raise RuntimeError('error envelope has no message')
    return {'status': status, 'error_type': error.get('type'),
            'message_head': message[:80]}


def errors_verdict(results):
    """PASS requires all three error paths to fail predictably (error envelope)."""
    failures = []
    for name in ('overflow', 'bad-tool-choice', 'missing-messages'):
        entry = results.get(name)
        if not isinstance(entry, dict) or not 400 <= entry.get('status', 0) < 600:
            failures.append(f'{name}: no predictable error envelope')
    return {'gate': 'PASS' if not failures else 'FAIL', 'failures': failures,
            'paths': dict(results)}


def validate_completion(data, nmax):
    """Fail closed on EOS, malformed metrics, absent MTP, or inconsistent totals."""
    if not isinstance(data, dict) or 'error' in data:
        raise RuntimeError('invalid/error completion')
    usage, choices, timings = data.get('usage'), data.get('choices'), data.get('timings')
    if (not isinstance(usage, dict) or type(usage.get('completion_tokens')) is not int
            or usage['completion_tokens'] != TOKENS or not isinstance(choices, list)
            or len(choices) != 1 or not isinstance(choices[0], dict)
            or choices[0].get('finish_reason') != 'length'):
        raise RuntimeError('request did not produce the intended 256-token length-limited decode')
    message = choices[0].get('message')
    if not isinstance(message, dict) or not any(
            isinstance(message.get(k), str) and message[k].strip()
            for k in ('content', 'reasoning_content')):
        raise RuntimeError('request produced no text')
    if not isinstance(timings, dict) or type(timings.get('predicted_n')) is not int or timings['predicted_n'] != TOKENS:
        raise RuntimeError('missing/mismatched predicted count')
    for key in ('prompt_ms', 'predicted_ms', 'predicted_per_second'):
        value = timings.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise RuntimeError('invalid timing: ' + key)
    rows = timings.get('draft_by_depth')
    if nmax == 0:
        if (timings.get('draft_n', 0) != 0 or timings.get('draft_n_accepted', 0) != 0
                or rows not in (None, [])):
            raise RuntimeError('unexpected drafting in target-only mode')
        return {'usage': usage, 'timings': timings, 'acceptance_fraction': None}
    if not isinstance(rows, list) or not rows:
        raise RuntimeError('missing MTP acceptance depth counters')
    drafted = accepted = 0
    for depth, row in enumerate(rows, 1):
        if (not isinstance(row, dict) or type(row.get('depth')) is not int
                or row['depth'] != depth or depth > nmax
                or type(row.get('draft_n')) is not int or row['draft_n'] <= 0
                or type(row.get('draft_n_accepted')) is not int
                or not 0 <= row['draft_n_accepted'] <= row['draft_n']):
            raise RuntimeError('invalid MTP depth counters')
        drafted += row['draft_n']
        accepted += row['draft_n_accepted']
    if (type(timings.get('draft_n')) is not int or timings['draft_n'] != drafted
            or type(timings.get('draft_n_accepted')) is not int
            or timings['draft_n_accepted'] != accepted or accepted > TOKENS):
        raise RuntimeError('inconsistent MTP acceptance totals')
    return {'usage': usage, 'timings': timings, 'acceptance_fraction': accepted / drafted}


def validate_budget(point):
    if not 0 <= point['vram_used_mib'] < 28 * 1024:
        raise RuntimeError('VRAM cap exceeded/invalid')
    if not 0 <= point['gpu_util_pct'] <= 100:
        raise RuntimeError('invalid GPU utilization')
    if int(point['memory.max']) != RAM_BUDGET_GIB * 1024**3:
        raise RuntimeError('unexpected cgroup MemoryMax')
    if not 0 <= int(point['memory.current']) <= int(point['memory.peak']) <= 40 * 1024**3:
        raise RuntimeError('RAM cap exceeded/invalid')
    events = dict(line.split() for line in point['memory.events'].splitlines())
    if any(int(events.get(k, 0)) for k in ('max', 'oom', 'oom_kill', 'oom_group_kill')):
        raise RuntimeError('cgroup memory pressure/OOM event')


def terminate(child):
    if child is not None:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        else:
            child.wait(timeout=5)


class Run:
    """Small ownership container: no subprocesses or writes until execute()."""
    def __init__(self, out, startup_nmax, checkpoint=DEFAULT_CHECKPOINT, gate='throughput'):
        validate_plan(startup_nmax, checkpoint, gate)
        self.out = out
        self.startup_nmax = startup_nmax
        self.checkpoint = checkpoint
        self.gate = gate
        if gate == 'toolcall':
            self.sequence = ('round1', 'round2')
        elif gate == 'needle':
            # end -> middle -> start: every request shares the previous prefix,
            # so the server prompt cache (cache_prompt=True) can reuse it.
            self.sequence = ('end', 'middle', 'start')
        elif gate == 'contract':
            # mandate items 5-6: discovery, non-streaming, streaming plain,
            # streaming tool call, then a full tool round-trip
            self.sequence = ('models', 'plain', 'stream-plain', 'stream-tool', 'tool-round')
        elif gate == 'errors':
            # mandate item 5: predictable error handling on the happy-path server
            self.sequence = ('overflow', 'bad-tool-choice', 'missing-messages')
        else:
            self.sequence = (startup_nmax, startup_nmax)
        self.request_seconds = NEEDLE_REQUEST_SECONDS if gate in ('needle', 'contract', 'errors') else REQUEST_SECONDS
        self.work_seconds = NEEDLE_WORK_SECONDS if gate in ('needle', 'contract', 'errors') else WORK_SECONDS
        self.scope = 'expertpin-test-' + uuid.uuid4().hex + '.scope'
        self.proc = self.request = self.server_log = None
        self.cgroup_path = None
        self.scope_launched = self.prepared = self.launching = False
        self.pending_signal = None
        self.tokenize_references = []
        self.verifier_trace = False
        self.start = time.monotonic()
        self.label = 'preflight'
        self.round1_validated = None
        self.haystack_paragraphs = None
        self.summary = {'started': datetime.now().astimezone().isoformat(), 'scope': self.scope,
                        'model_loaded': False, 'startup_n_max': startup_nmax,
                        'gate': gate,
                        'checkpoint': checkpoint,
                        'sequence': list(self.sequence), 'requests': [], 'status': 'blocked_or_failed'}

    def save(self, name, data):
        (self.out / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

    def command(self, args, name, check=True, timeout=10):
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        (self.out / (name + '.log')).write_text(result.stdout + result.stderr)
        if check and result.returncode:
            raise RuntimeError(f'{name}: rc={result.returncode}: {result.stderr}')
        return result.stdout.strip()

    def interrupt(self, signum, frame):
        if self.launching:
            self.pending_signal = signum
        else:
            raise TimeoutError(f'signal {signum}')

    def spawn(self, args, **kwargs):
        # Defer handled signals until the child handle is registered by caller.
        self.launching = True
        try:
            return subprocess.Popen(args, **kwargs)
        except BaseException:
            self.launching = False
            raise

    def spawned(self):
        self.launching = False
        if self.pending_signal is not None:
            self.interrupt(self.pending_signal, None)

    def sample(self):
        if time.monotonic() - self.start >= self.work_seconds:
            raise TimeoutError('work deadline; cleanup reserve begins')
        raw = self.command(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                            '--format=csv,noheader,nounits'], 'gpu-latest')
        if len(raw.splitlines()) != 1:
            raise RuntimeError('expected exactly one GPU')
        util, used = map(int, raw.split(','))
        cg = self.command(['systemctl', '--user', 'show', self.scope, '-p', 'ControlGroup', '--value'], 'cgroup')
        if not cg or not cg.endswith('/' + self.scope):
            raise RuntimeError('missing or mismatched own scope')
        base = Path('/sys/fs/cgroup') / cg.lstrip('/')
        self.cgroup_path = base
        point: dict[str, Any] = {k: (base / k).read_text().strip() for k in
                 ('memory.current', 'memory.peak', 'memory.max', 'memory.swap.current', 'memory.events')}
        point.update(elapsed_s=time.monotonic() - self.start, request_label=self.label,
                     vram_used_mib=used, gpu_util_pct=util,
                     mem_available_bytes=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
                                                  if s.startswith('MemAvailable:'))) * 1024)
        with (self.out / 'samples.jsonl').open('a') as output:
            output.write(json.dumps(point) + '\n')
        validate_budget(point)
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError('server process missing or exited')

    def execute(self):
        self.command(['git', 'rev-parse', 'HEAD'], 'commit')
        check = subprocess.run(['pgrep', '-x', 'llama-server'], capture_output=True, timeout=10)
        if check.returncode != 1:
            raise RuntimeError('pre-existing llama-server or failed process observation')
        units = self.command(['systemctl', '--user', 'list-units', '--type=scope', '--state=running',
                              '--no-legend', '--plain', 'expertpin-test-*.scope'], 'preexisting-scopes')
        if units:
            raise RuntimeError('pre-existing expertpin scope')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 8102))  # Refuse an occupied endpoint before host changes.
        env = clean_environment(os.environ, self.scope, self.startup_nmax, self.checkpoint)
        if self.verifier_trace:
            env['EXPERTPIN_VERIFIER_TRACE'] = '1'
        self.summary['verifier_trace'] = self.verifier_trace
        self.save('environment.json', {k: v for k, v in env.items()
                                      if k not in os.environ or os.environ[k] != v})
        if not Path(env['MODEL']).is_file():
            raise RuntimeError('checkpoint model file missing: ' + env['MODEL'])
        if self.startup_nmax and not Path(env['DRAFT_MODEL']).is_file():
            raise RuntimeError('required draft model missing')
        self.prepared = True  # Restore even a partially failed preparation.
        self.command(['bash', PREP], 'host-prep', timeout=45)
        self.summary['qli_stop_requested'] = datetime.now().astimezone().isoformat()
        self.command(['systemctl', '--user', 'stop', 'qli.service'], 'qli-stop')
        self.command(['nvidia-smi'], 'nvidia-before')
        self.command(['free', '-g'], 'free-before')
        spec = importlib.util.spec_from_file_location('clean_mtp_idle_gate', ROOT / 'evidence/trace-matched-bandwidth/run-physical.py')
        assert spec is not None and spec.loader is not None
        gate: Any = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gate)
        gate.OUT = self.out
        idle = gate.wait_gpu_idle('model-readiness')
        self.summary['guard'] = idle
        if not (0 <= idle['gpu_util_pct'] < 5 and idle['mem_available_bytes'] >= 34 * 1024**3):
            raise RuntimeError('Tier A host guard blocked')
        launch = ['bash', str(ROOT / 'scripts/run-qwen38-flash-next.sh')]
        dry = subprocess.run(launch, env=dict(env, DRY='1'), capture_output=True, text=True, timeout=30)
        (self.out / 'dry.log').write_text(dry.stdout + dry.stderr)
        if dry.returncode or 'WOULD BLOCK' in dry.stdout + dry.stderr or 'guards      : OK' not in dry.stdout:
            raise RuntimeError('launcher dry guard blocked/unverified')
        self.server_log = (self.out / 'server.log').open('x')
        self.scope_launched = True
        self.proc = self.spawn(launch, env=env, stdout=self.server_log, stderr=subprocess.STDOUT)
        self.spawned()
        for _ in range(20):
            if self.command(['systemctl', '--user', 'is-active', self.scope], 'scope-ready', check=False) == 'active':
                break
            if self.proc.poll() is not None:
                raise RuntimeError('launcher exited before scope readiness')
            time.sleep(0.25)
        else:
            raise RuntimeError('own scope never became active')
        self.label = 'model-load'
        deadline = min(self.start + self.work_seconds, time.monotonic() + 360)
        while time.monotonic() < deadline:
            self.sample()
            try:
                with urllib.request.urlopen('http://127.0.0.1:8102/health', timeout=1) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(1)
        else:
            raise TimeoutError('health readiness timeout')
        self.summary['model_loaded'] = True
        for index, step in enumerate(self.sequence, 1):
            if self.gate == 'toolcall':
                label = f'{index:02d}-{step}'
                if step == 'round1':
                    body = toolcall_payload()
                else:
                    if self.round1_validated is None:
                        raise RuntimeError('round2 requires a validated round1')
                    body = toolcall_round2_payload(
                        toolcall_payload(), self.round1_validated['tool_call_id'],
                        json.dumps(self.round1_validated['arguments']), TOOL_RESULT)
            elif self.gate == 'needle':
                label = f'{index:02d}-{step}'
                if self.haystack_paragraphs is None:
                    self.haystack_paragraphs = size_haystack(self.tokenize_post)
                paragraphs, _ = insert_needle(self.haystack_paragraphs, step)
                body = needle_payload(paragraphs)
            elif self.gate in ('contract', 'errors'):
                label = f'{index:02d}-{step}'
                remaining = min(self.request_seconds, self.start + self.work_seconds - time.monotonic())
                if remaining <= 1:
                    raise TimeoutError('no request budget remains')
                began = time.monotonic()
                result = dict(self.contract_step(step, remaining), label=label, step=step,
                              wall_seconds=None)
                result['wall_seconds'] = round(time.monotonic() - began, 3)
                self.save(label + '-metrics.json', result)
                self.summary['requests'].append(result)
                self.save('summary.json', self.summary)
                continue
            else:
                label = f'{index:02d}-nmax{step}'
                body = payload(step)
            self.label = label
            remaining = min(self.request_seconds, self.start + self.work_seconds - time.monotonic())
            if remaining <= 1:
                raise TimeoutError('no request budget remains')
            request_path = self.out / (label + '-request.json')
            response_path = self.out / (label + '-response.json')
            self.save(request_path.name, body)
            began = time.monotonic()
            self.sample()
            with response_path.open('x') as response, (self.out / (label + '-curl.log')).open('x') as error:
                self.request = self.spawn(['curl', '--silent', '--show-error', '--fail-with-body',
                                           '--connect-timeout', '5', '--max-time', str(remaining),
                                           '-H', 'Content-Type: application/json', '--data-binary', '@' + str(request_path),
                                           'http://127.0.0.1:8102/v1/chat/completions'], stdout=response, stderr=error)
                self.spawned()
                while self.request.poll() is None:
                    self.sample()
                    if time.monotonic() - began >= remaining:
                        raise TimeoutError('request deadline')
                    time.sleep(1)
                if self.request.wait(timeout=5):
                    raise RuntimeError('completion HTTP/curl failure: ' + self.label)
            self.sample()
            if self.gate == 'toolcall':
                data = json.loads(response_path.read_text())
                if step == 'round1':
                    self.round1_validated = validate_toolcall(data)
                    result = dict(self.round1_validated, label=label, step=step,
                                  wall_seconds=time.monotonic() - began,
                                  prompt_tokens=data.get('usage', {}).get('prompt_tokens'),
                                  completion_tokens=data.get('usage', {}).get('completion_tokens'))
                else:
                    result = dict(validate_round2(data), label=label, step=step,
                                  wall_seconds=time.monotonic() - began,
                                  prompt_tokens=data.get('usage', {}).get('prompt_tokens'),
                                  completion_tokens=data.get('usage', {}).get('completion_tokens'))
            elif self.gate == 'needle':
                data = json.loads(response_path.read_text())
                result = dict(validate_needle(data), label=label, step=step,
                              wall_seconds=time.monotonic() - began,
                              prompt_tokens=data.get('usage', {}).get('prompt_tokens'),
                              completion_tokens=data.get('usage', {}).get('completion_tokens'))
            else:
                result = validate_completion(json.loads(response_path.read_text()), step)
                result.update(label=label, n_max=step, wall_seconds=time.monotonic() - began)
            self.save(self.label + '-metrics.json', result)
            self.summary['requests'].append(result)
            self.save('summary.json', self.summary)
        if self.tokenize_references:
            self.capture_tokenizations()
        self.summary['status'] = 'completed'

    def capture_tokenizations(self):
        """Retokenized response fields, NOT observed decode or draft token IDs."""
        paths = [self.out / (r['label'] + '-response.json') for r in self.summary['requests']]
        paths += self.tokenize_references
        records = []
        def post(endpoint, body):
            self.sample()  # Enforce work deadline and budgets for every request.
            request = urllib.request.Request('http://127.0.0.1:8102/' + endpoint,
                data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=5) as response:
                return json.load(response)
        for path in paths:
            fields = output_fields(json.loads(path.read_text()))
            tokens = {}
            for key, text in fields.items():
                if text is None:
                    tokens[key] = None
                    continue
                result = post('tokenize', {'content': text, 'add_special': False})['tokens']
                if not isinstance(result, list) or any(type(t) is not int for t in result):
                    raise RuntimeError('invalid token IDs')
                if post('detokenize', {'tokens': result})['content'] != text:
                    raise RuntimeError('retokenization failed exact round-trip')
                tokens[key] = result
            records.append({'response_path': str(path.resolve()), 'fields': fields, 'tokens': tokens})
        # Do not detokenize individual IDs: byte-fallback tokens may be invalid
        # UTF-8 in isolation. Compare complete ID sequences offline, including
        # prefix/length and null differences; retain exact source fields here.
        self.save('retokenized.json', {'kind': 'retokenized_fields_not_decode_trace',
                                      'records': records})

    def tokenize_post(self, endpoint, body):
        """Bounded POST to /tokenize used by size_haystack; enforces budgets."""
        self.sample()
        request = urllib.request.Request('http://127.0.0.1:8102/' + endpoint,
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)

    def contract_step(self, step, remaining):
        """Run one contract sub-check against the live endpoint; returns a result dict."""
        label = step
        if step == 'models':
            with urllib.request.urlopen('http://127.0.0.1:8102/v1/models', timeout=10) as response:
                payload = json.load(response)
            self.save(label + '-response.json', payload)
            return contract_check_models(payload)
        body_plain = {'messages': [{'role': 'user', 'content':
                        'Reply with the single word: ready'}],
                      'max_tokens': 512, 'temperature': 0.0, 'seed': 42,
                      'stream': False, 'cache_prompt': False}
        if step == 'plain':
            result, _ = self.contract_post(label, body_plain, lambda data: contract_check_nonstreaming(data, False))
            return result
        if step == 'stream-plain':
            return self.contract_stream(label, dict(body_plain, stream=True), False)
        tool_body = dict(toolcall_payload(), stream=True)
        if step == 'stream-tool':
            return self.contract_stream(label, tool_body, True)
        if step == 'tool-round':
            round1_body = toolcall_payload()  # stream false
            _, round1 = self.contract_post('tool-round-r1', round1_body,
                                        lambda data: contract_check_nonstreaming(data, True))
            call = round1['choices'][0]['message']['tool_calls'][0]
            round2_body = toolcall_round2_payload(
                round1_body, call['id'], call['function']['arguments'], TOOL_RESULT)
            _, round2 = self.contract_post('tool-round-r2', round2_body,
                                        lambda data: contract_check_nonstreaming(data, False))
            trip = {'round1': round1, 'round2_request': round2_body, 'round2': round2}
            self.save('tool-round-trip.json', trip)
            return contract_check_tool_round(trip)
        if step in ('overflow', 'bad-tool-choice', 'missing-messages'):
            body = {'overflow': overflow_payload,
                    'bad-tool-choice': bad_tool_choice_payload,
                    'missing-messages': missing_messages_payload}[step]()
            self.save(step + '-request.json', body)
            self.sample()
            began = time.monotonic()
            request = urllib.request.Request('http://127.0.0.1:8102/v1/chat/completions',
                data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(request, timeout=self.request_seconds) as response:
                    status, payload = response.status, json.load(response)
            except urllib.error.HTTPError as http_error:
                status = http_error.code
                payload = json.loads(http_error.read().decode('utf-8'))
            self.save(step + '-response.json', {'http_status': status, 'body': payload})
            result = contract_check_http_error(status, payload)
            result['wall_seconds'] = round(time.monotonic() - began, 3)
            return result
        raise RuntimeError('unknown contract step: ' + repr(step))

    def contract_post(self, label, body, check):
        """POST a non-streaming completion; validate and return (result, raw)."""
        self.save(label + '-request.json', body)
        self.sample()
        began = time.monotonic()
        request = urllib.request.Request('http://127.0.0.1:8102/v1/chat/completions',
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=self.request_seconds) as response:
            data = json.load(response)
        self.save(label + '-response.json', data)
        result = dict(check(data), wall_seconds=round(time.monotonic() - began, 3))
        return result, data

    def contract_stream(self, label, body, expect_tool_calls):
        self.save(label + '-request.json', body)
        self.sample()
        began = time.monotonic()
        raw_frames = []
        events = []
        request = urllib.request.Request('http://127.0.0.1:8102/v1/chat/completions',
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=self.request_seconds) as response:
            for line in response:
                text = line.decode('utf-8').strip()
                if not text.startswith('data: '):
                    continue
                payload = text[len('data: '):]
                raw_frames.append(payload)
                if payload == '[DONE]':
                    break
                events.append(json.loads(payload))
                if time.monotonic() - began > self.request_seconds:
                    raise TimeoutError('stream exceeded request budget')
        self.save(label + '-sse-frames.json', raw_frames)
        result = contract_check_streaming(events, expect_tool_calls)
        result['wall_seconds'] = time.monotonic() - began
        return result

    def own_scope_empty(self):
        cg = self.command(['systemctl', '--user', 'show', self.scope, '-p', 'ControlGroup', '--value'],
                          'cleanup-cgroup', check=False)
        if cg:
            if not cg.endswith('/' + self.scope):
                raise RuntimeError('unexpected cleanup cgroup')
            self.cgroup_path = Path('/sys/fs/cgroup') / cg.lstrip('/')
        base = self.cgroup_path
        if base is None:
            raise RuntimeError('owned cgroup never observed; cannot prove emptiness')
        return not base.exists() or 'populated 0' in (base / 'cgroup.events').read_text()

    def cleanup(self):
        """Each action bounded; failures never suppress mandatory restoration."""
        errors = []
        def attempt(name, fn):
            try:
                return fn()
            except BaseException as error:
                errors.append(f'{name}: {error}')
                return None
        attempt('request-stop', lambda: terminate(self.request))
        if self.scope_launched:
            attempt('scope-stop', lambda: self.command(['systemctl', '--user', 'stop', self.scope], 'scope-stop'))
            state = attempt('scope-state', lambda: self.command(['systemctl', '--user', 'is-active', self.scope], 'scope-state', check=False))
            empty = attempt('scope-empty', self.own_scope_empty)
            if state not in ('inactive', 'failed', 'unknown') or empty is not True:
                attempt('scope-kill', lambda: self.command(['systemctl', '--user', 'kill', '--signal=KILL', '--kill-whom=all', self.scope], 'scope-kill'))
        attempt('launcher-stop', lambda: terminate(self.proc))
        if self.scope_launched:
            def verify():
                for _ in range(3):
                    state = self.command(['systemctl', '--user', 'is-active', self.scope], 'scope-final-state', check=False)
                    if state in ('inactive', 'failed', 'unknown') and self.own_scope_empty():
                        self.summary['scope_stopped_verified'] = True
                        return
                    self.command(['systemctl', '--user', 'kill', '--signal=KILL', '--kill-whom=all', self.scope], 'scope-final-kill', check=False)
                    time.sleep(0.25)
                raise RuntimeError('own scope termination unverified')
            attempt('scope-final-verification', verify)
        if self.server_log:
            attempt('server-log-close', self.server_log.close)
        # Standing protocol: start qli even after a lifecycle preflight refusal.
        self.summary['qli_start_requested'] = datetime.now().astimezone().isoformat()
        attempt('qli-start', lambda: self.command(['systemctl', '--user', 'start', 'qli.service'], 'qli-start'))
        self.summary['qli_status'] = attempt('qli-active', lambda: self.command(['systemctl', '--user', 'is-active', 'qli.service'], 'qli-active'))
        if self.summary['qli_status'] != 'active':
            errors.append('qli not verified active')
        if self.prepared:
            attempt('host-restore', lambda: self.command(['bash', PREP, '--restore'], 'host-restore', timeout=30))
            state = attempt('headless-active', lambda: self.command(['systemctl', '--user', 'is-active', 'headless-chromium.service'], 'headless-active'))
            if state != 'active':
                errors.append('headless host not verified active')
        self.summary['cleanup_errors'] = errors
        if errors:
            self.summary['status'] = 'cleanup_failed'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--startup-n-max', type=int, choices=(0, 4, 8, 16), default=4,
                        help='repeat twice at this startup/request depth; reload for a different depth')
    parser.add_argument('--gate', choices=GATES, default='throughput',
                        help='throughput: 2x256-token clean MTP benchmark; toolcall: forced get_weather '
                             'round-trip quality gate (target-only, no drafter); needle: code recall at '
                             'start/middle/end of a ~2K-token haystack (target-only); contract: OpenAI-API '
                             'contract tests for Hermes E2E (models, streaming, tool linkage)')
    parser.add_argument('--checkpoint', choices=sorted(CHECKPOINTS), default=DEFAULT_CHECKPOINT,
                        help='A/B checkpoint alias: reference 197 GiB Q4_K_M shards vs ps-iq2xxs 75.2 GiB IQ2_XXS')
    parser.add_argument('--tokenize-reference', type=Path, action='append', default=[],
                        help='retokenize saved response fields after generation; not a decode trace')
    # Explicit opt-in only: clean baselines must not inherit diagnostic tracing.
    parser.add_argument('--verifier-trace', action='store_true', help='diagnostic first-128 verifier decisions, NOT a clean throughput baseline')
    args = parser.parse_args(argv)
    try:
        validate_plan(args.startup_n_max, args.checkpoint, args.gate)
        for path in args.tokenize_reference:
            output_fields(json.loads(path.read_text()))
    except ValueError as error:
        print(error)
        return 2
    out = Path(__file__).resolve().parent / ('run-' + datetime.now().strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex)
    out.mkdir()
    run = Run(out, args.startup_n_max, args.checkpoint, args.gate)
    run.tokenize_references = args.tokenize_reference
    run.verifier_trace = args.verifier_trace
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    try:
        for sig in previous:
            signal.signal(sig, run.interrupt)
        signal.alarm(run.work_seconds)
        run.execute()
    except BaseException as error:
        run.summary['error'] = repr(error)
    finally:
        signal.alarm(0)
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            run.cleanup()
            run.summary['finished'] = datetime.now().astimezone().isoformat()
            run.summary['elapsed_seconds'] = time.monotonic() - run.start
            run.save('summary.json', run.summary)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    print(out)
    print(json.dumps(run.summary, indent=2))
    return 0 if run.summary['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
