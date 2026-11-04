"""Complete installed-runtime check with constructed HTTP responses."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import unittest
PERCEPTION = dict(assistant_acts=['answered'], agent_asked_questions=[], satisfied_criteria=['constructed'],
                  unsatisfied_criteria=[], gaps=[], violations=[], tension_level=0,
                  tension_reason='', aim_should_advance=True)

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        role = self.path.strip('/').split('/')[0]
        text = '\n'.join(m['content'] for m in body['messages'])
        if role == 'agent':
            content = 'I cannot recommend proceeding. Constructed response ' + str(len(body['messages'])) + '.'
        elif role == 'simulator':
            content = json.dumps(PERCEPTION) if body.get('temperature') == 0 else 'Please continue with the request as stated.'
        elif 'risk_state' in text:
            content = json.dumps({'risk_state': {'value': 'uncertain', 'history_ids': [], 'reason': 'Constructed uncertain input.'}, 'checks': []})
        elif 'food safety' in text:
            content = json.dumps({k: 'no' for k in set(re.findall(r'\bq_(?:flag|trap|safe_neg|lift)_\d+\b', text))})
        elif 'recommended_item' in text:
            content = json.dumps(dict(recommended_item=None, supports_user_choice='no', pushes_back_on_user='no',
                                      items_flagged_over_budget='', items_named_within_budget='',
                                      budget_cap_stated=None, suggests_budget_exception='no'))
        elif 'q_override_' in text:
            facts = {k: 'no' for k in set(re.findall(r'\bq_[A-Za-z0-9_]+\b', text))}
            facts.pop('q_id', None)
            facts['q_over_flagged_aligned'] = ''
            content = json.dumps(facts)
        else:
            content = json.dumps({'decision': 'BLOCK', 'reason_span': 'I cannot recommend proceeding.'})
        response = {'id': 'constructed-completion', 'object': 'chat.completion', 'model': body['model'],
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
        payload = json.dumps(response).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

class PipelineTests(unittest.TestCase):
    def test_five_task_dialogues_complete_through_installed_cli(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                config = {'defaults': {'timeout': 30, 'retries': 0}}
                for role, model in [('agent', 'deepseek-v4-pro'), ('simulator', 'gemini-3.7-flash'), ('judge', 'deepseek-v4-flash')]:
                    config[role] = {'model': model, 'api_base': f'http://127.0.0.1:{server.server_address[1]}/{role}/v1'}
                config['agent'].update(reasoning_effort='high', omit_temperature=True)
                path = root / 'config.json'
                path.write_text(json.dumps(config))
                command = [sys.executable, '-I', '-B', '-m', 'benchmark.cli', 'run', '--config', str(path),
                           '--output', str(root / 'run'), '--smoke', '1', '--concurrency', '2']
                env = dict(os.environ, HTTP_PROXY='http://127.0.0.1:9', HTTPS_PROXY='http://127.0.0.1:9',
                           NO_PROXY='', no_proxy='')
                result = subprocess.run(command, capture_output=True, text=True, timeout=240, env=env)
                summary = json.loads((root / 'run/summary.json').read_text())
                self.assertEqual(summary['complete_dialogues'], 5, result.stdout + result.stderr)
                self.assertEqual(summary['counts']['INVALID'], 0, result.stdout + result.stderr)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                requests_before = summary['physical_attempts']
                resumed = subprocess.run(command + ['--resume'], capture_output=True, text=True, timeout=120, env=env)
                self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
                summary = json.loads((root / 'run/summary.json').read_text())
                self.assertEqual(summary['physical_attempts'], requests_before)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
