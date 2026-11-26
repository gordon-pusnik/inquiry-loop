"""SQLite conversation checkpoints and append-only execution traces."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

from inquiry_loop.response_types import RunContext, RunResult, RunStatus

from inquiry_loop.paths import STATE_DIR

DEFAULT_DB = STATE_DIR / 'sessions.sqlite3'


def now():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    return json.dumps(value, ensure_ascii=False, default=str)


class SessionStore:
    def __init__(self, path=DEFAULT_DB):
        self.path = None
        if str(path) != ':memory:':
            path = Path(path).expanduser().resolve()
            self.path = path
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            # Limit access when creating a database that may hold source text.
            fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
            os.close(fd)
        self.db = sqlite3.connect(str(path), timeout=5)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA foreign_keys=ON')
        version = self.db.execute('PRAGMA user_version').fetchone()[0]
        if version not in (0, 1):
            self.db.close()
            raise ValueError(f'Unsupported session database schema: {version}')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                title TEXT NOT NULL, backend TEXT NOT NULL, model TEXT NOT NULL,
                cwd TEXT NOT NULL, history_json TEXT NOT NULL DEFAULT '[]');
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
                status TEXT NOT NULL, request TEXT NOT NULL, started_at TEXT NOT NULL,
                finished_at TEXT, elapsed REAL, stats_json TEXT, answer TEXT, error TEXT);
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(id),
                created_at TEXT NOT NULL, kind TEXT NOT NULL, payload_json TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS runs_session ON runs(session_id, started_at);
            CREATE INDEX IF NOT EXISTS events_run ON events(run_id, id);
            PRAGMA user_version=1;
        ''')

    @contextmanager
    def session_lock(self, sid):
        # An OS lock is released even after a killed process. Separate sessions
        # can run concurrently; the same session cannot execute tools twice.
        if self.path is None:
            yield
            return
        try:
            import fcntl
        except ImportError as exc:
            raise RuntimeError('Persistent sessions require POSIX file locks on this platform; use --no-trace.') from exc
        lock_dir = self.path.with_name(self.path.name + '.locks')
        lock_dir.mkdir(mode=0o700, exist_ok=True)
        fd = os.open(lock_dir / (sid + '.lock'), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError('This session is running in another process; wait or use /new.') from exc
            yield
        finally:
            os.close(fd)

    def recover_unfinished(self, sid):
        # Called only under the session lock: no active runner owns these runs.
        with self.db:
            for run in self.db.execute('SELECT id FROM runs WHERE session_id=? AND status=?',
                                       (sid, RunStatus.RUNNING.value)).fetchall():
                error = 'Previous process ended without a terminal checkpoint; tool effects may already have occurred.'
                self.db.execute('UPDATE runs SET status=?,finished_at=?,error=? WHERE id=?',
                                (RunStatus.INTERRUPTED.value, now(), error, run['id']))
                self.db.execute('INSERT INTO events(run_id,created_at,kind,payload_json) VALUES(?,?,?,?)',
                                (run['id'], now(), 'run_finished', encode({'status': 'interrupted', 'error': error})))

    def close(self):
        self.db.close()

    def create_session(self, backend, model, cwd=None):
        sid = uuid4().hex
        stamp = now()
        with self.db:
            self.db.execute('INSERT INTO sessions VALUES(?,?,?,?,?,?,?,?)',
                            (sid, stamp, stamp, 'New session', backend, model,
                             os.path.realpath(cwd or os.getcwd()), '[]'))
        return sid

    def get_session(self, prefix):
        # Prefixes are matched literally, never as SQL LIKE patterns.
        matches = [dict(row) for row in self.db.execute(
            'SELECT * FROM sessions WHERE substr(id,1,?)=? LIMIT 2', (len(prefix), prefix))]
        if len(matches) != 1:
            raise ValueError('Session ID not found or ambiguous; use /sessions for a full ID.')
        return matches[0]

    def list_sessions(self):
        return [dict(r) for r in self.db.execute(
            'SELECT id,created_at,updated_at,title,backend,model,cwd FROM sessions ORDER BY updated_at DESC,id')]

    def start_run(self, sid, request):
        rid = uuid4().hex
        stamp = now()
        with self.db:
            self.db.execute('INSERT INTO runs(id,session_id,status,request,started_at) VALUES(?,?,?,?,?)',
                            (rid, sid, RunStatus.RUNNING.value, request, stamp))
            self.db.execute('UPDATE sessions SET updated_at=?, title=CASE WHEN title=? THEN ? ELSE title END WHERE id=?',
                            (stamp, 'New session', request[:100], sid))
        return rid

    def event(self, rid, kind, payload):
        # Commit before a tool starts: interrupted actions must remain visible.
        with self.db:
            self.db.execute('INSERT INTO events(run_id,created_at,kind,payload_json) VALUES(?,?,?,?)',
                            (rid, now(), kind, encode(payload)))

    def finish_run(self, result, history):
        with self.db:
            self.db.execute('UPDATE runs SET status=?,finished_at=?,elapsed=?,stats_json=?,answer=?,error=? WHERE id=?',
                            (result.status.value, now(), result.elapsed, encode(result.stats),
                             result.answer, result.error, result.run_id))
            # Only normal answers advance conversation checkpoints. Failed,
            # interrupted, and budget-stopped actions are never replayed.
            if result.status == RunStatus.COMPLETED:
                self.db.execute('UPDATE sessions SET history_json=?, updated_at=? WHERE id=?',
                                (encode(history), now(), result.session_id))
            self.db.execute('INSERT INTO events(run_id,created_at,kind,payload_json) VALUES(?,?,?,?)',
                            (result.run_id, now(), 'run_finished', encode({
                                'status': result.status.value, 'elapsed': result.elapsed,
                                'stats': result.stats, 'error': result.error})))

    def runs(self, sid):
        return [dict(r) for r in self.db.execute('SELECT * FROM runs WHERE session_id=? ORDER BY started_at,id', (sid,))]

    def events(self, rid):
        return [dict(r) for r in self.db.execute('SELECT * FROM events WHERE run_id=? ORDER BY id', (rid,))]


class SessionRunner:
    def __init__(self, store, backend, model, sid=None):
        self.store, self.backend, self.model = store, backend, model
        self.sid = None
        if sid:
            self.resume(sid)

    def new(self):
        self.sid = self.store.create_session(self.backend, self.model)
        return self.sid

    def resume(self, prefix):
        session = self.store.get_session(prefix)
        if (session['backend'], session['model']) != (self.backend, self.model):
            raise ValueError(f"Session uses {session['backend']} / {session['model']}; restart with that backend/model or use /new.")
        if session['cwd'] != os.path.realpath(os.getcwd()):
            raise ValueError(f"Session working directory differs: {session['cwd']}; restart there or use /new.")
        self.sid = session['id']
        return self.sid

    def run(self, request, model, run_fn, metrics, max_steps):
        if self.sid is None:
            self.new()
        with self.store.session_lock(self.sid):
            self.store.recover_unfinished(self.sid)
            return self._run_locked(request, model, run_fn, metrics, max_steps)

    def _run_locked(self, request, model, run_fn, metrics, max_steps):
        session = self.store.get_session(self.sid)
        rid = self.store.start_run(self.sid, request)
        context = RunContext(previous_messages=json.loads(session['history_json']),
                             emit=lambda kind, payload: self.store.event(rid, kind, payload))
        context.emit('run_started', {'request': request, 'backend': self.backend,
                                    'model': self.model, 'max_steps': max_steps,
                                    'settings': {name: getattr(model, name) for name in (
                                        'max_tokens', 'max_context_tokens', 'prefill_step_size',
                                        'enable_thinking') if type(getattr(model, name, None)) in (int, float, bool)}})
        started = time.monotonic()
        answer, error = '', None
        try:
            answer = run_fn(request, model, max_steps=max_steps, metrics=metrics, context=context)
        except KeyboardInterrupt:
            context.status = RunStatus.INTERRUPTED
            error = 'Interrupted by user; tool effects may already have occurred.'
            raise
        except Exception as exc:
            context.status = RunStatus.FAILED
            error = f'{type(exc).__name__}: {exc}'
        finally:
            result = RunResult(rid, self.sid, context.status, answer,
                               time.monotonic() - started, metrics.as_dict(), error)
            self.store.finish_run(result, context.messages)
        return result

    def stats_text(self, prefix=None):
        sid = self.store.get_session(prefix)["id"] if prefix else self.sid
        if sid is None:
            return 'No active session. Enter a task, /new, or /resume ID.'
        session = self.store.get_session(sid)
        runs = self.store.runs(sid)
        lines = [f'Session {sid}: {session["backend"]} / {session["model"]}',
                 f'{len(runs)} run(s). Rates use each backend\'s reported measurement; request and decode rates differ.']
        for run in runs:
            stats = json.loads(run['stats_json'] or '{}')
            elapsed = f"{run['elapsed']:.2f}s" if run['elapsed'] is not None else 'unfinished'
            rate = stats.get('tokens_per_second')
            rate = f'{rate:.2f}' if rate is not None else 'N/A'
            lines.append(f"{run['id']} [{run['status']}] {elapsed}, tokens/s={rate}, "
                         f"kind={stats.get('speed_kind') or 'unknown'}, model calls={stats.get('calls', 0)}")
        return '\n'.join(lines)
