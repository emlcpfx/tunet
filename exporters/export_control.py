"""On-demand export control channel for Spark training jobs.

A running Spark training job can't be poked directly, so the tunet-web UI drops
a tiny request file in a ShareSync control path; this module — called once per
epoch from train.py — polls it and, on a NEW request, exports the current model
inline (it's already on the GPU, so no separate compute) and self-uploads the
result to the job's ShareSync output dir.

Why self-upload: the Spark agent only syncs /output → ShareSync when the job
EXITS (see spark_start.sh), so a mid-run export wouldn't be downloadable until
the run ends. We upload it ourselves via WebDAV so it shows up immediately.
(If the output WebDAV base isn't known, the export still lands in /output and
syncs at exit — graceful fallback.)

Env (set by tunet-web's training-jobs submit; all optional — absent locally, so
this is a silent no-op for desktop training):
  TUNET_FILES_BASE   — ShareSync space base URL (WebDAV root)
  TUNET_BEARER       — Spark bearer token (read the request, write the export)
  TUNET_CONTROL_KEY  — per-job key = the output subdir name (e.g. "my_job")
"""

import os
import json
import time
import logging
import urllib.request
import urllib.parse

_last_nonce: int = 0
_output_dav: str | None = None       # cached job output WebDAV base (from meta.json)
_meta_tried: bool = False

# Cached Spark bearer for the control channel (see _bearer).
_tok: dict = {'access': '', 'exp': 0.0, 'refresh': ''}


def _bearer() -> str:
    """A currently-valid Spark bearer for control-channel reads + self-uploads.

    The token baked into the job env (TUNET_BEARER) expires in minutes, but a
    run polls/exports for hours. When refresh material is present
    (TUNET_REFRESH_TOKEN/URL) we mint a fresh access token via tunet-web's
    /api/spark/refresh proxy, cached until ~1 min before expiry, and track the
    rotated refresh token. Falls back to the static TUNET_BEARER when no refresh
    material is configured (legacy jobs). Never raises."""
    refresh_url = os.environ.get('TUNET_REFRESH_URL', '')
    if not _tok['refresh']:
        _tok['refresh'] = os.environ.get('TUNET_REFRESH_TOKEN', '')
    static = os.environ.get('TUNET_BEARER', '')
    if not (refresh_url and _tok['refresh']):
        return static
    now = time.time()
    if _tok['access'] and now < _tok['exp'] - 60:
        return _tok['access']
    try:
        body = json.dumps({'refreshToken': _tok['refresh']}).encode()
        r = urllib.request.Request(refresh_url, data=body, method='POST',
                                   headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(r, timeout=20) as resp:
            data = json.loads(resp.read())
        if data.get('accessToken'):
            _tok['access'] = data['accessToken']
            _tok['exp'] = now + int(data.get('expiresIn') or 300)
            if data.get('refreshToken'):
                _tok['refresh'] = data['refreshToken']
            return _tok['access']
    except Exception as e:
        logging.warning(f"[export-control] token refresh failed: {e}")
    return _tok['access'] or static


def _req(url: str, token: str, method: str = 'GET', data: bytes | None = None,
         timeout: int = 30) -> bytes:
    headers = {'Authorization': f'Bearer {token}'}
    if data is not None:
        headers['Content-Type'] = 'application/octet-stream'
    r = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return resp.read()


def _mkcol(url: str, token: str) -> None:
    try:
        _req(url, token, method='MKCOL', timeout=20)
    except Exception:
        pass  # already exists / racing — harmless


def poll_export_request() -> dict | None:
    """Check the control file once. Returns {'flame': bool, 'nuke': bool} for a
    NEW request (and marks it consumed via the nonce), else None.

    Cheap — a single small GET. Split from the export itself so the training loop
    can do this on rank 0 only, then coordinate the (collective) export across all
    ranks. Fully defensive: any failure returns None rather than raising.
    """
    global _last_nonce
    base  = os.environ.get('TUNET_FILES_BASE', '').rstrip('/')
    token = _bearer()
    key   = os.environ.get('TUNET_CONTROL_KEY', '')
    if not (base and token and key):
        return None  # no control channel configured (e.g. local training)

    ctrl = f"{base}/_tunet_control/{urllib.parse.quote(key)}"
    try:
        reqd  = json.loads(_req(f"{ctrl}/export_request.json", token, timeout=5))
        nonce = int(reqd.get('nonce', 0))
    except Exception:
        return None  # no request yet / unreachable / malformed

    if nonce <= _last_nonce:
        return None
    _last_nonce = nonce
    want_flame = bool(reqd.get('flame'))
    want_nuke  = bool(reqd.get('nuke'))
    if not (want_flame or want_nuke):
        return None
    return {'flame': want_flame, 'nuke': want_nuke}


def run_export(model, config, ckpt_prefix, export_res, completed_ep,
               want_flame: bool, want_nuke: bool) -> None:
    """Export the current model inline and self-upload the new files. The caller
    has already decided (via poll_export_request) that an export was requested.
    Defensive: failures are logged and swallowed so training is never disrupted.
    """
    logging.info(f"[export-control] on-demand export (flame={want_flame}, nuke={want_nuke}) @ epoch {completed_ep}")
    out_dir = config.data.output_dir
    before  = set(_list_exports(out_dir))
    try:
        from exporters.auto_export import export_flame, export_nuke
        if want_flame:
            export_flame(model, config, out_dir, completed_ep, export_res,
                         loss_mode=config.training.loss, ckpt_prefix=ckpt_prefix)
        if want_nuke:
            export_nuke(model, config, out_dir, completed_ep, export_res,
                        loss_mode=config.training.loss, ckpt_prefix=ckpt_prefix)
    except Exception as e:
        logging.error(f"[export-control] export failed: {e}", exc_info=True)
        return

    new_files = [p for p in _list_exports(out_dir) if p not in before]
    base  = os.environ.get('TUNET_FILES_BASE', '').rstrip('/')
    token = _bearer()
    key   = os.environ.get('TUNET_CONTROL_KEY', '')
    if base and token and key:
        _self_upload(new_files, out_dir, base, key, token)


def maybe_handle_export_request(model, config, ckpt_prefix, export_res, completed_ep) -> None:
    """Convenience for single-process / epoch-boundary use: poll once and export
    if a new request is present. Multi-GPU callers should use poll_export_request
    + run_export with their own broadcast/barrier coordination instead.
    """
    req = poll_export_request()
    if req:
        run_export(model, config, ckpt_prefix, export_res, completed_ep,
                   req['flame'], req['nuke'])


def _list_exports(out_dir: str) -> list[str]:
    root, found = os.path.join(out_dir, 'exports'), []
    for dp, _dirs, fns in os.walk(root):
        for fn in fns:
            found.append(os.path.join(dp, fn))
    return found


def _self_upload(files: list[str], out_dir: str, base: str, key: str, token: str) -> None:
    if not files:
        return
    dav = _resolve_output_dav(base, key, token)
    if not dav:
        logging.warning("[export-control] output ShareSync URL unknown — export saved to /output, "
                        "will sync to ShareSync when the job exits")
        return
    # Encode any raw spaces the base URL may carry (e.g. ".../Spark Fuse Jobs/...")
    # — urllib rejects URLs with control chars/spaces. Only spaces are at risk;
    # existing %xx escapes are untouched (so no double-encoding).
    dav = dav.rstrip('/').replace(' ', '%20')
    for f in files:
        rel   = os.path.relpath(f, out_dir).replace(os.sep, '/')  # e.g. exports/flame/foo.onnx
        parts = rel.split('/')
        cur   = dav
        for seg in parts[:-1]:                                    # MKCOL each parent collection
            cur = f"{cur}/{urllib.parse.quote(seg)}"
            _mkcol(cur, token)
        url = dav + '/' + '/'.join(urllib.parse.quote(p) for p in parts)
        try:
            with open(f, 'rb') as fh:
                _req(url, token, method='PUT', data=fh.read(), timeout=600)
            logging.info(f"[export-control] uploaded {rel}")
        except Exception as e:
            logging.warning(f"[export-control] upload failed for {rel}: {e}")


def _resolve_output_dav(base: str, key: str, token: str) -> str | None:
    """The job's output WebDAV base — tunet-web writes it to the control area
    (meta.json) right after submit, since the node can't know its own
    Spark-assigned output path otherwise."""
    global _output_dav, _meta_tried
    if _output_dav:
        return _output_dav
    if _meta_tried:
        return None
    _meta_tried = True
    try:
        meta = json.loads(_req(f"{base}/_tunet_control/{urllib.parse.quote(key)}/meta.json", token, timeout=5))
        _output_dav = meta.get('outputDav') or None
    except Exception:
        _output_dav = None
    return _output_dav
