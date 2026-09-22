"""Regression tests without Ollama or the research dataset.

Run: python -m unittest discover -s tests -v
"""
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'models'), str(ROOT / 'frontend')]
import serve
import turn
from upload import parse_uploaded_csv, UploadError
from fastapi.testclient import TestClient


class UploadValidation(unittest.TestCase):
    def test_nonfinite_samples_and_invalid_time_are_rejected(self):
        for raw in (b'0,1\n1,inf\n2,3', b'0,1\n1,2\n1,3',
                    b'0,1\n2,2\n1,3\n3,4', b'0,1,2\n1,2,3'):
            with self.subTest(raw=raw), self.assertRaises(UploadError):
                parse_uploaded_csv(io.BytesIO(raw), None)

    def test_headerless_first_sample_preserved(self):
        t, x, fs = parse_uploaded_csv(io.BytesIO(b'0,10\n0.004,20\n0.008,30'), None)
        self.assertEqual(list(x), [10, 20, 30])
        self.assertEqual(fs, 250)

    def test_sample_rate_changes_invalidate_upload_cache(self):
        f = serve._UploadShim('signal.csv', b'1\n2\n3')
        session = turn.new_session()
        with patch.object(turn, 'parse_uploaded_csv', return_value=([], [], 250)), \
             patch.object(turn, 'load_uploaded_segment', return_value=object()) as load, \
             patch.object(turn, '_upload_question', return_value={}):
            for rate in (250, 500, 500):
                session['sample_rate'] = rate
                turn._upload_turn(session, '', f)
            self.assertEqual(load.call_count, 2)

    def test_older_upload_chart_never_uses_latest_recording(self):
        session = turn.new_session()
        session['uploads'] = {'first': {'name': 'first'}, 'second': {'name': 'second'}}
        session['last_upload'] = 'second'
        with patch.object(turn, '_upload_chart', side_effect=lambda c, theme: c['name']):
            self.assertEqual(turn.render_chart_ref(session, {'source': 'upload', 'upload_id': 'first'}), 'first')
            self.assertIsNone(turn.render_chart_ref(session, {'source': 'upload'}))
            self.assertIsNone(turn.render_chart_ref(session, {'source': 'upload', 'upload_id': 'expired'}))


class ChartAlignment(unittest.TestCase):
    def test_exact_requested_window_matches_classifier_features(self):
        import numpy as np
        from types import SimpleNamespace
        import render_window as renderer
        import classify_biceps as cb
        fs = 250
        t = np.arange(12 * fs) / fs
        x = np.random.default_rng(4).normal(size=t.size)
        segment = SimpleNamespace(data=x[:, None], t=t, gap_mask=np.zeros(t.size, dtype=bool))
        for requested in (4.0, 4.37, 11.9):
            start = min(int(np.searchsorted(t, requested)), x.size - 4 * fs)
            expected = cb.window_features(x[start:start + 4 * fs], fs)['mdf']
            figures = []
            def capture(fig, **kwargs):
                figures.append(fig)
                return '<div>test</div>'
            with self.subTest(requested=requested), \
                 patch.object(renderer, '_load_subject', return_value=(segment, fs, None, None)), \
                 patch.object(renderer.go.Figure, 'to_html', capture):
                renderer.render_window(13, requested, 'R')
                title = figures[0].layout.title.text
                self.assertIn(f'window start={t[start]:.2f}s', title)
                self.assertIn(f'window MDF={round(expected, 2):.1f} Hz', title)


class BrowserDelivery(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(serve.app)

    def test_shared_runtime_is_cacheable_and_not_repeated_in_chart(self):
        from render_window import _plotly_basic_js
        original = f'<script>{_plotly_basic_js()}</script><div>chart</div>'
        delivered = serve._browser_chart(original)
        self.assertLess(len(delivered), len(original) / 100)
        response = self.client.get('/assets/plotly-basic-3.7.0.min.js')
        self.assertEqual(response.status_code, 200)
        self.assertIn('immutable', response.headers['cache-control'])
        self.assertEqual(response.text, _plotly_basic_js())

    def test_chart_endpoint_passes_upload_identity(self):
        with patch.object(turn, 'render_chart_ref', return_value='<div>chart</div>') as render:
            response = self.client.post('/chart', data={'session_id': 'regression', 'source': 'upload', 'upload_id': 'first'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(render.call_args.args[1]['upload_id'], 'first')

    def test_large_upload_rejected_before_analysis(self):
        with patch.object(turn, 'handle_turn') as handle:
            response = self.client.post('/turn', data={'session_id': 'oversize'}, files={'file': ('large.csv', b'0' * (20 * 1024 * 1024 + 1))})
            self.assertEqual(response.status_code, 413)
            handle.assert_not_called()


if __name__ == '__main__':
    unittest.main()
