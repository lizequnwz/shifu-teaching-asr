import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.workflow_cloud import BudgetExceeded, CloudError, CloudLedger, RequestPending, api_key


class RateLimited(Exception):
    status_code = 429


def answer(response, raw):
    return raw["text"]


class CloudLedgerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.job = Path(temporary.name)
        self.ledger = CloudLedger(self.job, 1)

    def request(self, send, *, provider="Gemini", reservation=0.25, ledger=None, parse=answer):
        ledger = self.ledger if ledger is None else ledger
        return ledger.request(provider, "model", "transcription", {"audio_sha256": "clip"}, reservation,
                              lambda: object(), send, parse, lambda usage: usage.get("spent"))

    def test_both_providers_share_budget_and_cached_results_do_not_use_it(self):
        send = Mock(return_value={"text": "cached", "usage": {"spent": 0.24}})
        self.assertEqual(self.request(send), "cached")
        self.assertAlmostEqual(self.ledger.usage()["exposure_usd"], 0.30)
        self.assertEqual(self.request(send, reservation=20), "cached")
        with self.assertRaises(BudgetExceeded):
            self.request(send, provider="OpenAI", reservation=0.75)
        send.assert_called_once()

    def test_timeout_retains_reservation_and_explicit_retry_preserves_prior_exposure(self):
        send = Mock(side_effect=TimeoutError("private provider error details"))
        with self.assertRaises(RequestPending) as raised:
            self.request(send)
        request_id = raised.exception.request_id
        self.assertNotIn("private provider error details", str(raised.exception))
        self.assertEqual(self.ledger.usage()["unresolved_requests"][0]["request_id"], request_id)
        self.assertAlmostEqual(self.ledger.usage()["exposure_usd"], 0.25)

        still_pending = CloudLedger(self.job, 1)
        with self.assertRaises(RequestPending):
            self.request(send, ledger=still_pending)
        send.assert_called_once()

        send.side_effect = None
        send.return_value = {"text": "retried", "usage": {"spent": 0.20}}
        retry = CloudLedger(self.job, 1, retry_request=request_id)
        self.assertEqual(self.request(send, ledger=retry), "retried")
        self.assertEqual(send.call_count, 2)
        self.assertEqual(retry.usage()["unresolved_requests"], [])
        self.assertAlmostEqual(retry.usage()["exposure_usd"], 0.50)

    @patch("scripts.workflow_cloud.time.sleep")
    def test_only_rate_limited_requests_retry_automatically(self, sleep):
        send = Mock(side_effect=[RateLimited(), RateLimited()])
        with self.assertRaisesRegex(CloudError, "rate-limited"):
            self.request(send)
        self.assertEqual(send.call_count, 2)
        sleep.assert_called_once()
        usage = self.ledger.usage()
        self.assertEqual(usage["by_status"], {"rate_limited": 2})
        self.assertEqual(usage["exposure_usd"], 0)

    def test_invalid_content_is_not_cached_as_a_success(self):
        send = Mock(return_value={"text": "not structured JSON", "usage": {"spent": 0.1}})
        for _ in range(2):
            with self.assertRaises(RequestPending):
                self.request(send, parse=lambda _, raw: json.loads(raw["text"]))
        send.assert_called_once()
        self.assertEqual(self.ledger.usage()["by_status"], {"invalid_response": 1})

    def test_explicit_environment_key_takes_precedence_over_literal_dotenv(self):
        pretend_module = self.job / "scripts" / "workflow_cloud.py"
        (self.job / ".env").write_text("export SAMPLE_KEY='literal $(not a command)'\n", "utf-8")
        with patch("scripts.workflow_cloud.__file__", str(pretend_module)), patch.dict(os.environ, clear=True):
            self.assertEqual(api_key("SAMPLE_KEY"), "literal $(not a command)")
            os.environ["SAMPLE_KEY"] = "from environment"
            self.assertEqual(api_key("SAMPLE_KEY"), "from environment")


if __name__ == "__main__":
    unittest.main()
