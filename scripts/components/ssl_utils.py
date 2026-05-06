"""
Scoped SSL context for corporate proxy environments (Zscaler).
Patches requests.Session.request to always use verify=False.
"""
import ssl
import urllib3
import warnings
from requests import Session

# Suppress InsecureRequestWarning globally
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

_session = None

# Patch requests.Session.request to always disable SSL verification.
# This catches ALL requests made through the requests library, including
# those made by third-party SDKs (firecrawl, tavily, etc.) that use requests.post/get.
_orig_session_request = Session.request

def _unverified_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return _orig_session_request(self, method, url, **kwargs)

Session.request = _unverified_request


def get_unsafe_session():
    """
    Returns a requests Session with SSL verification disabled.
    Safe to use: only affects this session, not the global context.
    """
    global _session
    if _session is None:
        _session = Session()
        _session.verify = False
        urllib3.disable_warnings()
    return _session
