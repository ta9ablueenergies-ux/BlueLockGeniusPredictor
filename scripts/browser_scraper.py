"""Optional browser-backed scraping helpers.

The pipeline prefers cheap, deterministic sources first. This module is the
browser fallback layer for pages that require JavaScript rendering or site
interaction. Playwright is tried first, SeleniumBase is the fallback.

The scraper is intentionally config-driven:
  - define browser sources in ``web/data/browser_sources.json``
  - each source can expose one or more URL templates and CSS selectors
  - if the config is absent, the module stays inert

This keeps the browser path efficient and optional instead of turning it into
another hard dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional
from urllib.parse import parse_qs, urlparse

from team_name_normalizer import canonical_fixture_key, canonical_team_name

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "web", "data", "browser_sources.json")
DEFAULT_CACHE_DIR = os.path.join(PROJECT_ROOT, "cache", "browser")


@dataclass
class BrowserSourceSpec:
    name: str
    league: Optional[str] = None
    kind: str = "fixtures"
    enabled: bool = True
    url_templates: List[str] = field(default_factory=list)
    wait_selector: Optional[str] = None
    card_selector: Optional[str] = None
    home_selector: Optional[str] = None
    away_selector: Optional[str] = None
    date_selector: Optional[str] = None
    time_selector: Optional[str] = None
    score_selector: Optional[str] = None
    home_odds_selector: Optional[str] = None
    draw_odds_selector: Optional[str] = None
    away_odds_selector: Optional[str] = None
    link_selector: Optional[str] = None
    text_pattern: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    extra: Dict[str, object] = field(default_factory=dict)


def _safe_date(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(text[:10], fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return text[:10]


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: object) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def _ftr_from_goals(home_goals: Optional[int], away_goals: Optional[int]) -> Optional[str]:
    if home_goals is None or away_goals is None:
        return None
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _extract_flashscore_id(url: object) -> Optional[str]:
    text = str(url or "").strip()
    if not text:
        return None
    try:
        parsed = urlparse(text)
        mid = parse_qs(parsed.query).get("mid", [None])[0]
        if mid:
            return str(mid).strip()
    except Exception:
        pass
    match = re.search(r"(?:mid=|match-row-g_\d_)([A-Za-z0-9]+)", text)
    return match.group(1) if match else None


def _normalize_league_key(league: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(league or "").strip().lower()).strip("-")


def _clean_flashscore_team_name(value: object) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    for marker in (
        "Advancing to next round:",
        "Winner:",
        "After Penalties:",
        "After Extra Time:",
    ):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text.strip(" -|")


def _normalize_date_key(value: object) -> Optional[str]:
    return _safe_date(value)


def _cache_key(url: str, selector: Optional[str]) -> str:
    payload = f"{url}::{selector or ''}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _cache_path(url: str, selector: Optional[str]) -> str:
    os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
    return os.path.join(DEFAULT_CACHE_DIR, f"{_cache_key(url, selector)}.json")


def _load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _store_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _template_context(match_date: str, league: str) -> Dict[str, str]:
    date_value = _safe_date(match_date) or str(match_date)
    try:
        dt = datetime.strptime(date_value[:10], "%Y-%m-%d")
    except Exception:
        dt = None
    return {
        "date": date_value,
        "date_compact": dt.strftime("%Y%m%d") if dt else date_value.replace("-", ""),
        "date_ddmmyyyy": dt.strftime("%d%m%Y") if dt else date_value.replace("-", "")[-8:],
        "date_slash": dt.strftime("%d/%m/%Y") if dt else date_value,
        "league": league,
        "league_slug": _normalize_league_key(league),
    }


def _render_templates(templates: Iterable[str], match_date: str, league: str) -> List[str]:
    ctx = _template_context(match_date, league)
    rendered: List[str] = []
    for template in templates:
        if not template:
            continue
        try:
            rendered.append(template.format(**ctx))
        except Exception:
            rendered.append(template)
    return rendered


def load_browser_source_specs(config_path: Optional[str] = None) -> List[BrowserSourceSpec]:
    path = config_path or os.environ.get("BROWSER_SOURCE_CONFIG") or DEFAULT_CONFIG_PATH
    payload = _load_json(path)
    if not payload:
        return []
    raw_sources = payload.get("sources") if isinstance(payload, dict) else payload
    if not isinstance(raw_sources, list):
        return []

    specs: List[BrowserSourceSpec] = []
    for item in raw_sources:
        if not isinstance(item, dict):
            continue
        templates = item.get("url_templates") or item.get("url_template") or []
        if isinstance(templates, str):
            templates = [templates]
        specs.append(
            BrowserSourceSpec(
                name=str(item.get("name") or item.get("source") or "browser_source"),
                league=item.get("league"),
                kind=str(item.get("kind") or "fixtures"),
                enabled=bool(item.get("enabled", True)),
                url_templates=[str(template) for template in templates if template],
                wait_selector=item.get("wait_selector"),
                card_selector=item.get("card_selector"),
                home_selector=item.get("home_selector"),
                away_selector=item.get("away_selector"),
                date_selector=item.get("date_selector"),
                time_selector=item.get("time_selector"),
                score_selector=item.get("score_selector"),
                home_odds_selector=item.get("home_odds_selector"),
                draw_odds_selector=item.get("draw_odds_selector"),
                away_odds_selector=item.get("away_odds_selector"),
                link_selector=item.get("link_selector"),
                text_pattern=item.get("text_pattern"),
                headers=item.get("headers") or {},
                extra=item.get("extra") or {},
            )
        )
    return specs


def _probe_package(name: str) -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _probe_browser_binary() -> Dict[str, Optional[str]]:
    candidates = [
        os.environ.get("EDGE_BINARY_PATH"),
        os.environ.get("CHROME_BINARY_PATH"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return {"binary": candidate}
    return {"binary": None}


def browser_runtime_status() -> Dict[str, bool]:
    binary = _probe_browser_binary()["binary"]
    return {
        "playwright": _probe_package("playwright"),
        "seleniumbase": _probe_package("seleniumbase"),
        "browser_binary": bool(binary),
    }


class BrowserRenderer:
    """Batch browser renderer with a Playwright-first execution path."""

    def __init__(
        self,
        engine: Optional[str] = None,
        headless: bool = True,
        timeout_ms: int = 20000,
        block_resources: bool = True,
        browser_binary: Optional[str] = None,
        browser_channel: Optional[str] = None,
    ):
        self.engine = (engine or os.environ.get("BROWSER_ENGINE") or "auto").strip().lower()
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.block_resources = block_resources
        self.browser_binary = browser_binary or _probe_browser_binary()["binary"]
        self.browser_channel = browser_channel or os.environ.get("BROWSER_CHANNEL") or "msedge"
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._sb_cm = None
        self._sb = None
        self._active_engine = None

    def __enter__(self) -> "BrowserRenderer":
        if self.engine == "auto":
            preferred = os.environ.get("BROWSER_AUTO_PREFER", "seleniumbase").strip().lower()
            engines = [preferred, "playwright" if preferred == "seleniumbase" else "seleniumbase"]
        else:
            engines = [self.engine]

        for engine in engines:
            if engine == "playwright" and _probe_package("playwright"):
                try:
                    self._start_playwright()
                    self._active_engine = "playwright"
                    return self
                except Exception as exc:
                    if self.engine == "playwright":
                        raise
                    self._playwright_error = exc
            if engine == "seleniumbase" and _probe_package("seleniumbase"):
                try:
                    self._start_seleniumbase()
                    self._active_engine = "seleniumbase"
                    return self
                except Exception as exc:
                    if self.engine == "seleniumbase":
                        raise
                    self._seleniumbase_error = exc

        raise RuntimeError("No browser engine could be started")

    def __exit__(self, exc_type, exc, tb):
        if self._page is not None:
            try:
                self._page.close()
            except Exception:
                pass
        if self._context is not None:
            try:
                self._context.close()
            except Exception:
                pass
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:
                pass
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
        if self._sb_cm is not None:
            try:
                self._sb_cm.__exit__(exc_type, exc, tb)
            except Exception:
                pass
        return False

    @property
    def active_engine(self) -> Optional[str]:
        return self._active_engine

    def _start_playwright(self) -> None:
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        launch_kwargs = {
            "headless": self.headless,
        }
        use_bundled = os.environ.get("PLAYWRIGHT_USE_BUNDLED", "0").strip().lower() in {"1", "true", "yes", "on"}
        if not use_bundled:
            prefer_channel = os.environ.get("PLAYWRIGHT_USE_EXECUTABLE_PATH", "0").strip().lower() not in {"1", "true", "yes", "on"}
            if self.browser_channel and prefer_channel:
                launch_kwargs["channel"] = self.browser_channel
            elif self.browser_binary:
                launch_kwargs["executable_path"] = self.browser_binary
            elif self.browser_channel:
                launch_kwargs["channel"] = self.browser_channel
        launch_kwargs["args"] = [
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-blink-features=AutomationControlled",
        ]
        self._browser = self._playwright.chromium.launch(**launch_kwargs)
        self._context = self._browser.new_context(
            viewport={"width": 1440, "height": 1080},
            ignore_https_errors=True,
        )
        if self.block_resources:
            self._context.route("**/*", self._route_handler)
        self._page = self._context.new_page()

    def _start_seleniumbase(self) -> None:
        from seleniumbase import SB

        browser_name = os.environ.get("SELENIUM_BROWSER")
        if not browser_name:
            # Infer browser from binary path so the driver version matches
            if self.browser_binary and "edge" in self.browser_binary.lower():
                browser_name = "edge"
            else:
                browser_name = "chrome"

        kwargs = {
            "browser": browser_name,
            "headless": self.headless,
            "uc": True,      # Undetected-Chromedriver to bypass Cloudflare
            "uc_cdp": True,  # CDP stealth
            "block_images": True,
            "disable_js": False,
            "timeout_multiplier": 1,
        }
        if self.browser_binary:
            kwargs["binary_location"] = self.browser_binary

        self._sb_cm = SB(**kwargs)
        self._sb = self._sb_cm.__enter__()

    def _route_handler(self, route) -> None:
        try:
            resource_type = route.request.resource_type
            if resource_type in {"image", "media", "font", "stylesheet"}:
                route.abort()
            else:
                route.continue_()
        except Exception:
            try:
                route.continue_()
            except Exception:
                pass

    def fetch(self, url: str, wait_selector: Optional[str] = None, use_cache: bool = True) -> Dict[str, object]:
        cache_path = _cache_path(url, wait_selector)
        if use_cache:
            cached = _load_json(cache_path)
            cache_ttl = int(os.environ.get("BROWSER_CACHE_TTL_SECONDS", "1800"))
            if cached:
                cached_at = cached.get("cached_at")
                if cached_at:
                    try:
                        age = time.time() - float(cached_at)
                        if age <= cache_ttl:
                            return cached["payload"]
                    except Exception:
                        pass

        if self._active_engine == "playwright":
            payload = self._fetch_playwright(url, wait_selector)
        elif self._active_engine == "seleniumbase":
            payload = self._fetch_seleniumbase(url, wait_selector)
        else:
            raise RuntimeError("Browser renderer not initialized")

        if use_cache:
            _store_json(cache_path, {"cached_at": time.time(), "payload": payload})
        return payload

    def _fetch_playwright(self, url: str, wait_selector: Optional[str] = None) -> Dict[str, object]:
        assert self._page is not None
        self._page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
        try:
            self._page.wait_for_load_state("networkidle", timeout=self.timeout_ms)
        except Exception:
            pass
        if wait_selector:
            try:
                self._page.wait_for_selector(wait_selector, timeout=self.timeout_ms)
            except Exception:
                pass
        html = self._page.content()
        text = ""
        try:
            text = self._page.locator("body").inner_text(timeout=self.timeout_ms)
        except Exception:
            pass
        return {
            "engine": "playwright",
            "url": url,
            "final_url": self._page.url,
            "title": self._page.title(),
            "html": html,
            "text": text,
        }

    def _fetch_seleniumbase(self, url: str, wait_selector: Optional[str] = None) -> Dict[str, object]:
        assert self._sb is not None
        self._sb.open(url)
        try:
            self._sb.wait_for_ready_state_complete()
        except Exception:
            pass
        if wait_selector:
            try:
                self._sb.wait_for_element_visible(wait_selector, timeout=self.timeout_ms / 1000.0)
            except Exception:
                pass
        try:
            html = self._sb.get_page_source()
        except Exception:
            html = ""
        try:
            text = self._sb.get_text("body")
        except Exception:
            text = ""
        try:
            title = self._sb.get_title()
        except Exception:
            title = ""
        try:
            final_url = self._sb.get_current_url()
        except Exception:
            final_url = url
        return {
            "engine": "seleniumbase",
            "url": url,
            "final_url": final_url,
            "title": title,
            "html": html,
            "text": text,
        }

    def extract_cards(self, page_payload: Dict[str, object], spec: BrowserSourceSpec) -> List[Dict[str, object]]:
        if self._active_engine == "playwright":
            return self._extract_cards_playwright(spec)
        if self._active_engine == "seleniumbase":
            return self._extract_cards_seleniumbase(spec)
        return []

    def _extract_cards_playwright(self, spec: BrowserSourceSpec) -> List[Dict[str, object]]:
        assert self._page is not None
        if not spec.card_selector:
            return []
        script = self._build_card_extractor_js(spec)
        try:
            return self._page.evaluate(script) or []
        except Exception:
            return []

    def _extract_cards_seleniumbase(self, spec: BrowserSourceSpec) -> List[Dict[str, object]]:
        assert self._sb is not None
        if not spec.card_selector:
            return []
        js = self._build_card_extractor_js(spec)
        try:
            return self._sb.driver.execute_script(js) or []
        except Exception:
            return []

    def _build_card_extractor_js(self, spec: BrowserSourceSpec) -> str:
        selector_payload = {
            "card_selector": spec.card_selector or "",
            "home_selector": spec.home_selector or "",
            "away_selector": spec.away_selector or "",
            "date_selector": spec.date_selector or "",
            "time_selector": spec.time_selector or "",
            "score_selector": spec.score_selector or "",
            "home_odds_selector": spec.home_odds_selector or "",
            "draw_odds_selector": spec.draw_odds_selector or "",
            "away_odds_selector": spec.away_odds_selector or "",
            "link_selector": spec.link_selector or "a[href]",
        }
        return f"""
        (() => {{
            const cfg = {json.dumps(selector_payload)};
            const cards = Array.from(document.querySelectorAll(cfg.card_selector));
            const grabText = (root, sel) => {{
                if (!sel) return "";
                const el = root.querySelector(sel);
                return el ? (el.textContent || el.innerText || "").trim() : "";
            }};
            const grabHref = (root, sel) => {{
                if (!sel) return "";
                const el = root.querySelector(sel);
                if (!el) return "";
                return el.href || el.getAttribute("href") || "";
            }};
            return cards.map((card) => {{
                const cardText = (card.innerText || "").trim();
                return {{
                    card_text: cardText,
                    home_team: grabText(card, cfg.home_selector),
                    away_team: grabText(card, cfg.away_selector),
                    match_date: grabText(card, cfg.date_selector),
                    kickoff_utc: grabText(card, cfg.time_selector),
                    score_text: grabText(card, cfg.score_selector),
                    home_odds: grabText(card, cfg.home_odds_selector),
                    draw_odds: grabText(card, cfg.draw_odds_selector),
                    away_odds: grabText(card, cfg.away_odds_selector),
                    match_url: grabHref(card, cfg.link_selector),
                }};
            }});
        }})()
        """


def _infer_match_from_text(text: str, target_date: str, league: str, source_name: str) -> Optional[Dict[str, object]]:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return None

    patterns = [
        r"^(?P<home>.+?)\s+(?:vs\.?|v\.?|versus)\s+(?P<away>.+?)(?:\s+\d{1,2}:\d{2}|\s+\d+-\d+|\s+LIVE|$)",
        r"^(?P<home>.+?)\s+-\s+(?P<away>.+?)(?:\s+\d{1,2}:\d{2}|\s+\d+-\d+|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            home = match.group("home").strip(" -|")
            away = match.group("away").strip(" -|")
            kickoff = None
            kickoff_match = re.search(r"\b\d{1,2}:\d{2}\b", cleaned)
            if kickoff_match:
                kickoff = kickoff_match.group(0)
            score_match = re.search(r"\b(\d+)\s*[-:]\s*(\d+)\b", cleaned)
            home_goals = int(score_match.group(1)) if score_match else None
            away_goals = int(score_match.group(2)) if score_match else None
            return {
                "league": league,
                "match_date": target_date,
                "home_team": home,
                "away_team": away,
                "kickoff_utc": kickoff,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "actual_ftr": _ftr_from_goals(home_goals, away_goals),
                "home_odds": None,
                "draw_odds": None,
                "away_odds": None,
                "source": f"browser:{source_name}",
                "source_url": None,
            }
    return None


def _extract_flashscore_matches_from_text(text: str, target_date: str, league: str, source_name: str) -> List[Dict[str, object]]:
    """Parse Flashscore-style rendered text that groups matches under date prefixes."""
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return []

    try:
        target_year = datetime.strptime(target_date[:10], "%Y-%m-%d").year
    except Exception:
        target_year = datetime.utcnow().year

    def _is_noise_line(line: str) -> bool:
        candidate = str(line or "").strip()
        if not candidate:
            return True
        if candidate.lower() in {
            "advertisement",
            "show more matches",
            "show more",
            "live standings",
            "standings",
            "form",
            "over/under",
            "ht/ft",
            "top scorers",
        }:
            return True
        if candidate.lower().startswith(("round ", "scheduled", "latest scores", "football", "summary", "odds", "news", "results", "fixtures", "standings", "archive")):
            return True
        if re.fullmatch(r"\d{1,2}", candidate):
            return True
        return False

    def _next_meaningful_line(lines: List[str], start_idx: int) -> Optional[str]:
        for idx in range(start_idx, len(lines)):
            candidate = str(lines[idx] or "").strip()
            if not _is_noise_line(candidate):
                return candidate
        return None

    raw_lines = [line.strip() for line in str(text or "").splitlines()]
    parsed: List[Dict[str, object]] = []
    for idx, line in enumerate(raw_lines):
        if not line:
            continue
        match = re.match(r"^(?P<date>\d{2}\.\d{2}\.)\s+(?P<time>\d{1,2}:\d{2})$", line)
        if not match:
            continue
        try:
            parsed_date = datetime.strptime(f"{match.group('date')}{target_year}", "%d.%m.%Y").strftime("%Y-%m-%d")
        except Exception:
            parsed_date = target_date
        if parsed_date != target_date:
            continue
        home = _next_meaningful_line(raw_lines, idx + 1)
        away = _next_meaningful_line(raw_lines, idx + 2)
        if not home or not away:
            continue
        if home == away:
            continue
        parsed.append(
            {
                "league": league,
                "match_date": target_date,
                "home_team": home.strip(" -|"),
                "away_team": away.strip(" -|"),
                "kickoff_utc": match.group("time"),
                "home_goals": None,
                "away_goals": None,
                "actual_ftr": None,
                "home_odds": None,
                "draw_odds": None,
                "away_odds": None,
                "source": f"browser:{source_name}",
                "source_url": None,
            }
        )
    if parsed:
        return parsed

    date_tokens = re.findall(r"(?<!\d)(\d{2}\.\d{2}\.)\s+", cleaned)
    if not date_tokens:
        inferred = _infer_match_from_text(cleaned, target_date, league, source_name)
        return [inferred] if inferred else []

    results: List[Dict[str, object]] = []
    segments = re.split(r"(?<!\d)(\d{2}\.\d{2}\.)\s+", cleaned)
    for idx in range(1, len(segments), 2):
        date_token = segments[idx].strip()
        segment_text = segments[idx + 1].strip() if idx + 1 < len(segments) else ""
        if not segment_text:
            continue
        try:
            parsed = datetime.strptime(f"{date_token}{target_year}", "%d.%m.%Y")
            segment_date = parsed.strftime("%Y-%m-%d")
        except Exception:
            segment_date = target_date
        if segment_date != target_date:
            continue
        chunk = segment_text
        chunk = chunk.replace("Show more", "").strip(" ,;")
        parts = [part.strip() for part in chunk.split(",") if part.strip()]
        for part in parts:
            inferred = _infer_match_from_text(part, target_date, league, source_name)
            if inferred:
                results.append(inferred)
    return results


def _extract_flashscore_matches_from_dom(renderer: BrowserRenderer, target_date: str, league: str, source_name: str) -> List[Dict[str, object]]:
    """Extract Flashscore event rows with durable match-detail URLs."""
    js = """
    () => Array.from(document.querySelectorAll('.event__match')).map((row) => {
        const textOf = (sel) => {
            const el = row.querySelector(sel);
            return el ? (el.textContent || el.innerText || '').trim() : '';
        };
        const link = row.querySelector('.eventRowLink');
        return {
            row_id: row.id || '',
            row_text: (row.innerText || row.textContent || '').trim(),
            time_text: textOf('.event__time') || textOf('.event__stage') || textOf('.eventTime'),
            home_team: textOf('.event__homeParticipant'),
            away_team: textOf('.event__awayParticipant'),
            home_score: textOf('.event__score--home'),
            away_score: textOf('.event__score--away'),
            match_url: link ? (link.href || link.getAttribute('href') || '') : '',
            is_live: row.classList.contains('event__match--live'),
            is_finished: row.classList.contains('event__match--last') || row.classList.contains('event__match--finished'),
        };
    }).filter((item) => item.home_team && item.away_team)
    """
    try:
        page = getattr(renderer, "_page", None)
        if page is not None:
            rows = page.evaluate(js) or []
        elif getattr(renderer, "_sb", None) is not None:
            rows = renderer._sb.driver.execute_script(f"return ({js})()") or []
        else:
            return []
    except Exception:
        return []

    try:
        target_year = datetime.strptime(target_date[:10], "%Y-%m-%d").year
    except Exception:
        target_year = datetime.utcnow().year
    today_key = datetime.now().strftime("%Y-%m-%d")

    parsed: List[Dict[str, object]] = []
    for row in rows:
        time_text = str(row.get("time_text") or "").strip()
        row_text = str(row.get("row_text") or "").strip()
        combined = f"{time_text}\n{row_text}"
        date_match = re.search(r"(?P<date>\d{2}\.\d{2}\.)\s*(?P<time>\d{1,2}:\d{2})", combined)
        time_match = re.search(r"\b(?P<time>\d{1,2}:\d{2})\b", time_text or row_text)
        if date_match:
            try:
                match_date = datetime.strptime(f"{date_match.group('date')}{target_year}", "%d.%m.%Y").strftime("%Y-%m-%d")
            except Exception:
                match_date = target_date
            kickoff = date_match.group("time")
        elif time_match:
            # No explicit date in row — assume target_date (the URL-requested date)
            # This handles future fixture pages where Flashscore shows only times
            match_date = target_date
            kickoff = time_match.group("time")
        else:
            continue

        if match_date != target_date:
            continue

        home_goals = _safe_int(row.get("home_score"))
        away_goals = _safe_int(row.get("away_score"))
        url = str(row.get("match_url") or "").strip() or None
        parsed.append(
            {
                "league": league,
                "match_date": match_date,
                "home_team": _clean_flashscore_team_name(row.get("home_team")),
                "away_team": _clean_flashscore_team_name(row.get("away_team")),
                "kickoff_utc": kickoff,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "actual_ftr": _ftr_from_goals(home_goals, away_goals),
                "home_odds": None,
                "draw_odds": None,
                "away_odds": None,
                "source": f"browser:{source_name}",
                "source_url": url,
                "source_id": _extract_flashscore_id(url or row.get("row_id")),
            }
        )
    return parsed


def _card_to_match_record(card: Dict[str, object], spec: BrowserSourceSpec, target_date: str) -> Optional[Dict[str, object]]:
    home = str(card.get("home_team") or "").strip()
    away = str(card.get("away_team") or "").strip()
    text = str(card.get("card_text") or "")
    inferred = _infer_match_from_text(text, target_date, spec.league or "", spec.name)

    if not home and inferred:
        home = inferred["home_team"]
    if not away and inferred:
        away = inferred["away_team"]
    if not home or not away:
        return inferred

    match_date = _safe_date(card.get("match_date")) or target_date
    kickoff = str(card.get("kickoff_utc") or "").strip() or (inferred.get("kickoff_utc") if inferred else None)
    score_text = str(card.get("score_text") or "").strip()
    home_goals = _safe_int(card.get("home_goals"))
    away_goals = _safe_int(card.get("away_goals"))
    if (home_goals is None or away_goals is None) and score_text:
        score_match = re.search(r"(\d+)\s*[-:]\s*(\d+)", score_text)
        if score_match:
            home_goals = int(score_match.group(1))
            away_goals = int(score_match.group(2))

    return {
        "league": spec.league or "",
        "match_date": match_date,
        "home_team": home,
        "away_team": away,
        "kickoff_utc": kickoff,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "actual_ftr": _ftr_from_goals(home_goals, away_goals),
        "home_odds": _safe_float(card.get("home_odds")),
        "draw_odds": _safe_float(card.get("draw_odds")),
        "away_odds": _safe_float(card.get("away_odds")),
        "source": f"browser:{spec.name}",
        "source_url": card.get("match_url") or None,
    }


def discover_browser_fixtures_for_date(
    match_date: str,
    leagues: Optional[List[str]] = None,
    config_path: Optional[str] = None,
    engine_preference: Optional[str] = None,
) -> List["FreeMatch"]:
    """Render browser pages and extract fixture cards for a selected date."""
    from free_football_source import FreeMatch  # lazy import to avoid cycles

    target_date = _normalize_date_key(match_date)
    if not target_date:
        return []

    specs = load_browser_source_specs(config_path=config_path)
    if not specs:
        return []

    allowed = set(leagues) if leagues else None
    engine = (engine_preference or os.environ.get("BROWSER_ENGINE") or "auto").strip().lower()
    timeout_ms = int(os.environ.get("BROWSER_TIMEOUT_MS", "20000"))
    headless = os.environ.get("BROWSER_HEADLESS", "1").strip().lower() not in {"0", "false", "no", "off"}

    matches: List[FreeMatch] = []
    with BrowserRenderer(engine=engine, headless=headless, timeout_ms=timeout_ms) as renderer:
        for spec in specs:
            if not spec.enabled:
                continue
            if allowed and spec.league and spec.league not in allowed:
                continue
            if spec.kind not in {"fixtures", "fixture_list", "schedule"}:
                continue

            for template in spec.url_templates:
                url = _render_templates([template], target_date, spec.league or "")[0]
                text_mode = str(spec.extra.get("text_mode") or "").strip().lower()
                page_payload = renderer.fetch(url, wait_selector=spec.wait_selector, use_cache=text_mode != "flashscore")
                cards = renderer.extract_cards(page_payload, spec)

                if cards:
                    for card in cards:
                        record = _card_to_match_record(card, spec, target_date)
                        if not record:
                            continue
                        if record.get("match_date") and _safe_date(record["match_date"]) != target_date:
                            continue
                        if allowed and record.get("league") and record["league"] not in allowed:
                            continue
                        matches.append(
                            FreeMatch(
                                league=str(record.get("league") or spec.league or ""),
                                match_date=str(record.get("match_date") or target_date),
                                home_team=str(record.get("home_team") or "").strip(),
                                away_team=str(record.get("away_team") or "").strip(),
                                kickoff_utc=str(record.get("kickoff_utc") or "").strip() or None,
                                home_goals=_safe_int(record.get("home_goals")),
                                away_goals=_safe_int(record.get("away_goals")),
                                actual_ftr=str(record.get("actual_ftr") or "").strip()[:1] or None,
                                home_odds=_safe_float(record.get("home_odds")),
                                draw_odds=_safe_float(record.get("draw_odds")),
                                away_odds=_safe_float(record.get("away_odds")),
                                source_url=str(record.get("source_url") or "").strip() or None,
                                source_id=_extract_flashscore_id(record.get("source_url")),
                                source=str(record.get("source") or f"browser:{spec.name}"),
                            )
                        )
                    continue

                if text_mode == "flashscore":
                    inferred_list = _extract_flashscore_matches_from_dom(
                        renderer,
                        target_date,
                        spec.league or "",
                        spec.name,
                    ) or _extract_flashscore_matches_from_text(
                        str(page_payload.get("text") or ""),
                        target_date,
                        spec.league or "",
                        spec.name,
                    )
                else:
                    inferred = _infer_match_from_text(
                        str(page_payload.get("text") or ""),
                        target_date,
                        spec.league or "",
                        spec.name,
                    )
                    inferred_list = [inferred] if inferred else []

                for inferred in inferred_list:
                    if not inferred:
                        continue
                    if allowed and inferred.get("league") and inferred["league"] not in allowed:
                        continue
                    matches.append(
                        FreeMatch(
                            league=str(inferred.get("league") or spec.league or ""),
                            match_date=str(inferred.get("match_date") or target_date),
                            home_team=str(inferred.get("home_team") or "").strip(),
                            away_team=str(inferred.get("away_team") or "").strip(),
                            kickoff_utc=str(inferred.get("kickoff_utc") or "").strip() or None,
                            home_goals=_safe_int(inferred.get("home_goals")),
                            away_goals=_safe_int(inferred.get("away_goals")),
                            actual_ftr=str(inferred.get("actual_ftr") or "").strip()[:1] or None,
                            home_odds=_safe_float(inferred.get("home_odds")),
                            draw_odds=_safe_float(inferred.get("draw_odds")),
                            away_odds=_safe_float(inferred.get("away_odds")),
                            source_url=str(inferred.get("source_url") or "").strip() or None,
                            source_id=str(inferred.get("source_id") or "").strip() or _extract_flashscore_id(inferred.get("source_url")),
                            source=str(inferred.get("source") or f"browser:{spec.name}"),
                        )
                    )

    seen = set()
    deduped: List[FreeMatch] = []
    for item in matches:
        item.home_team = canonical_team_name(item.league, item.home_team)
        item.away_team = canonical_team_name(item.league, item.away_team)
        key = canonical_fixture_key(item.league, item.match_date, item.home_team, item.away_team)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def discover_browser_fixtures(
    leagues: Optional[List[str]] = None,
    match_date: Optional[str] = None,
    config_path: Optional[str] = None,
    engine_preference: Optional[str] = None,
) -> List["FreeMatch"]:
    if not match_date:
        return []
    return discover_browser_fixtures_for_date(
        match_date=match_date,
        leagues=leagues,
        config_path=config_path,
        engine_preference=engine_preference,
    )
