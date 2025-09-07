
from flask import Flask, render_template, jsonify
import requests, traceback, os, math, logging
from datetime import datetime, timedelta, timezone
from requests.adapters import HTTPAdapter, Retry

API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "b7ea33d435964da0b0a65b1c6a029891")
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "EUR/JPY", "AUD/USD"]
INTERVAL = "5min"
INITIAL_CANDLES = 30
MAX_CANDLES_KEEP = 120
IST = timezone(timedelta(hours=5, minutes=30))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cwrv_v2")

app = Flask(__name__, static_folder="static", template_folder="templates")
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.8, status_forcelist=[429,500,502,503,504])
session.mount("https://", HTTPAdapter(max_retries=retries))
session.headers.update({"User-Agent": "CWRV-123-BO/2.0"})

state = {"series": {pair: [] for pair in PAIRS}, "last_fetch": {pair: None for pair in PAIRS}}

def now_str(): return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
def log_info(msg): logger.info(msg)

def td_url(symbol, last_n=None, start_date=None):
    base_url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": INTERVAL, "apikey": API_KEY}
    if last_n: params["outputsize"] = str(last_n)
    if start_date: params["start_date"] = start_date
    q = "&".join(f"{k}={requests.utils.quote(str(v))}" for k,v in params.items())
    return f"{base_url}?{q}"

def safe_get(url, timeout=20):
    try:
        resp = session.get(url, timeout=timeout)
    except requests.RequestException as e:
        return None, f"Network error: {e}"
    if resp.status_code != 200:
        try:
            j = resp.json()
            msg = j.get("message") or j.get("status") or str(j)
        except Exception:
            msg = resp.text[:200]
        return None, f"HTTP {resp.status_code}: {msg}"
    try:
        j = resp.json()
    except ValueError:
        return None, "Invalid JSON from API"
    if isinstance(j, dict) and (j.get("status") == "error" or ("code" in j and j.get("code") != 200)):
        msg = j.get("message") or j.get("status") or str(j)
        return None, f"API error: {msg}"
    return j, None

def parse_series(json_obj):
    if not json_obj:
        return [], "Empty response"
    vals = json_obj.get("values")
    if not vals:
        return [], "No 'values' in response"
    out = []
    for row in vals:
        try:
            dt = row.get("datetime")
            if not dt: return [], "Row missing datetime"
            t = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            out.append({"ts": int(t.timestamp()*1000), "open": float(row["open"]), "high": float(row["high"]), "low": float(row["low"]), "close": float(row["close"])})
        except Exception as e:
            return [], f"Parse error: {e}"
    out.sort(key=lambda x: x["ts"])
    return out, None

def fetch_initial(pair):
    url = td_url(pair, last_n=INITIAL_CANDLES)
    j, err = safe_get(url)
    if err: raise RuntimeError(err)
    candles, perr = parse_series(j)
    if perr: raise RuntimeError(perr)
    state["series"][pair] = candles
    state["last_fetch"][pair] = datetime.now(IST)
    return candles

def fetch_incremental(pair):
    existing = state["series"].get(pair, [])
    if not existing: return fetch_initial(pair)
    last_ts_ms = existing[-1]["ts"]
    last_dt_utc = datetime.fromtimestamp(last_ts_ms/1000, tz=timezone.utc) + timedelta(seconds=1)
    start_iso = last_dt_utc.strftime("%Y-%m-%d %H:%M:%S")
    url = td_url(pair, start_date=start_iso)
    j, err = safe_get(url)
    if err:
        log_info(f"Incremental fetch error for {pair}: {err}")
        return existing
    new_candles, perr = parse_series(j)
    if perr:
        log_info(f"Parse error on incremental for {pair}: {perr}")
        return existing
    merged = existing[:]
    for c in new_candles:
        if c["ts"] > last_ts_ms:
            merged.append(c)
    if len(merged) > MAX_CANDLES_KEEP:
        merged = merged[-MAX_CANDLES_KEEP:]
    state["series"][pair] = merged
    state["last_fetch"][pair] = datetime.now(IST)
    return merged

def body_size(c): return abs(c["close"] - c["open"])
def candle_range(c): return c["high"] - c["low"]
def is_bull(c): return c["close"] >= c["open"]
def is_bear(c): return c["close"] < c["open"]
def pct(x,y): return (x / y * 100.0) if y != 0 else 0.0

def find_swings(candles, lookback=3):
    highs, lows = [], []
    n = len(candles)
    for i in range(lookback, n - lookback):
        window = candles[i - lookback:i + lookback + 1]
        try:
            h = max(x["high"] for x in window)
            l = min(x["low"] for x in window)
        except Exception:
            continue
        if math.isclose(candles[i]["high"], h) or candles[i]["high"] == h: highs.append(i)
        if math.isclose(candles[i]["low"], l) or candles[i]["low"] == l: lows.append(i)
    return highs, lows

def detect_trend(candles):
    try:
        if len(candles) < 15: return "sideways", {}
        sh, sl = find_swings(candles, lookback=2)
        recent_hi = sh[-4:]; recent_lo = sl[-4:]; info = {"swings_high": recent_hi, "swings_low": recent_lo}
        def hh(i1,i2): return candles[i2]["high"] > candles[i1]["high"]
        def hl(i1,i2): return candles[i2]["low"] > candles[i1]["low"]
        def ll(i1,i2): return candles[i2]["low"] < candles[i1]["low"]
        def lh(i1,i2): return candles[i2]["high"] < candles[i1]["high"]
        upcount_hh = sum(1 for a,b in zip(recent_hi, recent_hi[1:]) if hh(a,b))
        upcount_hl = sum(1 for a,b in zip(recent_lo, recent_lo[1:]) if hl(a,b))
        downcount_ll = sum(1 for a,b in zip(recent_lo, recent_lo[1:]) if ll(a,b))
        downcount_lh = sum(1 for a,b in zip(recent_hi, recent_hi[1:]) if lh(a,b))
        if upcount_hh >= 2 and upcount_hl >= 1: trend="up"
        elif downcount_ll >= 2 and downcount_lh >= 1: trend="down"
        else:
            last = candles[-10:]; slope = last[-1]["close"] - last[0]["close"]
            trend = "sideways" if abs(slope) < 1e-8 else ("up" if slope>0 else "down")
        info["trend"]=trend; return trend, info
    except Exception as e:
        log_info(f"Trend detect failed: {e}"); return "sideways", {}

def find_snr_levels(candles, swings_window=2):
    try:
        sh, sl = find_swings(candles, lookback=swings_window)
        levels = [candles[i]["high"] for i in sh] + [candles[i]["low"] for i in sl]
        levels = sorted(levels); merged=[]
        if not levels: return []
        tol = max(1e-8, 0.0001 * (levels[-1] if levels else 1))
        for lv in levels:
            if not merged or abs(lv - merged[-1]) > tol: merged.append(lv)
            else: merged[-1] = (merged[-1] + lv)/2.0
        last_close = candles[-1]["close"]; merged.sort(key=lambda x: abs(x-last_close)); return merged[:6]
    except Exception as e:
        log_info(f"SNR build failed: {e}"); return []

def nearest_level(levels, price):
    if not levels: return None, None
    lv = min(levels, key=lambda x: abs(x-price)); return lv, abs(lv-price)

def cwrv_signal(candles):
    reasons=[]
    try:
        if len(candles) < 15: return {"prediction":"NO_TRADE","take_trade":False,"reasons":["Not enough candles"],"level":None}
        trend,_ = detect_trend(candles); reasons.append(f"Trend: {trend}")
        levels = find_snr_levels(candles); prev = candles[-2]; prev2 = candles[-3]
        rng = candle_range(prev); bdy = body_size(prev); strong_body = bdy >= 0.5 * rng if rng>0 else False
        if strong_body: reasons.append("C: Strong body")
        near_lv, dist = nearest_level(levels, prev["close"]); c_confirm=False
        if near_lv is not None:
            if is_bull(prev) and prev["close"] > near_lv >= prev["open"]: c_confirm=True
            if is_bear(prev) and prev["close"] < near_lv <= prev["open"]: c_confirm=True
        if c_confirm: reasons.append(f"C: Closed past S/R ({near_lv:.5f})")
        upper_wick = prev["high"] - max(prev["open"], prev["close"]); lower_wick = min(prev["open"], prev["close"]) - prev["low"]
        wick_reject=False
        if near_lv is not None:
            if upper_wick > bdy and prev["high"] >= near_lv >= prev["close"]: wick_reject=True
            if lower_wick > bdy and prev["low"] <= near_lv <= prev["close"]: wick_reject=True
        if wick_reject: reasons.append("W: Wick rejection")
        retest=False
        if near_lv is not None:
            def near(a,b,eps): return abs(a-b) <= eps
            eps = max(1e-8, 0.0008 * prev["close"])
            if any(near(x, near_lv, eps) for x in [prev2["high"], prev2["low"], prev2["open"], prev2["close"]]) and any(near(x, near_lv, eps) for x in [prev["high"], prev["low"], prev["open"], prev["close"]]): retest=True
        if retest: reasons.append("R: Retest level")
        v_ok=False
        if len(candles) >= 4:
            r0=candle_range(prev); r1=candle_range(prev2); r2=candle_range(candles[-4])
            if r0 > max(r1, r2): v_ok=True
        if v_ok: reasons.append("V: Range>prev2 (momentum)")
        score = sum([1 if strong_body else 0, 1 if c_confirm else 0, 1 if wick_reject else 0, 1 if v_ok else 0])
        prediction="NO_TRADE"; take_trade=False
        if score >= 3:
            if trend=="up": prediction="CALL"; take_trade=True; reasons.append("Score>=3 and with uptrend ⇒ CALL")
            elif trend=="down": prediction="PUT"; take_trade=True; reasons.append("Score>=3 and with downtrend ⇒ PUT")
            else: prediction="CALL" if is_bull(prev) else "PUT"; take_trade=True; reasons.append("Score>=3 but sideways ⇒ follow last candle direction")
        return {"prediction":prediction,"take_trade":take_trade,"reasons":reasons,"level":near_lv}
    except Exception as e:
        log_info(f"CWRV failed: {e}"); return {"prediction":"NO_TRADE","take_trade":False,"reasons":[f"Error in logic: {e}"],"level":None}

def quick_backtest(candles, lookback=25):
    try:
        if len(candles) < 10: return 0.0, 0
        start = max(5, len(candles)-lookback); wins=0; taken=0
        for i in range(start, len(candles)-1):
            segment = candles[:i+1]
            if len(segment) < 6: continue
            sig = cwrv_signal(segment)
            if sig["prediction"] in ("CALL","PUT") and sig["take_trade"]:
                taken += 1
                nxt = candles[i]
                if sig["prediction"] == "CALL":
                    if nxt["close"] >= nxt["open"]: wins += 1
                else:
                    if nxt["close"] < nxt["open"]: wins += 1
        acc = pct(wins, taken) if taken>0 else 0.0
        return round(acc,1), taken
    except Exception as e:
        log_info(f"Backtest failed: {e}"); return 0.0, 0

@app.route("/")
def index_route(): return "CWRV-123 Binary Predictor v2. Visit /app"
@app.route("/app")
def app_ui(): return render_template("index.html", pairs=PAIRS, interval=INTERVAL)

@app.route("/analyze", methods=["POST"])
def analyze_route():
    results=[]; errors=[]
    for pair in PAIRS:
        try:
            if state["series"].get(pair): fetch_incremental(pair)
            else: fetch_initial(pair)
        except Exception as e:
            msg = f"Data load failed for {pair}: {e}"
            log_info(msg); traceback.print_exc(); errors.append(msg)
    for pair in PAIRS:
        try:
            candles = state["series"].get(pair, [])
            if not candles:
                results.append({"pair":pair, "error":"No data"}); continue
            sig = cwrv_signal(candles); acc, ntrades = quick_backtest(candles)
            last = candles[-2] if len(candles) >= 2 else candles[-1]
            reasoning = " | ".join(sig.get("reasons", [])) if sig.get("reasons") else "—"
            results.append({"pair":pair, "prediction":sig.get("prediction","NO_TRADE"), "take_trade":bool(sig.get("take_trade", False)), "accuracy":acc, "tested_trades":ntrades, "reasoning":reasoning, "last_candle":{"open": last["open"], "high": last["high"], "low": last["low"], "close": last["close"]}, "level": sig.get("level")})
        except Exception as e:
            msg = f"Analysis failed for {pair}: {e}"
            log_info(msg); traceback.print_exc(); results.append({"pair":pair, "error": msg})
    return {"ok": True, "results": results, "errors": errors, "ts": now_str()}

if __name__ == "__main__":
    log_info("Starting app")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
