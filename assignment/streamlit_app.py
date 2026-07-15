"""TradeTutor — 전체 플랫폼 Streamlit 이식판.

FastAPI 웹 플랫폼(실시간 차트 · 전략 분석 · 연습 게임 · 학습 · 멀티에이전트 채팅)을
하나의 Streamlit 앱으로 옮겼다. 차트는 벤더링한 lightweight-charts 를 components.html 로
임베드해 그대로 쓰고(Binance WS 로 실시간), 나머지는 Streamlit 위젯 + 백엔드 함수 직접 호출.

실행: uv run streamlit run streamlit_app.py
"""
import json
import uuid
from pathlib import Path

import dotenv
dotenv.load_dotenv()

import streamlit as st
import streamlit.components.v1 as components

from backend.data import fetch_klines, TOP_COINS
from backend.strategies import (
    STRATEGY_LIBRARY, compute_timing, indicator_overlays, signal_markers,
    overlay_lines, pane_indicators, AVAILABLE_INDICATORS, default_indicators,
)
from backend import research as research_mod
from backend.agent import learn as agent_learn, trade_feedback
from backend.supervisor import run as agent_run
from backend.game import new_game, simulate

st.set_page_config(page_title="TradeTutor", page_icon="📈", layout="wide")
HISTORY = 1500


# ============================ 차트 (lightweight-charts 임베드) ============================
@st.cache_data
def _lwc_js():
    return (Path(__file__).parent / "static/vendor/lightweight-charts.js").read_text()


@st.cache_data(ttl=60)
def get_candles(symbol, interval, limit=HISTORY):
    raw = fetch_klines(symbol, interval, limit)
    return [{"time": int(k[0]) // 1000, "open": float(k[1]), "high": float(k[2]),
             "low": float(k[3]), "close": float(k[4])} for k in raw]


def candles_to_df(candles):
    import pandas as pd
    df = pd.DataFrame(candles)
    df["date"] = pd.to_datetime(df["time"], unit="s")
    return df.set_index("date")[["open", "high", "low", "close"]]


_CHART_TMPL = """
<div id="wrap" style="position:relative;height:__H__px;">
  <div id="c" style="height:__H__px;"></div>
  <div id="tb" style="position:absolute;top:6px;left:6px;z-index:6;display:flex;gap:4px;"></div>
</div>
<script>__LWC__</script>
<script>
const cfg = __CFG__;
const chart = LightweightCharts.createChart(document.getElementById('c'), {
  layout:{background:{color:'#0e1117'},textColor:'#8b949e'},
  grid:{vertLines:{color:'#1c232d'},horzLines:{color:'#1c232d'}},
  rightPriceScale:{borderColor:'#2a313c'},
  timeScale:{borderColor:'#2a313c',timeVisible:true,secondsVisible:false},
  height:__H__
});
const s = chart.addCandlestickSeries({upColor:'#26a69a',downColor:'#ef5350',borderVisible:false,wickUpColor:'#26a69a',wickDownColor:'#ef5350'});
s.setData(cfg.candles);
for (const o of cfg.overlays){ const l=chart.addLineSeries({color:o.color,lineWidth:1.5,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false}); l.setData(o.data); }
if (cfg.markers && cfg.markers.length) s.setMarkers(cfg.markers);
for (const p of cfg.priceLines){ s.createPriceLine({price:p.price,color:p.color,lineWidth:(p.width||2),lineStyle:(p.style===undefined?2:p.style),axisLabelVisible:true,title:p.title}); }
const panes = cfg.panes || [];
if (panes.length){
  chart.priceScale('right').applyOptions({scaleMargins:{top:0.05, bottom:0.06 + 0.28*panes.length}});
  panes.forEach((p, i)=>{
    p.lines.forEach(ln=>{ const ps=chart.addLineSeries({priceScaleId:p.id,color:ln.color,lineWidth:1.2,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false}); ps.setData(ln.data); });
    chart.priceScale(p.id).applyOptions({scaleMargins:{top:1-0.28*(panes.length-i)+0.04, bottom:0.28*(panes.length-i-1)+0.02}});
  });
}
chart.timeScale().fitContent();
if (cfg.live){
  const ws=new WebSocket(`wss://stream.binance.com:9443/ws/${cfg.live.symbol.toLowerCase()}@kline_${cfg.live.interval}`);
  ws.onmessage=(e)=>{const k=JSON.parse(e.data).k; s.update({time:Math.floor(k.t/1000),open:+k.o,high:+k.h,low:+k.l,close:+k.c});};
  window.addEventListener('beforeunload',()=>ws.close());
}
// ---------- 작도 도구 (추세선 / 수평선 / 지우기) — 트레이딩뷰식 ----------
const tb=document.getElementById('tb');
const bs='background:#161b22ee;color:#e6edf3;border:1px solid #2a313c;border-radius:5px;padding:3px 8px;font-size:11px;cursor:pointer;';
let mode='none', pending=null; const drawnS=[], drawnL=[];
[['none','✋ 커서'],['trend','╱ 추세선'],['hline','— 수평선'],['clear','🗑 지우기']].forEach(([m,label])=>{
  const b=document.createElement('button'); b.textContent=label; b.style.cssText=bs; b.dataset.m=m;
  b.onclick=()=>{
    if(m==='clear'){ drawnS.forEach(x=>chart.removeSeries(x)); drawnS.length=0; drawnL.forEach(x=>s.removePriceLine(x)); drawnL.length=0; pending=null; }
    else { mode=m; pending=null; } paint();
  };
  tb.appendChild(b);
});
function paint(){ [...tb.children].forEach(b=> b.style.outline=(b.dataset.m===mode&&mode!=='none')?'2px solid #3b82f6':'none');
  chart.applyOptions({handleScroll:mode==='none', handleScale:mode==='none'}); }
chart.subscribeClick(param=>{
  if(mode==='none'||mode==='clear'||!param.point) return;
  const price=s.coordinateToPrice(param.point.y);
  const time=chart.timeScale().coordinateToTime(param.point.x);
  if(price==null||time==null) return;
  if(mode==='hline'){ drawnL.push(s.createPriceLine({price:price,color:'#eab308',lineWidth:1,lineStyle:0,axisLabelVisible:true,title:(+price).toFixed(2)})); }
  else if(mode==='trend'){
    if(!pending){ pending={time:time,value:price}; }
    else{ let pts=[pending,{time:time,value:price}].sort((a,b)=>a.time-b.time);
      if(pts[0].time!==pts[1].time){ const ls=chart.addLineSeries({color:'#eab308',lineWidth:2,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false}); ls.setData(pts); drawnS.push(ls); }
      pending=null; }
  }
});
paint();
</script>
"""


def render_chart(candles, overlays=None, markers=None, price_lines=None, live=None, panes=None, height=480):
    cfg = {"candles": candles, "overlays": overlays or [], "markers": markers or [],
           "priceLines": price_lines or [], "live": live, "panes": panes or []}
    html = (_CHART_TMPL.replace("__H__", str(height))
            .replace("__CFG__", json.dumps(cfg))
            .replace("__LWC__", _lwc_js()))
    components.html(html, height=height + 12)


def _lines(t):
    return [{"price": t["entry"], "color": "#c9d1d9", "title": "진입"},
            {"price": t["take_profit"], "color": "#26a69a", "title": "익절"},
            {"price": t["stop_loss"], "color": "#ef5350", "title": "손절"}]


# ============================ 헤더 ============================
st.title("📈 TradeTutor")
st.caption("실시간 차트로 매매법을 배우는 학습 플랫폼 · ⚠️ 학습용 (투자 조언 아님)")

STRATS = [(k, v["name"]) for k, v in STRATEGY_LIBRARY.items()]
tab_chart, tab_game, tab_learn, tab_research, tab_chat = st.tabs(
    ["📈 차트 · 분석", "🎮 연습 게임", "📚 전략 학습", "🔬 전략 리서치", "💬 멀티에이전트 채팅"])


# ---------------------------- 탭 1: 차트 · 분석 ----------------------------
with tab_chart:
    c1, c2, c3 = st.columns([1, 1, 2])
    symbol = c1.selectbox("코인", TOP_COINS, key="ch_sym")
    interval = c2.selectbox("타임프레임", ["15m", "1h", "4h", "1d"], index=1, key="ch_iv")
    skey = c3.selectbox("전략", [k for k, _ in STRATS],
                        format_func=lambda k: STRATEGY_LIBRARY[k]["name"], key="ch_strat")

    inds = st.multiselect("📊 차트에 표시할 지표", AVAILABLE_INDICATORS,
                          default=default_indicators(skey), key="ch_inds")
    analyze = st.checkbox("전략 분석 표시 (매매 신호 마커 + 진입/손절/익절)", key="ch_an")

    candles = get_candles(symbol, interval)
    df = candles_to_df(candles)
    overlays = overlay_lines(df, inds)
    panes = pane_indicators(df, inds)
    markers = price_lines = None
    if analyze:
        t = compute_timing(df, skey)
        markers = signal_markers(df, skey)
        price_lines = _lines(t)

    render_chart(candles, overlays, markers, price_lines,
                 live={"symbol": symbol, "interval": interval}, panes=panes, height=460)

    if analyze:
        st.info(f"**{t['strategy_name']} · 현재 {t['signal']}**")
        st.markdown(
            f"- **진입 근거** {t['entry_reason']}\n"
            f"- **진입** {t['entry']:,.2f} · **손절** {t['stop_loss']:,.2f} ({t['sl_pct']:+.1f}%) · "
            f"**익절** {t['take_profit']:,.2f} ({t['tp_pct']:+.1f}%) · **손익비** {t['rr']:.2f}:1\n"
            f"- **손절 기준** {t['sl_method']}\n- **익절 기준** {t['tp_method']}\n"
            f"- ▲매수/▼매도 마커 = 이 전략이 과거에 신호를 낸 지점 · 지지 {t['support']:,.0f} / 저항 {t['resistance']:,.0f}")


# ---------------------------- 탭 2: 연습 게임 ----------------------------
with tab_game:
    st.markdown("과거 랜덤 시점의 차트가 나옵니다. 진입가는 그 시점 종가. "
                "**지표를 켜서 차트를 읽고, 익절·손절을 직접 정한 뒤** 결과를 확인하세요. 막히면 💡힌트.")
    gc1, gc2 = st.columns([2, 1])
    g_strat = gc1.selectbox("연습할 전략", [k for k, _ in STRATS],
                            format_func=lambda k: STRATEGY_LIBRARY[k]["name"], key="g_strat")
    if gc2.button("🎮 새 게임 (랜덤 과거 시점)"):
        gnew = new_game(st.session_state.get("ch_sym", "BTCUSDT"), "4h", g_strat)
        st.session_state.game = gnew
        st.session_state.pop("game_result", None)
        # 익절·손절은 '사람이 직접' — 전략 제안이 아니라 중립(±3%)에서 시작
        st.session_state.g_tp = round(gnew["entry_price"] * 1.03, 2)
        st.session_state.g_sl = round(gnew["entry_price"] * 0.97, 2)
        st.session_state.g_inds = default_indicators(g_strat)   # 전략에 맞는 지표 미리 켜줌
        st.session_state.g_hint = False

    g = st.session_state.get("game")
    if g:
        st.caption(f"전략: **{g['strategy_name']}** · 진입 시점 종가 기준. 익절·손절은 당신이 정합니다.")
        gi1, gi2 = st.columns([3, 1])
        inds = gi1.multiselect("📊 차트에 표시할 지표 (이평선 등)", AVAILABLE_INDICATORS, key="g_inds")
        hint = gi2.checkbox("💡 힌트 보기", key="g_hint")

        stp = max(0.01, round(g["entry_price"] * 0.001, 2))
        i1, i2, i3 = st.columns(3)
        i1.metric("진입가", f"{g['entry_price']:,.2f}")
        tp = i2.number_input("익절가 (직접 입력)", step=stp, format="%.2f", key="g_tp")
        sl = i3.number_input("손절가 (직접 입력)", step=stp, format="%.2f", key="g_sl")
        valid = tp > g["entry_price"] and sl < g["entry_price"]
        rr = (tp - g["entry_price"]) / (g["entry_price"] - sl) if valid else 0
        st.caption(f"익절 {(tp/g['entry_price']-1)*100:+.1f}% · 손절 {(sl/g['entry_price']-1)*100:+.1f}% · 손익비 **{rr:.2f}:1**"
                   + ("  ⚠️ 손익비 1 미만" if 0 < rr < 1 else "")
                   + ("" if valid else "  ⚠️ 익절가는 진입가보다 높게, 손절가는 낮게"))

        if hint:
            st.info(f"💡 **힌트 — {g['strategy_name']}**\n\n"
                    f"- 진입 시점 신호: {g['signal']}\n"
                    f"- 진입 근거: {g['entry_reason']}\n"
                    f"- 전략 제안 손절: {g['suggest_sl']:,.2f} — {g['sl_method']}\n"
                    f"- 전략 제안 익절: {g['suggest_tp']:,.2f} — {g['tp_method']}\n\n"
                    f"위 근거와 지표선을 보고 익절·손절을 직접 조정해보세요. (차트의 노란 점선 = 전략 제안)")

        res = st.session_state.get("game_result")
        cand = g["candles"] + (res["future"] if res else [])
        gdf = candles_to_df(cand)
        overlays = overlay_lines(gdf, inds)
        gpanes = pane_indicators(gdf, inds)
        plines = [{"price": g["entry_price"], "color": "#c9d1d9", "title": "진입"}]
        if tp > g["entry_price"]:
            plines.append({"price": tp, "color": "#26a69a", "title": "익절"})
        if sl < g["entry_price"]:
            plines.append({"price": sl, "color": "#ef5350", "title": "손절"})
        if hint:
            plines.append({"price": g["suggest_tp"], "color": "#d29922", "title": "힌트 익절", "style": 1, "width": 1})
            plines.append({"price": g["suggest_sl"], "color": "#d29922", "title": "힌트 손절", "style": 1, "width": 1})
        mk = None
        if res:
            mk = [{"time": res["exit_time"], "position": "aboveBar" if res["pnl_pct"] >= 0 else "belowBar",
                   "color": "#26a69a" if res["pnl_pct"] >= 0 else "#ef5350", "shape": "circle", "text": res["outcome_kr"]}]
        render_chart(cand, overlays=overlays, price_lines=plines, markers=mk, panes=gpanes, height=420)

        if st.button("▶ 결과 확인 (미래 재생)"):
            if not valid:
                st.error("익절가는 진입가보다 높게, 손절가는 낮게 설정하세요.")
            else:
                r = simulate(g["symbol"], g["interval"], g["entry_time"], g["entry_price"], tp, sl)
                r["outcome_kr"] = {"tp": "익절 🎉", "sl": "손절 😢", "timeout": "미체결(시간 초과)"}[r["outcome"]]
                r["feedback"] = trade_feedback({
                    "entry": g["entry_price"], "tp": tp, "sl": sl,
                    "tp_pct": (tp/g["entry_price"]-1)*100, "sl_pct": (sl/g["entry_price"]-1)*100,
                    "rr": r["rr"], "outcome_kr": r["outcome_kr"], "pnl": r["pnl_pct"], "bh": r["bh_pct"]})
                st.session_state.game_result = r
                sc = st.session_state.setdefault("score", {"n": 0, "wins": 0, "pnl": 0.0})
                sc["n"] += 1; sc["wins"] += 1 if r["pnl_pct"] >= 0 else 0; sc["pnl"] += r["pnl_pct"]
                st.rerun()

        if res:
            good = res["pnl_pct"] >= 0
            (st.success if good else st.error)(f"**{res['outcome_kr']} · 수익률 {res['pnl_pct']:+.2f}%** · "
                f"청산가 {res['exit_price']:,.2f} · 끝까지 보유 시 {res['bh_pct']:+.2f}% · 손익비 {res['rr']:.2f}:1")
            st.markdown(f"**🎓 코치 피드백**\n\n{res['feedback']}")
        sc = st.session_state.get("score")
        if sc and sc["n"]:
            st.caption(f"📊 전적: {sc['n']}판 · 승률 {round(sc['wins']/sc['n']*100)}% · 누적 수익률 {sc['pnl']:+.1f}%")


# ---------------------------- 탭 3: 전략 학습 ----------------------------
with tab_learn:
    lc1, lc2 = st.columns([2, 1])
    l_strat = lc1.selectbox("배울 전략", [k for k, _ in STRATS],
                            format_func=lambda k: STRATEGY_LIBRARY[k]["name"], key="l_strat")
    if lc2.button("학습하기"):
        with st.spinner("규칙 정리 + 백테스트(약 4년) 중…"):
            st.session_state.learn = agent_learn(l_strat, st.session_state.get("ch_sym", "BTCUSDT"))
    lr = st.session_state.get("learn")
    if lr:
        st.subheader(lr["strategy_name"])
        st.markdown(lr["rules"])
        b = lr["backtest"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("전략 수익률", f"{b['total_return']*100:.1f}%")
        m2.metric("매수 후 보유", f"{b['bh_return']*100:.1f}%")
        m3.metric("최대낙폭(MDD)", f"{b['mdd']*100:.1f}%")
        m4.metric("승률 / 매매", f"{b['win_rate']*100:.0f}% / {b['n_trades']}회")
        st.markdown(f"**🎓 학습 코칭**\n\n{lr['coaching']}")


# ---------------------------- 탭 4: 전략 리서치 ----------------------------
with tab_research:
    st.markdown("**AI가 새 전략을 발굴 → 실데이터 백테스트 → 유효/폐기 판정.** "
                "유효하면 **편입**해서 학습·게임·차트에 추가하고, 불필요하면 **폐기**하세요. 결정 내역도 남습니다.")
    rc1, rc2 = st.columns([3, 1])
    theme = rc1.text_input("리서치 방향 (선택)", placeholder="예: 변동성 돌파 / 모멘텀 / 평균회귀 …", key="r_theme")
    if rc2.button("🔬 새 전략 리서치"):
        with st.spinner("전략 발굴 → 백테스트(5코인·4년) → 판정 중…"):
            cfg = research_mod.research_strategy(theme)
            ev = research_mod.evaluate(cfg)
            if "error" in ev:
                st.session_state.research = {"cfg": cfg, "error": ev["error"]}
            else:
                st.session_state.research = {"cfg": cfg, "ev": ev, "judge": research_mod.judge(cfg, ev)}

    rr = st.session_state.get("research")
    if rr:
        cfg = rr["cfg"]
        st.subheader(f"🧪 {cfg['name']}")
        st.caption(cfg["desc"])
        e, x = cfg["entry"], cfg["exit"]
        st.markdown(f"- **진입**: `{e['a']} {e['op']} {e['b']}`\n- **청산**: `{x['a']} {x['op']} {x['b']}`\n- **지표**: {', '.join(cfg.get('indicators', []))}")
        if rr.get("error"):
            st.error(rr["error"])
        else:
            s = rr["ev"]["summary"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("평균 수익률", f"{s['ret']*100:.0f}%")
            m2.metric("수익/MDD", f"{s['rpm']:.2f}")
            m3.metric("승률", f"{s['win']*100:.0f}%")
            m4.metric("B&H 이긴 코인", f"{s['beats']}/5")
            j = rr["judge"]
            (st.success if j["valid"] else st.warning)(f"AI 판정: **{j['verdict']}**")
            st.markdown(j["text"])
            b1, b2 = st.columns(2)
            if j["valid"]:
                if b1.button("✅ 편입하기 (학습에 추가)"):
                    research_mod.adopt(cfg, rr["ev"])
                    st.session_state.pop("research")
                    st.success(f"'{cfg['name']}' 편입 완료! 학습·게임·차트 전략에 추가됐어요.")
                    st.rerun()
            if b2.button("❌ 폐기"):
                research_mod.discard(cfg, rr["ev"])
                st.session_state.pop("research")
                st.info("폐기 처리했습니다. (내역에 기록됨)")
                st.rerun()

    st.divider()
    st.markdown("#### 📜 리서치 내역")
    log = research_mod.load_log()
    if not log:
        st.caption("아직 없습니다. 위에서 리서치를 실행해보세요.")
    else:
        import pandas as pd
        st.dataframe(pd.DataFrame([{
            "시각": e["time"], "결정": e["decision"], "전략": e["name"],
            "수익률": f"{e['ret']*100:.0f}%", "수익/MDD": e["rpm"],
            "승률": f"{e['win']*100:.0f}%", "매매": e["trades"],
        } for e in log]), use_container_width=True, hide_index=True)


# ---------------------------- 탭 5: 멀티에이전트 채팅 ----------------------------
with tab_chat:
    st.caption("Supervisor 가 🎓Tutor · 🔍Researcher · ❓Quiz 중 알맞은 전문가에게 라우팅합니다.")
    AGENT_KR = {"tutor": "🎓 튜터", "researcher": "🔍 리서처", "quiz": "❓ 퀴즈"}
    st.session_state.setdefault("tid", uuid.uuid4().hex)
    st.session_state.setdefault("msgs", [])
    for m in st.session_state.msgs:
        with st.chat_message(m["role"]):
            if m.get("agent"):
                st.caption(f"{AGENT_KR.get(m['agent'], '담당')} 응답")
            st.write(m["content"])
    if prompt := st.chat_input("예: 골든크로스 설명해줘 / 지금 BTC 매수 타이밍이야? / 퀴즈 내줘"):
        st.session_state.msgs.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("전문가에게 연결 중…"):
                r = agent_run(prompt, st.session_state.tid)
            if r["agent"]:
                st.caption(f"{AGENT_KR.get(r['agent'], '담당')} 응답")
            st.write(r["reply"])
        st.session_state.msgs.append({"role": "assistant", "content": r["reply"], "agent": r["agent"]})


with st.sidebar:
    st.header("📈 TradeTutor")
    st.markdown(
        "**한 앱에 전부 (Streamlit 이식)**\n\n"
        "- 📈 실시간 차트 + 전략 분석\n"
        "- 🎮 연습 게임 (과거로 매매 연습)\n"
        "- 📚 전략 학습 (규칙 + 백테스트)\n"
        "- 💬 멀티에이전트 채팅 (Supervisor)")
    st.divider()
    st.caption("차트: lightweight-charts + Binance WS(실시간). 전략 5종 백테스트로 검증·조정됨.")
