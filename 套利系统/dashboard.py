# -*- coding: utf-8 -*-
"""
套利系统 - 可视化管理系统
布局：顶部 Tab 导航、全宽内容区、卡片化分区。
"""

import os
import json
import sys
import subprocess
from pathlib import Path

import streamlit as st
import pandas as pd
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    go = None
    make_subplots = None
    HAS_PLOTLY = False

ARBITRAGE_DIR = Path(__file__).resolve().parent
if str(ARBITRAGE_DIR) not in sys.path:
    sys.path.insert(0, str(ARBITRAGE_DIR))

import config

# 结果目录：优先用与 dashboard 同目录的 results，避免运行目录不同导致找不到图
RESULTS_DIR = ARBITRAGE_DIR / 'results'
if not RESULTS_DIR.exists():
    RESULTS_DIR = Path(config.RESULTS_DIR)
DATA_DIR = Path(config.DATA_DIR)


def load_json_safe(path: Path, default=None):
    if default is None:
        default = {}
    if not path.exists():
        return default
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default


def inject_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;600;700&display=swap');
    :root {
        --bg-main: #1a1d24;
        --bg-panel: #22262e;
        --bg-card: #262b34;
        --bg-card-hover: #2c313b;
        --accent: #6b8cae;
        --accent-dim: #5a7a99;
        --text: #d8dce4;
        --text-2: #a8b0bc;
        --text-3: #7a8494;
        --border: rgba(255, 255, 255, 0.06);
        --positive: #6b9b7a;
        --negative: #b87a7a;
        --radius: 10px;
        --radius-sm: 6px;
        --shadow: 0 2px 12px rgba(0,0,0,0.15);
    }
    html, body, [class*="css"] { font-family: 'Noto Sans SC', sans-serif !important; }
    .stApp { background: var(--bg-main) !important; }
    .main .block-container {
        padding: 1.5rem 2rem 2rem;
        max-width: 100% !important;
        width: 100% !important;
    }
    section.main > div { max-width: 100% !important; width: 100% !important; }
    [data-testid="stAppViewContainer"] main { max-width: 100% !important; }
    [data-testid="stHorizontalBlock"] { width: 100% !important; }
    .main p, .main span, .main label, .main .stMarkdown { color: var(--text) !important; }
    .main .stCaptionContainer { color: var(--text-2) !important; }

    .main .block-container > div:first-of-type .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-panel) !important; border-radius: var(--radius-sm);
        padding: 6px; gap: 4px; border: 1px solid var(--border); margin-bottom: 1.5rem;
    }
    .main .block-container > div:first-of-type .stTabs [data-baseweb="tab"] {
        border-radius: 6px; padding: 10px 20px; font-size: 0.95rem; font-weight: 500;
        background: transparent !important; color: var(--text-2);
    }
    .main .block-container > div:first-of-type .stTabs [aria-selected="true"] {
        background: var(--bg-card) !important; color: var(--accent) !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.15);
    }

    .hero { margin-bottom: 1.5rem; }
    .hero h1 { font-size: 1.5rem; font-weight: 600; color: var(--text); margin: 0 0 0.25rem 0; letter-spacing: -0.02em; }
    .hero p { font-size: 0.875rem; color: var(--text-2); margin: 0; }

    .section { margin-bottom: 1.5rem; }
    .section-title { font-size: 0.8rem; font-weight: 600; color: var(--text-3); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.75rem; }
    .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
    .kpi-card {
        background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius);
        padding: 1.25rem; box-shadow: var(--shadow); transition: border-color 0.2s;
    }
    .kpi-card:hover { border-color: rgba(255,255,255,0.12); }
    .kpi-card.positive { border-left: 4px solid var(--positive); }
    .kpi-card.negative { border-left: 4px solid var(--negative); }
    .kpi-card .kpi-label { font-size: 0.72rem; color: var(--text-3); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem; }
    .kpi-card .kpi-value { font-size: 1.45rem; font-weight: 700; color: var(--text); letter-spacing: -0.02em; }
    .kpi-card.positive .kpi-value { color: var(--positive); }
    .kpi-card.negative .kpi-value { color: var(--negative); }

    .card-wrap {
        background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius);
        padding: 1.25rem; margin-bottom: 1rem; box-shadow: var(--shadow);
    }
    .card-wrap h3 { font-size: 0.95rem; font-weight: 600; color: var(--text); margin: 0 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border); }

    div[data-testid="stMetricValue"] { font-size: 1.3rem !important; font-weight: 700 !important; color: var(--text) !important; }
    div[data-testid="stMetricLabel"] { color: var(--text-2) !important; }
    div[data-testid="stDataFrame"] { border-radius: var(--radius); overflow: hidden; border: 1px solid var(--border); }
    [data-testid="stExpander"] { border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-card); margin-bottom: 0.75rem; }
    .stButton > button {
        border-radius: var(--radius-sm) !important; font-weight: 600 !important;
        background: var(--accent) !important; color: #1a1d24 !important; border: none !important;
        padding: 0.6rem 1.4rem !important; box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .stButton > button:hover { filter: brightness(1.08); }
    [data-testid="stAlert"] { border-radius: var(--radius); border: 1px solid var(--border); background: var(--bg-card); }
    </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title='套利系统',
        page_icon='▣',
        layout='wide',
        initial_sidebar_state='collapsed'
    )
    inject_theme()
    if 'strategy' not in st.session_state:
        st.session_state.strategy = 'martin'

    # 顶部：标题 + 策略选择
    st.markdown('<div class="hero"><h1>套利系统</h1><p>马丁双向 · 套利策略 · 回测与结果</p></div>', unsafe_allow_html=True)
    strat = st.radio(
        '选择策略',
        ['martin', 'arbitrage'],
        format_func=lambda x: '马丁策略' if x == 'martin' else '套利策略',
        horizontal=True,
        key='strategy_radio',
        label_visibility='collapsed',
    )
    st.session_state.strategy = strat
    st.markdown('---')

    tab_overview, tab_backtest, tab_results, tab_config = st.tabs(['概览', '回测', '结果', '配置'])

    with tab_overview:
        render_overview()
    with tab_backtest:
        render_backtest()
    with tab_results:
        render_results()
    with tab_config:
        render_config()


def render_overview():
    martin = load_json_safe(RESULTS_DIR / 'martin_bidirectional_summary.json', {})
    arb = load_json_safe(RESULTS_DIR / 'results_summary.json', {})
    is_martin = st.session_state.get('strategy', 'martin') == 'martin'

    pnl = martin.get('最终盈亏', 0)
    kpi_class = 'positive' if pnl >= 0 else 'negative'
    st.markdown(f'''
    <div class="kpi-grid">
        <div class="kpi-card {kpi_class}"><div class="kpi-label">马丁盈亏 (USDT)</div><div class="kpi-value">{pnl:,.2f}</div></div>
        <div class="kpi-card"><div class="kpi-label">马丁交易次数</div><div class="kpi-value">{martin.get("总交易次数", 0)}</div></div>
        <div class="kpi-card"><div class="kpi-label">套利收益率</div><div class="kpi-value">{arb.get("最终收益率(%)", 0):.2f}%</div></div>
        <div class="kpi-card"><div class="kpi-label">套利交易次数</div><div class="kpi-value">{arb.get("总交易次数", 0)}</div></div>
    </div>''', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])
    with col_left:
        if is_martin:
            st.markdown('<div class="card-wrap"><h3>马丁策略 · 快捷回测</h3></div>', unsafe_allow_html=True)
            last = load_json_safe(RESULTS_DIR / 'martin_bidirectional_summary.json', {}).get('数据配置', {})
            sym = last.get('symbol') or 'BTC-USDT-SWAP'
            start = last.get('start_date') or '2026-02-01'
            end = last.get('end_date') or '2026-02-02'
            with st.form('quick_backtest'):
                q_symbol = st.text_input('交易对', value=sym, key='q_sym')
                q_start = st.text_input('开始日期', value=start, key='q_start')
                q_end = st.text_input('结束日期', value=end, key='q_end')
                if st.form_submit_button('执行马丁回测'):
                    cfg = {
                        'run_mode': 'backtest',
                        'data_config': {
                            'exchange': 'okx', 'symbol': q_symbol.strip() or sym,
                            'interval': '1m', 'start_date': q_start, 'end_date': q_end,
                            'data_dir': config.DATA_DIR, 'binance_futures': False, 'proxies': None,
                        },
                        'strategy_config': _default_strategy_from_last(),
                    }
                    path = RESULTS_DIR / 'run_config.json'
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(cfg, f, ensure_ascii=False, indent=2)
                    run_script(path)
            st.caption('使用默认策略参数，完整参数请在「回测」页设置。')
        else:
            st.markdown('<div class="card-wrap"><h3>套利策略</h3></div>', unsafe_allow_html=True)
            st.info('套利策略回测与逻辑请通过 method/logic.py 或相关脚本执行，结果将写入 results/ 并在「结果」页查看。')
            if arb:
                st.metric('当前套利收益率', f"{arb.get('最终收益率(%)', 0):.2f}%")
                st.metric('套利交易次数', arb.get('总交易次数', 0))

    with col_right:
        st.markdown(f'<div class="card-wrap"><h3>{"马丁" if is_martin else "套利"} · 最近结果摘要</h3></div>', unsafe_allow_html=True)
        if is_martin:
            if martin:
                st.metric('最终盈亏 (USDT)', martin.get('最终盈亏', 0))
                st.metric('总交易次数', martin.get('总交易次数', 0))
                st.caption(f"时间范围: {martin.get('数据时间范围', {}).get('开始', '-')} ~ {martin.get('数据时间范围', {}).get('结束', '-')}")
            else:
                st.info('暂无马丁回测结果，请先执行回测。')
        else:
            if arb:
                st.metric('最终收益率', f"{arb.get('最终收益率(%)', 0):.2f}%")
                st.metric('总交易次数', arb.get('总交易次数', 0))
            else:
                st.info('暂无套利结果摘要。')


def _default_strategy_from_last():
    """上次回测策略参数，缺省时用 config.MARTIN_STRATEGY_CONFIG。"""
    defaults = dict(getattr(config, 'MARTIN_STRATEGY_CONFIG', {}))
    defaults.setdefault('base_price', None)
    last = load_json_safe(RESULTS_DIR / 'martin_bidirectional_summary.json', {}).get('策略参数', {})
    return {**defaults, **last}


def run_script(config_path: Path):
    script_path = ARBITRAGE_DIR / 'method' / 'martin_bidirectional.py'
    if not script_path.exists():
        st.error(f'未找到脚本: {script_path}')
        return
    with st.spinner('回测运行中…'):
        try:
            r = subprocess.run(
                [sys.executable, str(script_path), '--config', str(config_path)],
                cwd=str(ARBITRAGE_DIR), capture_output=True, text=True, timeout=600, encoding='utf-8', errors='replace',
            )
            if r.returncode == 0:
                st.success('回测完成，请到「结果」页查看。')
                with st.expander('运行日志'):
                    st.code(r.stdout[-3500:] if len(r.stdout) > 3500 else r.stdout)
            else:
                st.error('回测失败')
                st.code(r.stderr or r.stdout)
        except subprocess.TimeoutExpired:
            st.error('回测超时（10 分钟）')
        except Exception as e:
            st.error(str(e))


def render_backtest():
    is_martin = st.session_state.get('strategy', 'martin') == 'martin'
    if not is_martin:
        st.markdown('<div class="section-title">套利策略</div>', unsafe_allow_html=True)
        st.info('套利策略回测需在本地运行 **method/logic.py** 或对应脚本，本页仅支持马丁双向策略的一键回测。结果将写入 results/，可在「结果」页选择套利策略查看。')
        return
    st.markdown('<div class="section-title">数据与策略配置 · 执行马丁双向回测</div>', unsafe_allow_html=True)
    last_summary = load_json_safe(RESULTS_DIR / 'martin_bidirectional_summary.json', {})
    # 默认值来自 config，上次回测结果覆盖
    martin_data_defaults = {**getattr(config, 'MARTIN_DATA_CONFIG', {}), 'data_dir': config.DATA_DIR}
    martin_strategy_defaults = dict(getattr(config, 'MARTIN_STRATEGY_CONFIG', {}))
    last_data = {**martin_data_defaults, **(last_summary.get('数据配置') or {})}
    last_strategy = {**martin_strategy_defaults, **(last_summary.get('策略参数') or {})}

    col_data, col_strategy = st.columns([1, 1])

    with col_data:
        st.markdown('<div class="card-wrap"><h3>数据配置</h3></div>', unsafe_allow_html=True)
        exchange = st.selectbox('交易所', ['okx', 'binance'], format_func=lambda x: 'OKX' if x == 'okx' else 'Binance', index=['okx', 'binance'].index(last_data.get('exchange', 'okx')), key='ex')
        default_sym = last_data.get('symbol') or ('BTC-USDT-SWAP' if exchange == 'okx' else 'BTCUSDT')
        symbol = st.text_input('交易对', value=default_sym, placeholder='OKX: BTC-USDT-SWAP', key='sym')
        interval = st.selectbox('K线周期', ['1m', '5m', '15m', '1h', '4h', '1d'], index=['1m', '5m', '15m', '1h', '4h', '1d'].index(last_data.get('interval', '1m')), key='int')
        start_date = st.date_input('开始日期', value=pd.to_datetime(last_data.get('start_date', '2026-02-01')).date() if last_data.get('start_date') else pd.to_datetime('2026-02-01').date(), key='sd')
        end_date = st.date_input('结束日期', value=pd.to_datetime(last_data.get('end_date', '2026-02-02')).date() if last_data.get('end_date') else pd.to_datetime('2026-02-02').date(), key='ed')
        binance_futures = st.checkbox('Binance U 本位合约', value=last_data.get('binance_futures', False), key='bf')

    with col_strategy:
        st.markdown('<div class="card-wrap"><h3>策略参数</h3></div>', unsafe_allow_html=True)
        sub1, sub2, sub3 = st.tabs(['网格', '手数', '止盈止损'])
        with sub1:
            grid_spacing_mode = st.selectbox('网格间距模式', ['pct', 'fixed', 'atr'], format_func=lambda x: {'pct': '百分比', 'fixed': '固定点数', 'atr': 'ATR'}[x], index=['pct', 'fixed', 'atr'].index(last_strategy.get('grid_spacing_mode', 'atr')), key='gsm')
            grid_spacing_pct = st.slider('网格间距 %', 0.1, 5.0, float(last_strategy.get('grid_spacing_pct', 1.0)) * 100, 0.1, key='gsp') / 100.0
            grid_spacing_fixed = st.number_input('固定点数', min_value=10.0, max_value=2000.0, value=float(last_strategy.get('grid_spacing_fixed', 100.0)), step=50.0, key='gsf')
            atr_period = st.slider('ATR 周期', 5, 30, int(last_strategy.get('atr_period', 14)), key='atrp')
            atr_multiplier = st.slider('ATR 倍数', 0.2, 3.0, float(last_strategy.get('atr_multiplier', 1.0)), 0.1, key='atrm')
            dynamic_base = st.checkbox('平仓后动态基准价', value=last_strategy.get('dynamic_base', True), key='db')
        with sub2:
            base_size = st.number_input('初始手数', min_value=0.001, max_value=1.0, value=float(last_strategy.get('base_size', 0.01)), step=0.001, format='%.3f', key='bs')
            multiplier = st.slider('马丁倍数', 1.0, 4.0, float(last_strategy.get('multiplier', 2.0)), 0.1, key='mul')
            normal_levels = st.slider('普通层数', 1, 10, int(last_strategy.get('normal_levels', 3)), key='nl')
            max_martin_levels = st.slider('最大马丁层数', 2, 20, int(last_strategy.get('max_martin_levels', 8)), key='mml')
            max_position_pct = st.slider('最大仓位比例', 0.3, 1.0, float(last_strategy.get('max_position_pct', 0.9)), 0.05, key='mpp')
            total_capital = st.number_input('总资金 (USDT)', min_value=100, value=int(last_strategy.get('total_capital', 10000)), step=500, key='tc')
            fee_rate = st.slider('手续费率 %', 0.0, 0.2, float(last_strategy.get('fee_rate', 0.0005)) * 100, 0.01, key='fr') / 100.0
        with sub3:
            take_profit_pct = st.slider('止盈 %', 0.1, 3.0, float(last_strategy.get('take_profit_pct', 0.005)) * 100, 0.1, key='tpp') / 100.0
            take_profit_mode = st.selectbox('止盈模式', ['unified', 'per_trade', 'layered'], format_func=lambda x: {'unified': '统一回本', 'per_trade': '逐笔', 'layered': '分层'}[x], index=['unified', 'per_trade', 'layered'].index(last_strategy.get('take_profit_mode', 'unified')), key='tpm')
            stop_loss_pct = st.slider('止损 %', 1.0, 30.0, float(last_strategy.get('stop_loss_pct', 0.1)) * 100, 0.5, key='slp') / 100.0

    st.markdown('---')
    if st.button('执行马丁双向回测', type='primary'):
        data_config = {
            'exchange': exchange, 'symbol': symbol.strip() or ('BTC-USDT-SWAP' if exchange == 'okx' else 'BTCUSDT'),
            'interval': interval, 'start_date': start_date.strftime('%Y-%m-%d'), 'end_date': end_date.strftime('%Y-%m-%d'),
            'data_dir': config.DATA_DIR, 'binance_futures': binance_futures, 'proxies': None,
        }
        strategy_config = {
            'base_price': None, 'grid_spacing_mode': grid_spacing_mode, 'grid_spacing_pct': grid_spacing_pct,
            'grid_spacing_fixed': grid_spacing_fixed, 'atr_period': atr_period, 'atr_multiplier': atr_multiplier,
            'base_size': base_size, 'multiplier': multiplier, 'max_martin_levels': max_martin_levels,
            'normal_levels': normal_levels, 'max_position_pct': max_position_pct, 'take_profit_pct': take_profit_pct,
            'stop_loss_pct': stop_loss_pct, 'take_profit_mode': take_profit_mode, 'dynamic_base': dynamic_base,
            'total_capital': total_capital, 'fee_rate': fee_rate,
        }
        config_path = RESULTS_DIR / 'run_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({'run_mode': 'backtest', 'data_config': data_config, 'strategy_config': strategy_config}, f, ensure_ascii=False, indent=2)
        run_script(config_path)
    st.caption('配置写入 results/run_config.json，回测脚本通过 --config 读取。')


def render_results():
    is_martin = st.session_state.get('strategy', 'martin') == 'martin'
    st.markdown(f'<div class="section-title">{"马丁策略" if is_martin else "套利策略"} · 结果</div>', unsafe_allow_html=True)

    if is_martin:
        r1, r2, r3 = st.tabs(['马丁摘要', '交易记录', '图表'])
        with r1:
            _render_martin_summary()
        with r2:
            _render_martin_trades()
        with r3:
            _render_martin_chart()
    else:
        r1, r2 = st.tabs(['套利结果', '图表'])
        with r1:
            _render_arbitrage_summary()
        with r2:
            _render_arbitrage_chart()


def _render_martin_summary():
    st.markdown('<div class="card-wrap"><h3>马丁双向回测摘要</h3></div>', unsafe_allow_html=True)
    summary = load_json_safe(RESULTS_DIR / 'martin_bidirectional_summary.json', {})
    if not summary:
        st.info('暂无马丁回测，请先在「回测」页执行。')
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, (label, key) in zip([c1, c2, c3, c4, c5], [('最终盈亏', '最终盈亏'), ('总交易', '总交易次数'), ('开买', '开买单次数'), ('开卖', '开卖单次数'), ('平仓', '平仓次数')]):
            with col:
                st.metric(label, summary.get(key, 0))
        with st.expander('数据与策略配置'):
            st.json({**summary.get('数据配置', {}), **summary.get('策略参数', {}), '时间范围': summary.get('数据时间范围', {})})


def _render_arbitrage_summary():
    st.markdown('<div class="card-wrap"><h3>套利策略结果</h3></div>', unsafe_allow_html=True)
    summary = load_json_safe(RESULTS_DIR / 'results_summary.json', {})
    if not summary:
        st.info('暂无套利结果摘要。')
    else:
        st.json(summary)


def _render_martin_trades():
    st.markdown('<div class="card-wrap"><h3>交易记录</h3></div>', unsafe_allow_html=True)
    sub1, sub2 = st.tabs(['交易流水', '平仓记录'])
    with sub1:
        trades = load_json_safe(RESULTS_DIR / 'martin_bidirectional_trades.json', [])
        if trades:
            df = pd.DataFrame(trades)
            if 'timestamp' in df.columns:
                df['时间'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info('暂无交易流水。')
    with sub2:
        closes = load_json_safe(RESULTS_DIR / 'martin_bidirectional_closes.json', [])
        if closes:
            df = pd.DataFrame(closes)
            if 'timestamp' in df.columns:
                df['时间'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info('暂无平仓记录。')


def _render_martin_chart():
    st.markdown('<div class="card-wrap"><h3>马丁回测图表</h3></div>', unsafe_allow_html=True)
    chart_data_path = RESULTS_DIR / 'martin_bidirectional_chart_data.json'
    if not chart_data_path.exists():
        st.info('**未找到交互图表数据。** 请到「回测」页点击「执行马丁双向回测」，完成后将生成可缩放、悬停的交互图。')
        _fallback_chart_image()
        return
    chart_data = load_json_safe(chart_data_path, {})
    if not chart_data or not chart_data.get('datetime'):
        st.info('图表数据为空，请先执行马丁双向回测。')
        _fallback_chart_image()
        return

    chart_shown = False
    if HAS_PLOTLY:
        try:
            fig = _build_interactive_chart(chart_data)
        except Exception as e:
            fig = None
            st.warning(f'构建交互图时出错: {e}')
        if fig is not None:
            try:
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key='martin_plotly_chart',
                    config={
                        'scrollZoom': True,
                        'displayModeBar': True,
                        'modeBarButtonsToAdd': ['zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
                        'displaylogo': False,
                    },
                )
                st.caption('支持：拖拽缩放、滚轮缩放、双击还原、悬停查看数值。')
                chart_shown = True
            except Exception as e:
                st.error(f'渲染 Plotly 图表时出错: {e}')
    if not chart_shown:
        html_chart = _build_plotly_js_html(chart_data)
        if html_chart:
            st.components.v1.html(html_chart, height=720, scrolling=False)
            st.caption('支持：拖拽缩放、滚轮缩放、双击还原、悬停查看数值。')
            chart_shown = True
    if not chart_shown:
        st.warning('交互图未显示，请执行 pip install plotly 后刷新。')
        _fallback_chart_image()


def _render_arbitrage_chart():
    st.markdown('<div class="card-wrap"><h3>套利结果图表</h3></div>', unsafe_allow_html=True)
    if (RESULTS_DIR / 'arbitrage_results.png').exists():
        st.image(str(RESULTS_DIR / 'arbitrage_results.png'), use_container_width=True)
    else:
        st.info('暂无套利结果图，请先运行套利策略回测。')


def _fallback_chart_image():
    img_path = RESULTS_DIR / 'martin_bidirectional_results.png'
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
    else:
        st.warning('未找到马丁回测图，请先执行回测。')


def _json_safe_number(x):
    """将 nan/inf 转为可 JSON 序列化的值"""
    if isinstance(x, (int, float)):
        if x != x:  # nan
            return None
        if x == float('inf') or x == float('-inf'):
            return None
    return x


def _build_plotly_js_html(chart_data: dict):
    """不依赖 Python plotly，用 Plotly.js（CDN）在浏览器中渲染交互图"""
    try:
        dt = chart_data['datetime']
        price = chart_data['price']
        base_price = chart_data.get('base_price') or []
        total_pnl = chart_data.get('total_pnl') or []
        buy_level = chart_data.get('buy_level') or []
        sell_level = chart_data.get('sell_level') or []
        buy_count = chart_data.get('buy_count') or []
        sell_count = chart_data.get('sell_count') or []
        trades_buy = chart_data.get('trades_buy') or []
        trades_sell = chart_data.get('trades_sell') or []
        trades_close = chart_data.get('trades_close') or []

        # 确保数值可 JSON 序列化（避免 nan/inf 导致 dumps 失败）
        price = [_json_safe_number(p) for p in price] if isinstance(price, list) else price
        total_pnl = [_json_safe_number(p) for p in total_pnl] if isinstance(total_pnl, list) else total_pnl

        # 构建 Plotly 的 data 数组（与 Plotly.js 兼容）
        data = []
        # 图1: 价格
        data.append({'x': dt, 'y': price, 'type': 'scatter', 'mode': 'lines', 'name': '价格', 'line': {'color': '#94a3b8', 'width': 1.2}, 'xaxis': 'x', 'yaxis': 'y', 'hovertemplate': '%{x}<br>价格: %{y:.2f}<extra></extra>'})
        base_valid = [(dt[i], b) for i, b in enumerate(base_price) if b is not None]
        if base_valid:
            data.append({'x': [x[0] for x in base_valid], 'y': [x[1] for x in base_valid], 'type': 'scatter', 'mode': 'lines', 'name': '基准价', 'line': {'color': '#eab308', 'width': 1.5, 'dash': 'dash'}, 'xaxis': 'x', 'yaxis': 'y'})
        if trades_buy:
            data.append({'x': [t['datetime'] for t in trades_buy], 'y': [t['price'] for t in trades_buy], 'type': 'scatter', 'mode': 'markers', 'name': '开买单', 'marker': {'symbol': 'triangle-up', 'size': 10, 'color': '#34d399'}, 'xaxis': 'x', 'yaxis': 'y'})
        if trades_sell:
            data.append({'x': [t['datetime'] for t in trades_sell], 'y': [t['price'] for t in trades_sell], 'type': 'scatter', 'mode': 'markers', 'name': '开卖单', 'marker': {'symbol': 'triangle-down', 'size': 10, 'color': '#f87171'}, 'xaxis': 'x', 'yaxis': 'y'})
        if trades_close:
            data.append({'x': [t['datetime'] for t in trades_close], 'y': [t['price'] for t in trades_close], 'type': 'scatter', 'mode': 'markers', 'name': '平仓', 'marker': {'symbol': 'x', 'size': 12, 'color': '#38bdf8'}, 'xaxis': 'x', 'yaxis': 'y'})
        # 图2: 累计盈亏
        if total_pnl:
            data.append({'x': dt, 'y': total_pnl, 'type': 'scatter', 'mode': 'lines', 'fill': 'tozeroy', 'name': '累计盈亏', 'line': {'color': '#6b8cae', 'width': 1.5}, 'xaxis': 'x', 'yaxis': 'y2'})
        # 图3: 层级
        if buy_level and sell_level:
            data.append({'x': dt, 'y': buy_level, 'type': 'scatter', 'mode': 'lines', 'fill': 'tozeroy', 'name': '买单层级', 'line': {'color': '#34d399', 'width': 1.2}, 'xaxis': 'x', 'yaxis': 'y3'})
            data.append({'x': dt, 'y': sell_level, 'type': 'scatter', 'mode': 'lines', 'fill': 'tozeroy', 'name': '卖单层级', 'line': {'color': '#f87171', 'width': 1.2}, 'xaxis': 'x', 'yaxis': 'y3'})
        # 图4: 持仓数
        if buy_count and sell_count:
            data.append({'x': dt, 'y': buy_count, 'type': 'scatter', 'mode': 'lines', 'fill': 'tozeroy', 'name': '买单持仓数', 'line': {'color': '#34d399', 'width': 1.2}, 'xaxis': 'x', 'yaxis': 'y4'})
            data.append({'x': dt, 'y': sell_count, 'type': 'scatter', 'mode': 'lines', 'fill': 'tozeroy', 'name': '卖单持仓数', 'line': {'color': '#f87171', 'width': 1.2}, 'xaxis': 'x', 'yaxis': 'y4'})

        layout = {
            'height': 700,
            'margin': {'t': 40, 'b': 40, 'l': 50, 'r': 30},
            'showlegend': True,
            'legend': {'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1},
            'template': 'plotly_dark',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': '#d8dce4', 'size': 11},
            'xaxis': {'domain': [0, 1], 'anchor': 'y4', 'showgrid': True, 'gridcolor': 'rgba(255,255,255,0.06)'},
            'yaxis': {'domain': [0.65, 1], 'title': '价格', 'showgrid': True, 'gridcolor': 'rgba(255,255,255,0.06)'},
            'yaxis2': {'domain': [0.35, 0.65], 'title': '累计盈亏', 'showgrid': True, 'gridcolor': 'rgba(255,255,255,0.06)'},
            'yaxis3': {'domain': [0.15, 0.35], 'title': '层级', 'showgrid': True, 'gridcolor': 'rgba(255,255,255,0.06)'},
            'yaxis4': {'domain': [0, 0.15], 'title': '持仓数', 'showgrid': True, 'gridcolor': 'rgba(255,255,255,0.06)'},
        }
        config_js = {'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False}

        def _json_default(obj):
            if isinstance(obj, float) and (obj != obj or obj == float('inf') or obj == float('-inf')):
                return None
            raise TypeError(type(obj).__name__)

        spec_json = json.dumps({'data': data, 'layout': layout, 'config': config_js}, ensure_ascii=False, default=_json_default)
        # 动态加载 Plotly.js 再绘图，避免内联脚本执行时 CDN 尚未加载
        html = f'''
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"/></head>
<body style="margin:0;background:transparent;">
<div id="chart" style="width:100%;height:700px;"></div>
<script>
(function() {{
  var spec = {spec_json};
  var div = document.getElementById("chart");
  function draw() {{
    if (typeof Plotly !== "undefined") {{
      Plotly.newPlot(div, spec.data, spec.layout, spec.config);
      return;
    }}
    div.innerHTML = "<p style=\"color:#94a3b8;padding:1em;\">Plotly.js 加载中…</p>";
  }}
  var s = document.createElement("script");
  s.src = "https://cdn.plot.ly/plotly-2.27.0.min.js";
  s.onload = function() {{ Plotly.newPlot(div, spec.data, spec.layout, spec.config); }};
  s.onerror = function() {{ div.innerHTML = "<p style=\"color:#b87a7a;padding:1em;\">Plotly.js 加载失败，请安装 Python: pip install plotly</p>"; }};
  document.head.appendChild(s);
}})();
</script>
</body>
</html>'''
        return html
    except Exception as e:
        st.warning(f'生成交互图 HTML 时出错: {e}')
        return None


def _to_plotly_series(arr):
    """将序列转为 Plotly 可用的列表（float/None），避免 numpy 或类型问题"""
    if not arr:
        return arr
    out = []
    for v in arr:
        if v is None or (isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf'))):
            out.append(None)
        elif isinstance(v, (int, float)):
            out.append(float(v))
        else:
            out.append(v)
    return out


def _build_interactive_chart(chart_data: dict):
    """根据 chart_data 构建可缩放、悬停的 Plotly 四子图。出错时抛出异常供调用方显示。"""
    if not HAS_PLOTLY or go is None:
        return None
    dt = chart_data['datetime']
    price = _to_plotly_series(chart_data.get('price') or [])
    base_price = chart_data.get('base_price') or []
    total_pnl = _to_plotly_series(chart_data.get('total_pnl') or [])
    buy_level = _to_plotly_series(chart_data.get('buy_level') or [])
    sell_level = _to_plotly_series(chart_data.get('sell_level') or [])
    buy_count = _to_plotly_series(chart_data.get('buy_count') or [])
    sell_count = _to_plotly_series(chart_data.get('sell_count') or [])
    trades_buy = chart_data.get('trades_buy') or []
    trades_sell = chart_data.get('trades_sell') or []
    trades_close = chart_data.get('trades_close') or []

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=('价格与交易点', '累计盈亏', '持仓层级', '持仓数量'),
        row_heights=[0.35, 0.25, 0.2, 0.2],
    )
    fig.add_trace(go.Scatter(x=dt, y=price, name='价格', line=dict(color='#94a3b8', width=1.2), hovertemplate='%{x}<br>价格: %{y:.2f}<extra></extra>'), row=1, col=1)
    base_valid = [(dt[i], b) for i, b in enumerate(base_price) if b is not None]
    if base_valid:
        fig.add_trace(go.Scatter(x=[x[0] for x in base_valid], y=[x[1] for x in base_valid], name='基准价', line=dict(color='#eab308', width=1.5, dash='dash'), hovertemplate='%{x}<br>基准价: %{y:.2f}<extra></extra>'), row=1, col=1)
    if trades_buy:
        fig.add_trace(go.Scatter(x=[t['datetime'] for t in trades_buy], y=[t['price'] for t in trades_buy], name='开买单', mode='markers', marker=dict(symbol='triangle-up', size=10, color='#34d399'), hovertemplate='开买单<br>%{x}<br>价格: %{y:.2f}<extra></extra>'), row=1, col=1)
    if trades_sell:
        fig.add_trace(go.Scatter(x=[t['datetime'] for t in trades_sell], y=[t['price'] for t in trades_sell], name='开卖单', mode='markers', marker=dict(symbol='triangle-down', size=10, color='#f87171'), hovertemplate='开卖单<br>%{x}<br>价格: %{y:.2f}<extra></extra>'), row=1, col=1)
    if trades_close:
        fig.add_trace(go.Scatter(x=[t['datetime'] for t in trades_close], y=[t['price'] for t in trades_close], name='平仓', mode='markers', marker=dict(symbol='x', size=12, color='#38bdf8'), hovertemplate='平仓<br>%{x}<br>价格: %{y:.2f}<extra></extra>'), row=1, col=1)
    if total_pnl:
        fig.add_trace(go.Scatter(x=dt, y=total_pnl, name='累计盈亏', fill='tozeroy', line=dict(color='#6b8cae', width=1.5), hovertemplate='%{x}<br>累计盈亏: %{y:.2f} USDT<extra></extra>'), row=2, col=1)
    if buy_level and sell_level:
        fig.add_trace(go.Scatter(x=dt, y=buy_level, name='买单层级', line=dict(color='#34d399', width=1.2), fill='tozeroy', hovertemplate='%{x}<br>买层: %{y}<extra></extra>'), row=3, col=1)
        fig.add_trace(go.Scatter(x=dt, y=sell_level, name='卖单层级', line=dict(color='#f87171', width=1.2), fill='tozeroy', hovertemplate='%{x}<br>卖层: %{y}<extra></extra>'), row=3, col=1)
    if buy_count and sell_count:
        fig.add_trace(go.Scatter(x=dt, y=buy_count, name='买单持仓数', line=dict(color='#34d399', width=1.2), fill='tozeroy', hovertemplate='%{x}<br>买数: %{y}<extra></extra>'), row=4, col=1)
        fig.add_trace(go.Scatter(x=dt, y=sell_count, name='卖单持仓数', line=dict(color='#f87171', width=1.2), fill='tozeroy', hovertemplate='%{x}<br>卖数: %{y}<extra></extra>'), row=4, col=1)

    fig.update_layout(
        height=700,
        margin=dict(t=40, b=40, l=50, r=30),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#d8dce4', size=11),
        xaxis4=dict(title='时间'),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.06)', zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.06)', zeroline=False)
    return fig


def render_config():
    st.markdown('<div class="section-title">config.py 只读配置（统一参数入口）</div>', unsafe_allow_html=True)
    with st.expander('路径'):
        st.json({'ARBITRAGE_DIR': config.ARBITRAGE_DIR, 'DATA_DIR': config.DATA_DIR, 'RESULTS_DIR': config.RESULTS_DIR})
    with st.expander('马丁策略 · 数据配置 MARTIN_DATA_CONFIG'):
        st.json(getattr(config, 'MARTIN_DATA_CONFIG', {}))
    with st.expander('马丁策略 · 策略参数 MARTIN_STRATEGY_CONFIG'):
        st.json(getattr(config, 'MARTIN_STRATEGY_CONFIG', {}))
    with st.expander('马丁策略 · 运行模式（回测/实盘）'):
        st.json({
            'RUN_MODE': getattr(config, 'RUN_MODE', 'backtest'),
            'POLL_INTERVAL_SECONDS': getattr(config, 'POLL_INTERVAL_SECONDS', 60),
            'MARTIN_PAPER_TRADE': getattr(config, 'MARTIN_PAPER_TRADE', True),
        })
    with st.expander('套利策略 · 参数 ARBITRAGE_CONFIG'):
        st.json(config.ARBITRAGE_CONFIG)
    with st.expander('交易配置 TRADING_CONFIG'):
        st.json(config.TRADING_CONFIG)
    with st.expander('交易所 API（OKX / Binance）'):
        st.json({'OKX': {'sandbox': config.OKX_CONFIG.get('sandbox'), 'api_key_set': bool(config.OKX_CONFIG.get('api_key'))}, 'BINANCE': {'testnet': config.BINANCE_CONFIG.get('testnet'), 'api_key_set': bool(config.BINANCE_CONFIG.get('api_key'))}})
    with st.expander('风险控制 RISK_CONFIG'):
        st.json(config.RISK_CONFIG)
    with st.expander('日志 LOG_CONFIG'):
        st.json(config.LOG_CONFIG)


if __name__ == '__main__':
    main()
