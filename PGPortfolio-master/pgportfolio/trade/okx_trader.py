from __future__ import absolute_import, division, print_function

import logging
import math
import os
import time
from typing import Dict

import numpy as np

from pgportfolio.marketdata.okx import Okx
from pgportfolio.tools.data import get_type_list
from pgportfolio.trade.okx_client import OkxRESTClient
from pgportfolio.trade.trader import Trader


class OkxTrader(Trader):
    """
    Live trading adapter that rebases the NN agent decisions into OKX spot orders.
    """

    def __init__(self, config, net_dir=None, agent=None, total_steps=None, device="cpu"):
        trading_config = config.get("trading", {})
        waiting_period = int(config["input"]["global_period"])
        derived_steps = total_steps
        if derived_steps is None:
            live_steps = int(trading_config.get("live_steps", 0) or 0)
            derived_steps = live_steps if live_steps > 0 else None

        quote_currency = trading_config.get("quote_currency", "BTC").upper()
        if quote_currency != "BTC":
            raise ValueError("OkxTrader currently supports BTC quote currency only.")

        super(OkxTrader, self).__init__(
            waiting_period=waiting_period,
            config=config,
            total_steps=derived_steps,
            net_dir=net_dir,
            agent=agent,
            agent_type="nn",
            device=device,
        )

        self._quote_currency = quote_currency
        self._feature_list = get_type_list(config["input"]["feature_number"])
        self._public_client = Okx()
        self._history_retries = 3
        self._min_notional = float(trading_config.get("min_order_btc", 5e-4))
        self._symbol_overrides = trading_config.get("symbol_overrides", {}) or {}
        self._paper_trading = bool(trading_config.get("paper_trading", True))
        self._market_prices: Dict[str, float] = {}
        self._pair_map = self._build_pair_map()
        self._rest_client = self._build_rest_client(trading_config)
        self._td_mode = trading_config.get("td_mode", "cash")

    # ------------------------------------------------------------------ #
    # Trader overrides
    # ------------------------------------------------------------------ #
    def generate_history_matrix(self):
        aligned_now = self._align_timestamp(int(time.time()))
        start = aligned_now - self._period * (self._window_size + 1)
        history = np.zeros(
            (len(self._feature_list), self._coin_number, self._window_size),
            dtype=np.float32,
        )

        for coin_idx, coin in enumerate(self._coin_name_list):
            pair = self._pair_map[coin]["pair"]
            inverted = coin.startswith("reversed_")
            candles = self._fetch_candles(pair, start, aligned_now)
            if len(candles) < self._window_size:
                raise RuntimeError(
                    "Not enough candles for {} (need {}, got {})".format(
                        pair, self._window_size, len(candles)
                    )
                )
            recent = candles[-self._window_size :]
            for t, candle in enumerate(recent):
                for feat_idx, feature in enumerate(self._feature_list):
                    value = float(candle[feature])
                    if inverted:
                        value = self._safe_inverse(value)
                    history[feat_idx, coin_idx, t] = value
            raw_close = float(recent[-1]["close"])
            self._market_prices[coin] = raw_close if not inverted else self._safe_inverse(history[0, coin_idx, -1])

        if np.isnan(history).any():
            raise ValueError("NaN detected in history matrix.")
        return history

    def trade_by_strategy(self, omega):
        if not self._market_prices:
            logging.warning("Market prices unavailable, skipping trade step.")
            return

        balances = self._fetch_balances()
        holdings = self._build_holdings(balances)
        total_capital = self._compute_total_capital(holdings)
        if total_capital <= 0:
            logging.warning("Total capital %.8f too small, skipping trade.", total_capital)
            return
        self._total_capital = total_capital

        orders = self._build_rebalance_orders(omega, holdings, total_capital)
        if not orders:
            logging.info("Portfolio already aligned with target weights.")
            self._update_asset_vector_from_holdings(holdings, total_capital)
            return

        for order in orders:
            self._execute_order(order)

        refreshed_balances = self._fetch_balances()
        if refreshed_balances:
            holdings = self._build_holdings(refreshed_balances)
            self._total_capital = self._compute_total_capital(holdings)
        self._update_asset_vector_from_holdings(holdings, self._total_capital)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _build_rest_client(self, trading_config):
        provider = (trading_config.get("provider") or "").lower()
        if provider not in ("okx", "okex"):
            logging.warning("Unsupported provider '%s'; running in paper mode.", provider)
            self._paper_trading = True
            return None

        api_key = (trading_config.get("api_key") or "").strip() or os.getenv(
            trading_config.get("api_key_env", "OKX_API_KEY")
        )
        secret = (trading_config.get("api_secret") or "").strip() or os.getenv(
            trading_config.get("api_secret_env", "OKX_API_SECRET")
        )
        passphrase = (trading_config.get("passphrase") or "").strip() or os.getenv(
            trading_config.get("passphrase_env", "OKX_PASSPHRASE")
        )
        base_url = (trading_config.get("base_url") or "").strip() or "https://www.okx.com"
        timeout = float(trading_config.get("timeout", 10))
        use_simulated = bool(trading_config.get("use_simulated", True))

        if not (api_key and secret and passphrase):
            logging.warning(
                "Missing OKX credentials (key=%s, secret=%s, passphrase=%s). "
                "Falling back to paper trading.",
                bool(api_key),
                bool(secret),
                bool(passphrase),
            )
            self._paper_trading = True
            return None

        try:
            client = OkxRESTClient(
                api_key=api_key,
                secret_key=secret,
                passphrase=passphrase,
                base_url=base_url,
                use_simulated=use_simulated,
                timeout=timeout,
            )
            logging.info(
                "Initialized OKX REST client (simulated=%s, base_url=%s, timeout=%ss).",
                use_simulated,
                base_url,
                timeout,
            )
            return client
        except Exception as exc:
            logging.error("Failed to initialize OKX REST client: %s", exc)
            self._paper_trading = True
            return None

    def _build_pair_map(self):
        mapping = {}
        for coin in self._coin_name_list:
            override = self._symbol_overrides.get(coin)
            if override:
                inst_id = override
                pair = self._inst_to_pair(inst_id)
            else:
                pair = self._coin_to_pair(coin)
                inst_id = Okx._pair_to_inst_id(pair)
            mapping[coin] = {"pair": pair, "inst": inst_id}
        return mapping

    def _coin_to_pair(self, coin_name):
        base = coin_name.replace("reversed_", "").upper()
        if coin_name.startswith("reversed_"):
            return "{}_{}".format(base, self._quote_currency)
        return "{}_{}".format(self._quote_currency, base)

    @staticmethod
    def _inst_to_pair(inst_id):
        parts = inst_id.split("-")
        if len(parts) != 2:
            raise ValueError("Invalid instId '{}'".format(inst_id))
        base, quote = parts
        if quote == "BTC":
            return "{}_{}".format(quote, base)
        if base == "BTC":
            return "{}_{}".format(base, quote)
        raise ValueError("Unsupported instId '{}' (only BTC quote supported)".format(inst_id))

    def _fetch_candles(self, pair, start, end):
        last_error = None
        for _ in range(self._history_retries):
            try:
                candles = self._public_client.marketChart(pair, self._period, start, end)
                if candles:
                    return candles
            except Exception as exc:
                last_error = exc
                time.sleep(1)
        if last_error:
            raise last_error
        return []

    def _fetch_balances(self):
        if not self._rest_client:
            return {}
        try:
            payload = self._rest_client.get_account_balances(inst_type="SPOT")
        except Exception as exc:
            logging.error("Failed to fetch OKX balances: %s", exc)
            return {}
        balances = {}
        for entry in payload or []:
            for detail in entry.get("details", []):
                currency = detail.get("ccy")
                if not currency:
                    continue
                avail = detail.get("availBal") or detail.get("cashBal") or detail.get("eq")
                try:
                    balances[currency.upper()] = float(avail)
                except (TypeError, ValueError):
                    continue
        return balances

    def _build_holdings(self, balances):
        holdings = {"BTC": balances.get(self._quote_currency, 0.0)}
        for coin in self._coin_name_list:
            currency = self._coin_to_currency(coin)
            holdings[coin] = balances.get(currency, 0.0)
        return holdings

    def _coin_to_currency(self, coin_name):
        return coin_name.replace("reversed_", "").upper()

    def _compute_total_capital(self, holdings):
        btc_value = holdings.get("BTC", 0.0)
        for coin in self._coin_name_list:
            qty = holdings.get(coin, 0.0)
            price = self._market_prices.get(coin)
            if qty and price:
                btc_value += qty * price
        return btc_value

    def _build_rebalance_orders(self, omega, holdings, total_capital):
        orders = []
        for idx, coin in enumerate(self._coin_name_list):
            price = self._market_prices.get(coin)
            if not price or price <= 0:
                continue
            target_weight = float(omega[idx + 1])
            target_value = total_capital * target_weight
            target_qty = target_value / price
            current_qty = holdings.get(coin, 0.0)
            diff = target_qty - current_qty
            notional = abs(diff) * price
            if notional < self._min_notional:
                continue
            side = "buy" if diff > 0 else "sell"
            orders.append(
                {
                    "coin": coin,
                    "inst": self._pair_map[coin]["inst"],
                    "side": side,
                    "size": self._format_size(abs(diff)),
                    "notional": notional,
                }
            )
        return orders

    def _execute_order(self, order):
        msg = "%s %s %s (≈ %.8f %s)" % (
            order["side"].upper(),
            order["size"],
            order["inst"],
            order["notional"],
            self._quote_currency,
        )
        if self._paper_trading or not self._rest_client:
            logging.info("[paper] %s", msg)
            return
        try:
            response = self._rest_client.place_order(
                inst_id=order["inst"],
                side=order["side"],
                ord_type="market",
                size=order["size"],
                td_mode=self._td_mode,
            )
            logging.info("Executed order %s -> %s", msg, response)
        except Exception as exc:
            logging.error("Order %s failed: %s", msg, exc)

    def _update_asset_vector_from_holdings(self, holdings, total_capital):
        if total_capital <= 0:
            return
        self._asset_vector[0] = holdings.get("BTC", 0.0) / total_capital
        for idx, coin in enumerate(self._coin_name_list):
            qty = holdings.get(coin, 0.0)
            price = self._market_prices.get(coin, 0.0)
            value = qty * price
            self._asset_vector[idx + 1] = value / total_capital if total_capital else 0.0

    def _align_timestamp(self, ts):
        return ts - (ts % self._period)

    @staticmethod
    def _safe_inverse(value):
        if value == 0:
            return 0.0
        return 1.0 / value

    @staticmethod
    def _format_size(size):
        precision = max(0, min(8, -int(math.floor(math.log10(size))) + 4)) if size > 0 else 4
        fmt = "{:0.%df}" % precision
        formatted = fmt.format(size).rstrip("0").rstrip(".")
        return formatted or "0"


