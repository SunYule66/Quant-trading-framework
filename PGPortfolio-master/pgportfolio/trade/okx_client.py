import base64
import hashlib
import hmac
import json
import time
from urllib.parse import urlencode

import requests


class OkxRESTClient:
    """
    Minimal OKX private REST client that covers the endpoints required for
    live trading (balances, order submission and cancellation).

    The client stays stateless so you can wire it into custom trading loops
    or into Trader.trade_by_strategy later on.
    """

    def __init__(
        self,
        api_key,
        secret_key,
        passphrase,
        base_url="https://www.okx.com",
        use_simulated=False,
        timeout=10,
    ):
        if not (api_key and secret_key and passphrase):
            raise ValueError("OKX credentials (api_key, secret_key, passphrase) are required")
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "OK-ACCESS-KEY": self.api_key,
                "OK-ACCESS-PASSPHRASE": self.passphrase,
            }
        )
        if use_simulated:
            self._session.headers["x-simulated-trading"] = "1"

    # ----------------------
    # Public helper methods
    # ----------------------
    def get_server_time(self):
        return self._public_request("GET", "/api/v5/public/time")

    def get_account_balances(self, inst_type="SPOT"):
        return self._private_request(
            "GET", "/api/v5/account/balance", params={"ccy": "", "instType": inst_type}
        )

    def place_order(
        self,
        inst_id,
        side,
        ord_type,
        size,
        price=None,
        td_mode="cash",
        client_order_id=None,
    ):
        payload = {
            "instId": inst_id,
            "tdMode": td_mode,
            "side": side,
            "ordType": ord_type,
            "sz": str(size),
        }
        if price is not None:
            payload["px"] = str(price)
        if client_order_id:
            payload["clOrdId"] = client_order_id
        return self._private_request("POST", "/api/v5/trade/order", body=payload)

    def cancel_order(self, inst_id, order_id=None, client_order_id=None):
        if not (order_id or client_order_id):
            raise ValueError("order_id or client_order_id must be provided")
        payload = {"instId": inst_id}
        if order_id:
            payload["ordId"] = order_id
        if client_order_id:
            payload["clOrdId"] = client_order_id
        return self._private_request("POST", "/api/v5/trade/cancel-order", body=payload)

    # ----------------------
    # Internal HTTP helpers
    # ----------------------
    def _timestamp(self):
        return "{:.3f}".format(time.time())

    def _sign(self, timestamp, method, request_path, body):
        message = f"{timestamp}{method}{request_path}{body}"
        mac = hmac.new(
            self.secret_key.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        )
        return base64.b64encode(mac.digest()).decode("utf-8")

    def _public_request(self, method, path, params=None):
        url = self.base_url + path
        if params:
            url = url + "?" + urlencode(params)
        response = self._session.request(method, url, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") not in (None, 0, "0"):
            raise ValueError("OKX API error {}: {}".format(payload.get("code"), payload.get("msg")))
        return payload.get("data", payload)

    def _private_request(self, method, path, params=None, body=None):
        method = method.upper()
        request_path = path
        query = ""
        if params:
            query = urlencode(params)
            request_path = "{}?{}".format(path, query)
        body_str = json.dumps(body) if body else ""

        timestamp = self._timestamp()
        signature = self._sign(timestamp, method, request_path, body_str)

        headers = {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
        }

        url = self.base_url + path
        if query:
            url = "{}?{}".format(url, query)

        response = self._session.request(
            method,
            url,
            data=body_str if body_str else None,
            timeout=self.timeout,
            headers=headers,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") not in (None, 0, "0"):
            raise ValueError("OKX API error {}: {}".format(payload.get("code"), payload.get("msg")))
        return payload.get("data", payload)


