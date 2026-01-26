import requests
import csv
import time

# 设置你的代理服务器地址和端口
proxies = {
    "http": "http://127.0.0.1:7897",  # 请根据你的代理实际地址和端口修改
    "https": "http://127.0.0.1:7897"
}

def download_binance_data(symbol: str, filename: str, count: int = 100):
    url_price = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    url_funding = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'price', 'funding_rate'])
        for _ in range(count):
            try:
                price = float(requests.get(url_price, timeout=10, proxies=proxies).json()['price'])
                funding_rate = float(requests.get(url_funding, timeout=10, proxies=proxies).json()['lastFundingRate'])
                writer.writerow([int(time.time()), price, funding_rate])
            except Exception as e:
                print(f"下载失败: {e}")
            time.sleep(1)

def download_okx_data(instId: str, filename: str, count: int = 100):
    url_price = f"https://www.okx.com/api/v5/market/ticker?instId={instId}"
    url_funding = f"https://www.okx.com/api/v5/public/funding-rate?instId={instId}"
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'price', 'funding_rate'])
        for _ in range(count):
            try:
                price = float(requests.get(url_price, timeout=10, proxies=proxies).json()['data'][0]['last'])
                funding_rate = float(requests.get(url_funding, timeout=10, proxies=proxies).json()['data'][0]['fundingRate'])
                writer.writerow([int(time.time()), price, funding_rate])
            except Exception as e:
                print(f"下载失败: {e}")
            time.sleep(1)

if __name__ == "__main__":
    download_binance_data("BTCUSDT", "binance_btcusdt.csv", count=100)
    download_okx_data("BTC-USDT-SWAP", "okx_btcusdt.csv", count=100)
