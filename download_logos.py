import os
import requests

from symbols import SYMBOLS


LOGO_DIR = "logos_images"

os.makedirs(LOGO_DIR, exist_ok=True)


def download_logo(symbol):
    url = f"https://companiesmarketcap.com/img/company-logos/64/{symbol}.png"

    file_path = os.path.join(LOGO_DIR, f"{symbol}.png")

    try:
        r = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        if r.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(r.content)

            print(f"OK {symbol}")

        else:
            print(f"FAIL {symbol}: {r.status_code}")

    except Exception as e:
        print(f"ERROR {symbol}: {e}")


for symbol in SYMBOLS:
    download_logo(symbol)