"""
Kite Connect Setup Script
Generates access token from request token.

Steps:
1. Get API Key/Secret from https://developers.kite.trade/
2. Fill in config.py with your API key/secret
3. Run this script
4. It will give you a URL - open it in browser
5. Login and get the request token from the redirect URL
6. Paste the request token when prompted
7. Copy the generated access token to your .env file
"""

import os
from dotenv import load_dotenv

load_dotenv()

from config import KITE_API_KEY, KITE_API_SECRET
import kiteconnect as kite

def get_access_token():
    """Generate access token from request token."""
    if not KITE_API_KEY or not KITE_API_SECRET:
        print("ERROR: Please fill KITE_API_KEY and KITE_API_SECRET in .env first")
        return

    print("\n" + "="*60)
    print("KITE CONNECT SETUP")
    print("="*60)

    # Step 1: Generate login URL
    kite_obj = kite.KiteConnect(api_key=KITE_API_KEY)
    login_url = kite_obj.login_url()

    print(f"\n1. Open this URL in your browser:")
    print(f"\n   {login_url}\n")
    print("2. Login with your Zerodha credentials")
    print("3. After login, you'll be redirected to your registered redirect URL")
    print("4. Copy the 'request_token' from the redirect URL")
    print("   (It looks like: ?request_token=xxxxxxxxx&...)")
    print("\n")

    request_token = input("Enter the request token: ").strip()

    if not request_token:
        print("ERROR: No request token provided")
        return

    # Step 2: Generate access token
    try:
        print("\nGenerating access token...")
        data = kite_obj.generate_session(request_token, api_secret=KITE_API_SECRET)

        access_token = data.get("access_token")
        print(f"\n{'='*60}")
        print("SUCCESS! Your access token:")
        print(f"{'='*60}")
        print(f"\n{access_token}\n")
        print(f"{'='*60}")
        print("\nADD THIS TO YOUR .env FILE:")
        print(f"KITE_ACCESS_TOKEN={access_token}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    get_access_token()