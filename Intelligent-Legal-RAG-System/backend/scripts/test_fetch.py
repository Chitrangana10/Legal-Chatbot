import requests

headers = {"User-Agent": "Mozilla/5.0"}

try:
    indian_kanoon_response = requests.get(
        "https://indiankanoon.org/doc/1560742/",
        headers=headers,
        timeout=20,
    )
    print("URL 1: https://indiankanoon.org/doc/1560742/")
    print("status_code:", indian_kanoon_response.status_code)
    print(indian_kanoon_response.text[2000:6000])
    print("---")
except Exception as e:
    print(f"Request failed for URL 1: {e}")
    print("---")

try:
    session = requests.Session()
    session.get("https://www.indiacode.nic.in/", headers=headers, timeout=20)
    india_code_response = session.get(
        "https://www.indiacode.nic.in/handle/123456789/1362/simple-search?query=302&searchradio=section",
        headers=headers,
        timeout=20,
    )
    print("URL 2: https://www.indiacode.nic.in/handle/123456789/1362/simple-search?query=302&searchradio=section")
    print("status_code:", india_code_response.status_code)
    print(india_code_response.text[:3000])
except Exception as e:
    print(f"Request failed for URL 2: {e}")
