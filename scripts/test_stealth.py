import sys
import os
import time

# Add scripts directory to path
sys.path.append(os.path.join(os.getcwd(), 'scripts'))
from browser_scraper import BrowserRenderer

def test_scraper():
    print("Initializing Stealth BrowserRenderer...")
    # Force seleniumbase engine to test the new 'uc' integration
    os.environ['BROWSER_ENGINE'] = 'seleniumbase'
    os.environ['BROWSER_AUTO_PREFER'] = 'seleniumbase'
    
    test_urls = [
        "https://www.flashscore.com/football/england/premier-league/fixtures/",
        "https://nowsecure.nl/" # Industry standard Cloudflare/Anti-bot test
    ]
    
    try:
        with BrowserRenderer(engine="seleniumbase", headless=True, timeout_ms=30000) as renderer:
            print(f"Browser successfully initialized! Engine: {renderer.active_engine}")
            
            for url in test_urls:
                print(f"\n[Scraping] -> {url}")
                start_time = time.time()
                # Bypass cache to force a real fetch
                result = renderer.fetch(url, use_cache=False)
                elapsed = time.time() - start_time
                
                title = result.get('title', 'Unknown Title')
                text_len = len(result.get('text', ''))
                html_len = len(result.get('html', ''))
                
                print(f"  [+] Success! ({elapsed:.2f}s)")
                print(f"  [+] Page Title: '{title}'")
                print(f"  [+] Extracted HTML length: {html_len} bytes")
                print(f"  [+] Extracted Text length: {text_len} bytes")
                
                if "Just a moment" in title or "Cloudflare" in title or "Access Denied" in title:
                    print("  [!] WARNING: Bot detection triggered! Bypass failed.")
                else:
                    print("  [+] Stealth check PASSED! No bot detection found.")
                
    except Exception as e:
        print(f"\n[!] Error during execution: {e}")

if __name__ == "__main__":
    test_scraper()
