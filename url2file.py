from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
import aiofiles
import psutil
from pathlib import Path
from math import floor


def compute_max_sessions(
    mem_per_session_mb: int = 300,   # empirical Playwright tab cost
    safety_frac: float = 0.75,       # leave head-room for OS + Python
) -> int:
    """
    Decide how many concurrent browser contexts we can afford.

    Returns an int >= 1
    """
    total_mb = psutil.virtual_memory().total / 1024**2  
    usable_mb = total_mb * safety_frac
    return max(1, floor(usable_mb / mem_per_session_mb))

async def save_result(result, output_dir: str):
    """
    Save crawl result to markdown file using async file I/O.
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create a safe filename from URL
    safe_name = (
        result.url
        .replace("https://", "")
        .replace("http://", "")
        .replace("/", "_")
        .replace("?", "_")
        .replace("&", "_")
    )[:100]  # Limit length
    
    # Write file asynchronously
    async with aiofiles.open(
        Path(output_dir) / f"{safe_name}.md",
        "w",
        encoding="utf-8",
    ) as f:
        await f.write(result.markdown)

async def url2file_full(urls, output_dir="extracted_files"):
    # Ensure urls is a list
    if isinstance(urls, str):
        urls = [urls]
    
    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        stream=True  # Enable streaming mode
    )

    dispatcher = MemoryAdaptiveDispatcher(
        max_session_permit=10,
        memory_threshold_percent=80.0,
        check_interval=1.0,
        monitor=None  # Disable monitor to avoid termios issue on Windows
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Process results as they become available
        async for result in await crawler.arun_many(
            urls=urls,
            config=run_config,
            dispatcher=dispatcher
        ):
            if result.success:
                # Process each result immediately
                await save_result(result, output_dir)
            else:
                print(f"✗ Failed to crawl {result.url}: {result.error_message}")

async def url2file(urls, output_dir="extracted_files"):
    await url2file_full(urls=urls, output_dir=output_dir)





