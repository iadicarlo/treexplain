"""Fetch satellite data using STAC API."""
import os
from datetime import datetime, timedelta
from pathlib import Path

import pystac
from pystac_client import Client


def fetch_sentinel2(bbox, days_back=7, limit=5):
    """Fetch Sentinel-2 items from Planetary Computer.
    
    Args:
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
        days_back: Number of days to look back
        limit: Maximum number of items to return
        
    Returns:
        List of STAC items
    """
    client = Client.open(
        os.getenv("STAC_API_URL", "https://planetarycomputer.microsoft.com/api/stac/v1")
    )
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        limit=limit,
    )
    
    return list(search.get_items())
