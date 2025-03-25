"""
Rate limiting utilities for API clients.
"""
import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from typing import Callable, Deque, Dict, Optional, TypeVar

# Type variable for generic return type
T = TypeVar('T')


@dataclass
class RateLimiter:
    """
    Rate limiter that enforces a maximum number of requests per time window.
    """
    # Number of requests allowed in the window
    max_requests: int
    # Time window in seconds
    window_seconds: int
    # Queue of timestamps for recent requests
    request_timestamps: Deque[float] = field(default_factory=deque)
    
    def __post_init__(self):
        """Initialize the rate limiter."""
        # Ensure request_timestamps is initialized
        if not self.request_timestamps:
            self.request_timestamps = deque(maxlen=self.max_requests)
    
    async def acquire(self):
        """
        Acquire permission to make a request.
        
        This method will sleep if necessary to enforce the rate limit.
        """
        # Get current time
        now = time.time()
        
        # Remove timestamps older than the window
        window_start = now - self.window_seconds
        while self.request_timestamps and self.request_timestamps[0] < window_start:
            self.request_timestamps.popleft()
        
        # If we've reached the limit, sleep until we can make another request
        if len(self.request_timestamps) >= self.max_requests:
            # Calculate sleep time
            oldest_timestamp = self.request_timestamps[0]
            sleep_time = oldest_timestamp + self.window_seconds - now
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        # Add current time to the queue
        self.request_timestamps.append(time.time())
    
    def reset(self):
        """Reset the rate limiter by clearing all timestamps."""
        self.request_timestamps.clear()


def rate_limited(limiter: RateLimiter):
    """
    Decorator that applies rate limiting to a function.
    
    Args:
        limiter: The rate limiter to use
        
    Returns:
        Decorated function that respects rate limits
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            await limiter.acquire()
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Global rate limiters for different API services
_RATE_LIMITERS: Dict[str, RateLimiter] = {
    "semantic_scholar": RateLimiter(max_requests=100, window_seconds=300),  # 100 per 5 minutes
    "arxiv": RateLimiter(max_requests=5, window_seconds=1),  # 5 per second
    "biorxiv": RateLimiter(max_requests=10, window_seconds=1),  # 10 per second
    "openalex": RateLimiter(max_requests=10, window_seconds=1),  # 10 per second
}


def get_rate_limiter(service_name: str) -> RateLimiter:
    """
    Get a rate limiter for the specified service.
    
    Args:
        service_name: Name of the service
        
    Returns:
        Rate limiter for the service
    """
    if service_name not in _RATE_LIMITERS:
        # Default to a conservative limit for unknown services
        _RATE_LIMITERS[service_name] = RateLimiter(max_requests=1, window_seconds=1)
    
    return _RATE_LIMITERS[service_name] 