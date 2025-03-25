"""
Caching utilities for API responses.

This module provides a disk-based caching system to reduce redundant API calls,
improve performance, and respect rate limits. The cache stores the results of 
API requests based on their parameters and provides automatic expiration.

The caching system:
1. Generates unique keys based on function arguments
2. Stores results in a disk cache with configurable TTL
3. Automatically retrieves cached results for repeated calls
4. Handles complex data types through serialization
"""
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

from diskcache import Cache

# Type variable for generic return type - allows for proper type hinting
# when using the cache decorator with different return types
T = TypeVar('T')


@dataclass
class CacheConfig:
    """
    Configuration for the cache system.
    
    This class centralizes all cache configuration options and provides
    sensible defaults. Using a dataclass makes it easy to customize
    individual settings when needed.
    
    Attributes:
        cache_dir: Directory where cache files will be stored
        default_ttl: Default time-to-live in seconds for cached items (1 hour)
        size_limit: Maximum size of the cache in bytes (1GB)
        create_if_missing: Whether to create the cache directory if it doesn't exist
    """
    # Directory to store cache files
    cache_dir: str = ".cache"
    # Default cache expiration in seconds (1 hour)
    default_ttl: int = 3600
    # Maximum cache size in bytes (1GB)
    size_limit: int = 1_000_000_000
    # Whether to create the cache directory if it doesn't exist
    create_if_missing: bool = True


# Global cache instance - initialized once and reused for efficiency
_CACHE: Optional[Cache] = None


def get_cache(config: Optional[CacheConfig] = None) -> Cache:
    """
    Get the cache instance, creating it if necessary.
    
    This function implements the Singleton pattern for the cache object,
    ensuring that only one cache instance is created and reused throughout
    the application lifecycle.
    
    Args:
        config: Cache configuration, or None to use defaults
        
    Returns:
        Cache instance ready for use
    """
    global _CACHE
    if _CACHE is None:
        # Use default config if none provided
        if config is None:
            config = CacheConfig()
        
        # Create cache directory if it doesn't exist
        if config.create_if_missing:
            os.makedirs(config.cache_dir, exist_ok=True)
        
        # Create cache instance using the diskcache library
        # which provides persistent, dictionary-like storage
        _CACHE = Cache(
            directory=config.cache_dir,
            size_limit=config.size_limit,
        )
    
    return _CACHE


def generate_cache_key(prefix: str, *args: Any, **kwargs: Any) -> str:
    """
    Generate a unique cache key from the function arguments.
    
    This function creates a deterministic, unique key based on the function
    arguments. It handles complex types by using JSON serialization, and
    supports objects with a 'to_dict' method (like QueryParams).
    
    Args:
        prefix: Prefix for the cache key (usually the function name)
        *args: Positional arguments to be included in the key
        **kwargs: Keyword arguments to be included in the key
        
    Returns:
        A unique string key that can be used for cache lookups
    """
    # Process arguments to ensure they are JSON serializable
    # This is important for complex objects like custom classes
    processed_args = []
    for arg in args:
        if hasattr(arg, 'to_dict') and callable(getattr(arg, 'to_dict')):
            # For objects with a to_dict method (like QueryParams), use their dict representation
            processed_args.append(arg.to_dict())
        else:
            processed_args.append(arg)
    
    # Process kwargs to ensure they are JSON serializable
    processed_kwargs = {}
    for k, v in kwargs.items():
        if hasattr(v, 'to_dict') and callable(getattr(v, 'to_dict')):
            processed_kwargs[k] = v.to_dict()
        else:
            processed_kwargs[k] = v
    
    # Convert args and kwargs to a string representation using JSON
    # sort_keys ensures consistent ordering for dictionaries
    args_str = json.dumps(processed_args, sort_keys=True)
    kwargs_str = json.dumps(processed_kwargs, sort_keys=True)
    
    # Generate a hash of the combined string to create a short, fixed-length key
    # MD5 is used for speed and consistency, not for security purposes
    key_hash = hashlib.md5(f"{prefix}:{args_str}:{kwargs_str}".encode()).hexdigest()
    
    # Include the prefix in the final key for easier debugging and selective clearing
    return f"{prefix}:{key_hash}"


def cached(ttl: Optional[int] = None, prefix: Optional[str] = None, exclude_self: bool = True):
    """
    Decorator that caches function results to avoid redundant API calls.
    
    This async-compatible decorator wraps functions to cache their results based
    on the arguments provided. It's particularly useful for expensive API calls.
    
    The cache key is generated from:
    - The function name (or custom prefix)
    - The function arguments (excluding 'self' by default)
    
    Args:
        ttl: Time-to-live in seconds, or None to use the default (1 hour)
        prefix: Custom prefix for the cache key, or None to use the function name
        exclude_self: Whether to exclude the first parameter (self) when generating the cache key
                     This is useful for methods in adapter classes
        
    Returns:
        Decorator function that adds caching behavior to the wrapped function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            # Determine cache key prefix (function name or custom)
            key_prefix = prefix if prefix is not None else func.__name__
            
            # Generate cache key - exclude self parameter if requested and available
            # This is important for adapter methods where 'self' shouldn't affect the key
            cache_args = args[1:] if exclude_self and args else args
            key = generate_cache_key(key_prefix, *cache_args, **kwargs)
            
            # Check if result is already in cache
            cache = get_cache()
            result = cache.get(key)
            
            if result is not None:
                # Cache hit - return the cached result without calling the function
                return result
            
            # Cache miss - call the function and store the result
            result = await func(*args, **kwargs)
            
            # Use provided TTL or fall back to default
            cache_ttl = ttl if ttl is not None else CacheConfig().default_ttl
            cache.set(key, result, expire=cache_ttl)
            
            return result
        
        return wrapper
    
    return decorator


def clear_cache(prefix: Optional[str] = None):
    """
    Clear items from the cache.
    
    This utility function allows clearing either the entire cache or
    just items with a specific prefix.
    
    Args:
        prefix: Clear only keys with this prefix, or None to clear all items
    """
    cache = get_cache()
    
    if prefix is None:
        # Clear all keys (more efficient than selective clearing)
        cache.clear()
    else:
        # Clear keys with the given prefix
        # Note: This is not efficient for large caches as it has to iterate all keys
        # Future improvement: Use a more efficient prefix-based lookup
        for key in cache:
            if isinstance(key, str) and key.startswith(prefix):
                cache.delete(key) 