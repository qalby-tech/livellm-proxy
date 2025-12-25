"""Redis manager for encrypted provider settings storage"""

import json
import redis.asyncio as redis
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import logfire
from urllib.parse import urlparse, quote, urlunparse


class RedisManager:
    """Manager for encrypted Redis operations"""
    
    PROVIDERS_KEY = "livellm:providers"
    
    @staticmethod
    def _safe_encode_redis_url(redis_url: str) -> str:
        """
        Safely encode Redis URL credentials to handle special characters.
        
        Args:
            redis_url: Redis URL that may contain special characters in password
            
        Returns:
            URL with properly encoded credentials
        """
        try:
            parsed = urlparse(redis_url)
            
            # If there are credentials, encode them
            if parsed.username or parsed.password:
                # Encode username and password
                username = quote(parsed.username, safe='') if parsed.username else ''
                password = quote(parsed.password, safe='') if parsed.password else ''
                
                # Reconstruct netloc with encoded credentials
                if username and password:
                    netloc = f"{username}:{password}@{parsed.hostname}"
                elif password:
                    netloc = f":{password}@{parsed.hostname}"
                else:
                    netloc = f"{username}@{parsed.hostname}"
                
                # Add port if present
                if parsed.port:
                    netloc = f"{netloc}:{parsed.port}"
                
                # Reconstruct the URL
                encoded_url = urlunparse((
                    parsed.scheme,
                    netloc,
                    parsed.path,
                    parsed.params,
                    parsed.query,
                    parsed.fragment
                ))
                
                return encoded_url
            
            # No credentials, return as-is
            return redis_url
            
        except Exception as e:
            logfire.warn(f"Failed to parse Redis URL for encoding: {e}. Using original URL.")
            return redis_url
    
    def __init__(self, redis_url: Optional[str] = None, encryption_salt: Optional[str] = None):
        """
        Initialize Redis manager with optional encryption.
        
        Args:
            redis_url: Redis connection URL (e.g., "redis://localhost:6379/0" or "redis://:password@host:port/db")
                      Passwords with special characters will be automatically URL-encoded
            encryption_salt: Salt for encryption key derivation
        """
        # Safely encode the Redis URL if provided
        self.redis_url = self._safe_encode_redis_url(redis_url) if redis_url else None
        self.encryption_salt = encryption_salt
        self.redis_client: Optional[redis.Redis] = None
        self.cipher: Optional[Fernet] = None
        self.enabled = redis_url is not None
        
        if self.enabled and encryption_salt:
            # Derive encryption key from salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=encryption_salt.encode(),
                iterations=100000
            )
            key = base64.urlsafe_b64encode(kdf.derive(b"livellm-proxy-key"))
            self.cipher = Fernet(key)
            logfire.info("Redis manager initialized with encryption enabled")
        elif self.enabled:
            logfire.warn("Redis enabled but no encryption salt provided - data will be stored unencrypted")
        else:
            logfire.info("Redis manager disabled - no Redis URL provided")
    
    async def connect(self):
        """Establish Redis connection"""
        if not self.enabled:
            return
        
        try:
            self.redis_client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False  # We handle decoding after decryption
            )
            # Test connection
            await self.redis_client.ping()
            logfire.info("Successfully connected to Redis")
        except Exception as e:
            logfire.error(f"Failed to connect to Redis: {e}")
            self.enabled = False
            self.redis_client = None
    
    async def disconnect(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            logfire.info("Redis connection closed")
    
    def _encrypt(self, data: str) -> bytes:
        """Encrypt data if cipher is available"""
        if self.cipher:
            return self.cipher.encrypt(data.encode())
        return data.encode()
    
    def _decrypt(self, data: bytes) -> str:
        """Decrypt data if cipher is available"""
        if self.cipher:
            return self.cipher.decrypt(data).decode()
        return data.decode()
    
    async def save_provider_settings(self, uid: str, settings_dict: Dict[str, Any]) -> bool:
        """
        Save provider settings to Redis.
        
        Args:
            uid: Provider unique identifier
            settings_dict: Settings dictionary to save
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            # Convert to JSON
            json_data = json.dumps(settings_dict)
            # Encrypt
            encrypted_data = self._encrypt(json_data)
            # Store in hash
            await self.redis_client.hset(self.PROVIDERS_KEY, uid, encrypted_data)
            logfire.info(f"Saved provider settings for uid: {uid}")
            return True
        except Exception as e:
            logfire.error(f"Failed to save provider settings for {uid}: {e}")
            return False
    
    async def load_provider_settings(self, uid: str) -> Optional[Dict[str, Any]]:
        """
        Load provider settings from Redis.
        
        Args:
            uid: Provider unique identifier
            
        Returns:
            Settings dictionary or None if not found
        """
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            encrypted_data = await self.redis_client.hget(self.PROVIDERS_KEY, uid)
            if not encrypted_data:
                return None
            
            # Decrypt
            json_data = self._decrypt(encrypted_data)
            # Parse JSON
            settings_dict = json.loads(json_data)
            logfire.info(f"Loaded provider settings for uid: {uid}")
            return settings_dict
        except Exception as e:
            logfire.error(f"Failed to load provider settings for {uid}: {e}")
            return None
    
    async def load_all_provider_settings(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all provider settings from Redis.
        
        Returns:
            Dictionary mapping uid to settings dictionary
        """
        if not self.enabled or not self.redis_client:
            return {}
        
        try:
            all_data = await self.redis_client.hgetall(self.PROVIDERS_KEY)
            if not all_data:
                return {}
            
            result = {}
            for uid_bytes, encrypted_data in all_data.items():
                try:
                    uid = uid_bytes.decode() if isinstance(uid_bytes, bytes) else uid_bytes
                    json_data = self._decrypt(encrypted_data)
                    settings_dict = json.loads(json_data)
                    result[uid] = settings_dict
                except Exception as e:
                    logfire.error(f"Failed to decrypt/parse settings for uid {uid}: {e}")
                    continue
            
            logfire.info(f"Loaded {len(result)} provider settings from Redis")
            return result
        except Exception as e:
            logfire.error(f"Failed to load all provider settings: {e}")
            return {}
    
    async def delete_provider_settings(self, uid: str) -> bool:
        """
        Delete provider settings from Redis.
        
        Args:
            uid: Provider unique identifier
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            await self.redis_client.hdel(self.PROVIDERS_KEY, uid)
            logfire.info(f"Deleted provider settings for uid: {uid}")
            return True
        except Exception as e:
            logfire.error(f"Failed to delete provider settings for {uid}: {e}")
            return False
    
    async def clear_all_provider_settings(self) -> bool:
        """
        Clear all provider settings from Redis.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            await self.redis_client.delete(self.PROVIDERS_KEY)
            logfire.info("Cleared all provider settings from Redis")
            return True
        except Exception as e:
            logfire.error(f"Failed to clear provider settings: {e}")
            return False

