"""Redis manager for encrypted provider settings storage with Pub/Sub event publishing"""

import base64
import json
from typing import Any, Dict, Optional
from urllib.parse import quote, urlparse, urlunparse

import logfire
import redis.asyncio as redis
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class RedisManager:
    """Manager for encrypted Redis operations with Pub/Sub event publishing"""

    PROVIDERS_KEY = "livellm:providers"
    PROVIDERS_CHANNEL = "livellm:providers:events"

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

            if parsed.username or parsed.password:
                username = quote(parsed.username, safe="") if parsed.username else ""
                password = quote(parsed.password, safe="") if parsed.password else ""

                if username and password:
                    netloc = f"{username}:{password}@{parsed.hostname}"
                elif password:
                    netloc = f":{password}@{parsed.hostname}"
                else:
                    netloc = f"{username}@{parsed.hostname}"

                if parsed.port:
                    netloc = f"{netloc}:{parsed.port}"

                encoded_url = urlunparse(
                    (
                        parsed.scheme,
                        netloc,
                        parsed.path,
                        parsed.params,
                        parsed.query,
                        parsed.fragment,
                    )
                )

                return encoded_url

            return redis_url

        except Exception as e:
            logfire.warn(
                f"Failed to parse Redis URL for encoding: {e}. Using original URL."
            )
            return redis_url

    def __init__(self, redis_url: str, encryption_salt: Optional[str] = None):
        """
        Initialize Redis manager.

        Args:
            redis_url: Redis connection URL (required)
            encryption_salt: Salt for encryption key derivation. If not provided,
                             data will be stored as plain JSON bytes.
        """
        self.redis_url = self._safe_encode_redis_url(redis_url)
        self.encryption_salt = encryption_salt
        self.redis_client: Optional[redis.Redis] = None
        self.cipher: Optional[Fernet] = None

        if encryption_salt:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=encryption_salt.encode(),
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(b"livellm-proxy-key"))
            self.cipher = Fernet(key)
            logfire.info("Redis manager initialized with encryption enabled")
        else:
            logfire.warn(
                "Redis manager initialized without encryption — data will be stored as plain JSON"
            )

    async def connect(self):
        """Establish Redis connection"""
        try:
            self.redis_client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,  # We handle decoding after optional decryption
            )
            await self.redis_client.ping()
            logfire.info("Successfully connected to Redis")
        except Exception as e:
            logfire.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.aclose()
            logfire.info("Redis connection closed")

    # ------------------------------------------------------------------
    # Encryption helpers
    # ------------------------------------------------------------------

    def _encrypt(self, data: str) -> bytes:
        """Encrypt a JSON string. Returns Fernet token bytes, or plain UTF-8 bytes."""
        if self.cipher:
            return self.cipher.encrypt(data.encode())
        return data.encode()

    def _decrypt(self, data: bytes) -> str:
        """Decrypt bytes from Redis back to a JSON string."""
        if self.cipher:
            return self.cipher.decrypt(data).decode()
        return data.decode()

    # ------------------------------------------------------------------
    # Pub/Sub
    # ------------------------------------------------------------------

    async def publish_provider_event(self, action: str, uid: str) -> bool:
        """
        Publish a provider change event on the Pub/Sub channel so that all
        proxy replicas can hot-reload in-memory state without restarting.

        Args:
            action: "upsert" or "delete"
            uid:    Provider unique identifier
        """
        if not self.redis_client:
            return False
        try:
            msg = json.dumps({"action": action, "uid": uid})
            await self.redis_client.publish(self.PROVIDERS_CHANNEL, msg)
            logfire.debug(f"Published provider event: action={action} uid={uid}")
            return True
        except Exception as e:
            logfire.error(
                f"Failed to publish provider event (action={action} uid={uid}): {e}"
            )
            return False

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def save_provider_settings(
        self, uid: str, settings_dict: Dict[str, Any]
    ) -> bool:
        """
        Persist provider settings to Redis and notify all replicas via Pub/Sub.

        Args:
            uid:           Provider unique identifier
            settings_dict: Plain-Python dict of settings (including raw api_key string)

        Returns:
            True if successful, False otherwise
        """
        if not self.redis_client:
            return False
        try:
            json_data = json.dumps(settings_dict)
            encrypted_data = self._encrypt(json_data)

            await self.redis_client.hset(self.PROVIDERS_KEY, uid, encrypted_data)
            logfire.info(f"Saved provider settings for uid: {uid}")

            # Notify all replicas (including self) about the change
            await self.publish_provider_event("upsert", uid)
            return True
        except Exception as e:
            logfire.error(f"Failed to save provider settings for {uid}: {e}")
            return False

    async def load_provider_settings(self, uid: str) -> Optional[Dict[str, Any]]:
        """
        Load a single provider's settings from Redis.

        Args:
            uid: Provider unique identifier

        Returns:
            Settings dict or None if not found
        """
        if not self.redis_client:
            return None
        try:
            encrypted_data = await self.redis_client.hget(self.PROVIDERS_KEY, uid)
            if not encrypted_data:
                return None

            json_data = self._decrypt(encrypted_data)
            settings_dict = json.loads(json_data)
            logfire.debug(f"Loaded provider settings for uid: {uid}")
            return settings_dict
        except Exception as e:
            logfire.error(f"Failed to load provider settings for {uid}: {e}")
            return None

    async def load_all_provider_settings(self) -> Dict[str, Dict[str, Any]]:
        """
        Load every provider's settings from Redis in one call.

        Returns:
            Mapping of uid → settings dict
        """
        if not self.redis_client:
            return {}
        try:
            all_data = await self.redis_client.hgetall(self.PROVIDERS_KEY)
            if not all_data:
                return {}

            result = {}
            for uid_raw, encrypted_data in all_data.items():
                uid = uid_raw.decode() if isinstance(uid_raw, bytes) else uid_raw
                try:
                    if not encrypted_data:
                        logfire.warn(
                            f"Empty settings data for uid {uid} in Redis, skipping"
                        )
                        continue

                    json_data = self._decrypt(encrypted_data)
                    if not json_data:
                        logfire.warn(f"Decrypted data is empty for uid {uid}, skipping")
                        continue

                    settings_dict = json.loads(json_data)
                    result[uid] = settings_dict
                except Exception as e:
                    logfire.error(
                        f"Failed to decrypt/parse settings for uid {uid}: {e}"
                    )
                    continue

            logfire.info(f"Loaded {len(result)} provider settings from Redis")
            return result
        except Exception as e:
            logfire.error(f"Failed to load all provider settings: {e}")
            return {}

    async def delete_provider_settings(self, uid: str) -> bool:
        """
        Remove a provider from Redis and notify all replicas via Pub/Sub.

        Args:
            uid: Provider unique identifier

        Returns:
            True if successful, False otherwise
        """
        if not self.redis_client:
            return False
        try:
            await self.redis_client.hdel(self.PROVIDERS_KEY, uid)
            logfire.info(f"Deleted provider settings for uid: {uid}")

            # Notify all replicas about the deletion
            await self.publish_provider_event("delete", uid)
            return True
        except Exception as e:
            logfire.error(f"Failed to delete provider settings for {uid}: {e}")
            return False

    async def clear_all_provider_settings(self) -> bool:
        """
        Wipe all provider settings from Redis.

        Returns:
            True if successful, False otherwise
        """
        if not self.redis_client:
            return False
        try:
            await self.redis_client.delete(self.PROVIDERS_KEY)
            logfire.info("Cleared all provider settings from Redis")
            return True
        except Exception as e:
            logfire.error(f"Failed to clear provider settings: {e}")
            return False
