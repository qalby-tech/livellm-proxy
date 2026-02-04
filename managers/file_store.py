"""File manager for encrypted provider settings storage"""

import json
import aiofiles
import os
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import logfire

class FileStoreManager:
    """Manager for encrypted file-based operations"""
    
    def __init__(self, file_path: str, encryption_salt: Optional[str] = None):
        """
        Initialize FileStore manager with optional encryption.
        
        Args:
            file_path: Path to the JSON file for storage
            encryption_salt: Salt for encryption key derivation
        """
        self.file_path = file_path
        self.encryption_salt = encryption_salt
        self.cipher: Optional[Fernet] = None
        
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        if encryption_salt:
            # Derive encryption key from salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=encryption_salt.encode(),
                iterations=100000
            )
            key = base64.urlsafe_b64encode(kdf.derive(b"livellm-proxy-key"))
            self.cipher = Fernet(key)
            logfire.info("FileStore manager initialized with encryption enabled")
        else:
            logfire.warn("FileStore manager enabled but no encryption salt provided - data will be stored unencrypted")

    def _encrypt(self, data: str) -> str:
        """Encrypt data if cipher is available"""
        if self.cipher:
            return self.cipher.encrypt(data.encode()).decode()
        return data
    
    def _decrypt(self, data: str) -> str:
        """Decrypt data if cipher is available"""
        if self.cipher:
            return self.cipher.decrypt(data.encode()).decode()
        return data

    async def _read_file(self) -> Dict[str, str]:
        """Read all data from file"""
        if not os.path.exists(self.file_path):
            return {}
        
        try:
            async with aiofiles.open(self.file_path, 'r') as f:
                content = await f.read()
                if not content:
                    return {}
                return json.loads(content)
        except Exception as e:
            logfire.error(f"Failed to read file {self.file_path}: {e}")
            return {}

    async def _write_file(self, data: Dict[str, str]) -> bool:
        """Write data to file"""
        try:
            async with aiofiles.open(self.file_path, 'w') as f:
                await f.write(json.dumps(data, indent=2))
            return True
        except Exception as e:
            logfire.error(f"Failed to write to file {self.file_path}: {e}")
            return False

    async def save_provider_settings(self, uid: str, settings_dict: Dict[str, Any]) -> bool:
        """
        Save provider settings to file.
        
        Args:
            uid: Provider unique identifier
            settings_dict: Settings dictionary to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert to JSON
            json_data = json.dumps(settings_dict)
            # Encrypt
            encrypted_data = self._encrypt(json_data)
            
            # Read existing data
            all_data = await self._read_file()
            
            # Update data
            all_data[uid] = encrypted_data
            
            # Write back
            if await self._write_file(all_data):
                logfire.info(f"Saved provider settings for uid: {uid}")
                return True
            return False
        except Exception as e:
            logfire.error(f"Failed to save provider settings for {uid}: {e}")
            return False

    async def load_provider_settings(self, uid: str) -> Optional[Dict[str, Any]]:
        """
        Load provider settings from file.
        
        Args:
            uid: Provider unique identifier
            
        Returns:
            Settings dictionary or None if not found
        """
        try:
            all_data = await self._read_file()
            encrypted_data = all_data.get(uid)
            
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
        Load all provider settings from file.
        
        Returns:
            Dictionary mapping uid to settings dictionary
        """
        try:
            all_data = await self._read_file()
            if not all_data:
                return {}
            
            result = {}
            for uid, encrypted_data in all_data.items():
                try:
                    if not encrypted_data:
                        logfire.warn(f"Empty settings data for uid {uid} in file, skipping")
                        continue

                    json_data = self._decrypt(encrypted_data)
                    if not json_data:
                        logfire.warn(f"Decrypted data is empty for uid {uid}, skipping")
                        continue

                    settings_dict = json.loads(json_data)
                    result[uid] = settings_dict
                except Exception as e:
                    logfire.error(f"Failed to decrypt/parse settings for uid {uid}: {e}")
                    continue
            
            logfire.info(f"Loaded {len(result)} provider settings from file")
            return result
        except Exception as e:
            logfire.error(f"Failed to load all provider settings: {e}")
            return {}

    async def delete_provider_settings(self, uid: str) -> bool:
        """
        Delete provider settings from file.
        
        Args:
            uid: Provider unique identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            all_data = await self._read_file()
            if uid in all_data:
                del all_data[uid]
                await self._write_file(all_data)
                logfire.info(f"Deleted provider settings for uid: {uid}")
                return True
            return False
        except Exception as e:
            logfire.error(f"Failed to delete provider settings for {uid}: {e}")
            return False

    async def clear_all_provider_settings(self) -> bool:
        """
        Clear all provider settings from file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            await self._write_file({})
            logfire.info("Cleared all provider settings from file")
            return True
        except Exception as e:
            logfire.error(f"Failed to clear provider settings: {e}")
            return False


