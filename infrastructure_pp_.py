#!/usr/bin/env python3
"""
Infrastructure P&P (Protocols & Primitives) for Reus Veritas
البنية التحتية P&P (البروتوكولات والمكونات الأساسية) لمشروع Reus Veritas

هذا المكون مسؤول عن التوثيق والتشفير، الاتصال المجهول، والاسترجاع الآمن
باستخدام تقنيات IPFS، Tor، والتشفير المتقدم.

Author: Lotfi mahiddine
Date: 2025
"""

import json
import logging
import hashlib
import base64
import os
import socket
import threading
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import secrets
import ipaddress
import subprocess

# إعداد نظام السجلات
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('infrastructure_pp.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """مستويات الأمان"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class NetworkType(Enum):
    """أنواع الشبكات"""
    CLEARNET = "clearnet"
    TOR = "tor"
    I2P = "i2p"
    IPFS = "ipfs"

class EncryptionType(Enum):
    """أنواع التشفير"""
    AES_256 = "aes_256"
    RSA_4096 = "rsa_4096"
    HYBRID = "hybrid"

@dataclass
class DigitalFingerprint:
    """البصمة الرقمية"""
    id: str
    content_hash: str
    signature: str
    timestamp: datetime
    creator: str
    metadata: Dict[str, Any]
    verification_data: Dict[str, Any]

@dataclass
class SecureDocument:
    """مستند آمن"""
    id: str
    content: bytes
    fingerprint: DigitalFingerprint
    encryption_type: EncryptionType
    access_level: SecurityLevel
    ipfs_hash: Optional[str]
    created_at: datetime
    expires_at: Optional[datetime]

@dataclass
class NetworkNode:
    """عقدة شبكة"""
    id: str
    address: str
    port: int
    network_type: NetworkType
    public_key: str
    last_seen: datetime
    trust_score: float
    capabilities: List[str]

class CryptographyManager:
    """مدير التشفير"""
    
    def __init__(self):
        self.rsa_key_size = 4096
        self.aes_key_size = 32  # 256 bits
        self.salt_size = 16
        self.iv_size = 16
        
        # مفاتيح النظام
        self.system_private_key = None
        self.system_public_key = None
        
        self._initialize_system_keys()
    
    def _initialize_system_keys(self):
        """تهيئة مفاتيح النظام"""
        try:
            # محاولة تحميل المفاتيح الموجودة
            if os.path.exists("system_private_key.pem") and os.path.exists("system_public_key.pem"):
                self._load_system_keys()
            else:
                # إنشاء مفاتيح جديدة
                self._generate_system_keys()
            
            logger.info("تم تهيئة مفاتيح النظام بنجاح")
        
        except Exception as e:
            logger.error(f"فشل في تهيئة مفاتيح النظام: {e}")
            raise
    
    def _generate_system_keys(self):
        """إنشاء مفاتيح النظام"""
        # إنشاء مفتاح RSA
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.rsa_key_size
        )
        
        public_key = private_key.public_key()
        
        # حفظ المفتاح الخاص
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        with open("system_private_key.pem", "wb") as f:
            f.write(private_pem)
        
        # حفظ المفتاح العام
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        with open("system_public_key.pem", "wb") as f:
            f.write(public_pem)
        
        self.system_private_key = private_key
        self.system_public_key = public_key
        
        logger.info("تم إنشاء مفاتيح النظام الجديدة")
    
    def _load_system_keys(self):
        """تحميل مفاتيح النظام"""
        # تحميل المفتاح الخاص
        with open("system_private_key.pem", "rb") as f:
            private_pem = f.read()
        
        self.system_private_key = serialization.load_pem_private_key(
            private_pem,
            password=None
        )
        
        # تحميل المفتاح العام
        with open("system_public_key.pem", "rb") as f:
            public_pem = f.read()
        
        self.system_public_key = serialization.load_pem_public_key(public_pem)
        
        logger.info("تم تحميل مفاتيح النظام الموجودة")
    
    def generate_key_pair(self) -> Tuple[bytes, bytes]:
        """إنشاء زوج مفاتيح جديد"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.rsa_key_size
        )
        
        public_key = private_key.public_key()
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    def encrypt_data(self, data: bytes, encryption_type: EncryptionType = EncryptionType.HYBRID) -> Dict[str, Any]:
        """تشفير البيانات"""
        if encryption_type == EncryptionType.AES_256:
            return self._encrypt_aes(data)
        elif encryption_type == EncryptionType.RSA_4096:
            return self._encrypt_rsa(data)
        elif encryption_type == EncryptionType.HYBRID:
            return self._encrypt_hybrid(data)
        else:
            raise ValueError(f"نوع تشفير غير مدعوم: {encryption_type}")
    
    def decrypt_data(self, encrypted_data: Dict[str, Any]) -> bytes:
        """فك تشفير البيانات"""
        encryption_type = EncryptionType(encrypted_data["encryption_type"])
        
        if encryption_type == EncryptionType.AES_256:
            return self._decrypt_aes(encrypted_data)
        elif encryption_type == EncryptionType.RSA_4096:
            return self._decrypt_rsa(encrypted_data)
        elif encryption_type == EncryptionType.HYBRID:
            return self._decrypt_hybrid(encrypted_data)
        else:
            raise ValueError(f"نوع تشفير غير مدعوم: {encryption_type}")
    
    def _encrypt_aes(self, data: bytes) -> Dict[str, Any]:
        """تشفير AES"""
        # إنشاء مفتاح عشوائي
        key = secrets.token_bytes(self.aes_key_size)
        iv = secrets.token_bytes(self.iv_size)
        
        # تشفير البيانات
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        
        # إضافة padding
        padded_data = self._add_padding(data)
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        return {
            "encryption_type": EncryptionType.AES_256.value,
            "encrypted_data": base64.b64encode(encrypted_data).decode(),
            "key": base64.b64encode(key).decode(),
            "iv": base64.b64encode(iv).decode()
        }
    
    def _decrypt_aes(self, encrypted_data: Dict[str, Any]) -> bytes:
        """فك تشفير AES"""
        key = base64.b64decode(encrypted_data["key"])
        iv = base64.b64decode(encrypted_data["iv"])
        data = base64.b64decode(encrypted_data["encrypted_data"])
        
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        
        padded_data = decryptor.update(data) + decryptor.finalize()
        return self._remove_padding(padded_data)
    
    def _encrypt_rsa(self, data: bytes) -> Dict[str, Any]:
        """تشفير RSA"""
        if not self.system_public_key:
            raise ValueError("المفتاح العام غير متاح")
        
        # RSA يمكنه تشفير بيانات محدودة الحجم فقط
        max_chunk_size = (self.rsa_key_size // 8) - 42  # OAEP padding
        
        if len(data) > max_chunk_size:
            raise ValueError(f"البيانات كبيرة جداً للتشفير RSA. الحد الأقصى: {max_chunk_size} بايت")
        
        encrypted_data = self.system_public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return {
            "encryption_type": EncryptionType.RSA_4096.value,
            "encrypted_data": base64.b64encode(encrypted_data).decode()
        }
    
    def _decrypt_rsa(self, encrypted_data: Dict[str, Any]) -> bytes:
        """فك تشفير RSA"""
        if not self.system_private_key:
            raise ValueError("المفتاح الخاص غير متاح")
        
        data = base64.b64decode(encrypted_data["encrypted_data"])
        
        decrypted_data = self.system_private_key.decrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return decrypted_data
    
    def _encrypt_hybrid(self, data: bytes) -> Dict[str, Any]:
        """تشفير هجين (AES + RSA)"""
        # تشفير البيانات بـ AES
        aes_result = self._encrypt_aes(data)
        
        # تشفير مفتاح AES بـ RSA
        aes_key = base64.b64decode(aes_result["key"])
        rsa_result = self._encrypt_rsa(aes_key)
        
        return {
            "encryption_type": EncryptionType.HYBRID.value,
            "encrypted_data": aes_result["encrypted_data"],
            "encrypted_key": rsa_result["encrypted_data"],
            "iv": aes_result["iv"]
        }
    
    def _decrypt_hybrid(self, encrypted_data: Dict[str, Any]) -> bytes:
        """فك تشفير هجين"""
        # فك تشفير مفتاح AES
        rsa_data = {"encrypted_data": encrypted_data["encrypted_key"], "encryption_type": "rsa_4096"}
        aes_key = self._decrypt_rsa(rsa_data)
        
        # فك تشفير البيانات
        aes_data = {
            "encrypted_data": encrypted_data["encrypted_data"],
            "key": base64.b64encode(aes_key).decode(),
            "iv": encrypted_data["iv"],
            "encryption_type": "aes_256"
        }
        
        return self._decrypt_aes(aes_data)
    
    def _add_padding(self, data: bytes) -> bytes:
        """إضافة padding للبيانات"""
        block_size = 16
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length] * padding_length)
        return data + padding
    
    def _remove_padding(self, data: bytes) -> bytes:
        """إزالة padding من البيانات"""
        padding_length = data[-1]
        return data[:-padding_length]
    
    def create_digital_signature(self, data: bytes) -> str:
        """إنشاء توقيع رقمي"""
        if not self.system_private_key:
            raise ValueError("المفتاح الخاص غير متاح")
        
        signature = self.system_private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return base64.b64encode(signature).decode()
    
    def verify_digital_signature(self, data: bytes, signature: str, public_key_pem: Optional[bytes] = None) -> bool:
        """التحقق من التوقيع الرقمي"""
        try:
            if public_key_pem:
                public_key = serialization.load_pem_public_key(public_key_pem)
            else:
                public_key = self.system_public_key
            
            if not public_key:
                return False
            
            signature_bytes = base64.b64decode(signature)
            
            public_key.verify(
                signature_bytes,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True
        
        except Exception as e:
            logger.error(f"فشل في التحقق من التوقيع: {e}")
            return False

class IPFSManager:
    """مدير IPFS"""
    
    def __init__(self, api_url: str = "http://127.0.0.1:5001"):
        self.api_url = api_url
        self.session = requests.Session()
        self.is_available = self._check_ipfs_availability()
    
    def _check_ipfs_availability(self) -> bool:
        """التحقق من توفر IPFS"""
        try:
            response = self.session.get(f"{self.api_url}/api/v0/version", timeout=5)
            return response.status_code == 200
        except:
            logger.warning("IPFS غير متاح. سيتم استخدام التخزين المحلي.")
            return False
    
    def add_content(self, content: bytes) -> Optional[str]:
        """إضافة محتوى إلى IPFS"""
        if not self.is_available:
            return self._store_locally(content)
        
        try:
            files = {'file': content}
            response = self.session.post(
                f"{self.api_url}/api/v0/add",
                files=files,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                ipfs_hash = result.get("Hash")
                logger.info(f"تم رفع المحتوى إلى IPFS: {ipfs_hash}")
                return ipfs_hash
            else:
                logger.error(f"فشل في رفع المحتوى إلى IPFS: {response.status_code}")
                return self._store_locally(content)
        
        except Exception as e:
            logger.error(f"خطأ في رفع المحتوى إلى IPFS: {e}")
            return self._store_locally(content)
    
    def get_content(self, ipfs_hash: str) -> Optional[bytes]:
        """استرجاع محتوى من IPFS"""
        if not self.is_available:
            return self._retrieve_locally(ipfs_hash)
        
        try:
            response = self.session.post(
                f"{self.api_url}/api/v0/cat",
                params={'arg': ipfs_hash},
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"تم استرجاع المحتوى من IPFS: {ipfs_hash}")
                return response.content
            else:
                logger.error(f"فشل في استرجاع المحتوى من IPFS: {response.status_code}")
                return self._retrieve_locally(ipfs_hash)
        
        except Exception as e:
            logger.error(f"خطأ في استرجاع المحتوى من IPFS: {e}")
            return self._retrieve_locally(ipfs_hash)
    
    def _store_locally(self, content: bytes) -> str:
        """تخزين محلي كبديل"""
        content_hash = hashlib.sha256(content).hexdigest()
        
        # إنشاء مجلد التخزين المحلي
        storage_dir = Path("local_ipfs_storage")
        storage_dir.mkdir(exist_ok=True)
        
        # حفظ المحتوى
        file_path = storage_dir / f"{content_hash}.bin"
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"تم حفظ المحتوى محلياً: {content_hash}")
        return f"local:{content_hash}"
    
    def _retrieve_locally(self, hash_id: str) -> Optional[bytes]:
        """استرجاع محلي"""
        if hash_id.startswith("local:"):
            content_hash = hash_id[6:]  # إزالة "local:"
        else:
            content_hash = hash_id
        
        file_path = Path("local_ipfs_storage") / f"{content_hash}.bin"
        
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            logger.info(f"تم استرجاع المحتوى محلياً: {content_hash}")
            return content
        except FileNotFoundError:
            logger.error(f"المحتوى غير موجود محلياً: {content_hash}")
            return None

class AnonymousNetworkManager:
    """مدير الشبكات المجهولة"""
    
    def __init__(self):
        self.tor_proxy = None
        self.i2p_proxy = None
        self._check_anonymous_networks()
    
    def _check_anonymous_networks(self):
        """التحقق من توفر الشبكات المجهولة"""
        # التحقق من Tor
        if self._check_tor():
            self.tor_proxy = {
                'http': 'socks5://127.0.0.1:9050',
                'https': 'socks5://127.0.0.1:9050'
            }
            logger.info("Tor متاح")
        
        # التحقق من I2P
        if self._check_i2p():
            self.i2p_proxy = {
                'http': 'http://127.0.0.1:4444',
                'https': 'http://127.0.0.1:4444'
            }
            logger.info("I2P متاح")
    
    def _check_tor(self) -> bool:
        """التحقق من توفر Tor"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('127.0.0.1', 9050))
            sock.close()
            return result == 0
        except:
            return False
    
    def _check_i2p(self) -> bool:
        """التحقق من توفر I2P"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('127.0.0.1', 4444))
            sock.close()
            return result == 0
        except:
            return False
    
    def make_anonymous_request(self, url: str, network_type: NetworkType = NetworkType.TOR, 
                             method: str = "GET", data: Optional[Dict] = None) -> Optional[requests.Response]:
        """إجراء طلب مجهول"""
        try:
            session = requests.Session()
            
            if network_type == NetworkType.TOR and self.tor_proxy:
                session.proxies.update(self.tor_proxy)
            elif network_type == NetworkType.I2P and self.i2p_proxy:
                session.proxies.update(self.i2p_proxy)
            elif network_type == NetworkType.CLEARNET:
                pass  # لا حاجة لproxy
            else:
                logger.warning(f"الشبكة المطلوبة غير متاحة: {network_type.value}")
                return None
            
            # إعداد headers لتجنب الكشف
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            if method.upper() == "GET":
                response = session.get(url, headers=headers, timeout=30)
            elif method.upper() == "POST":
                response = session.post(url, headers=headers, json=data, timeout=30)
            else:
                logger.error(f"طريقة HTTP غير مدعومة: {method}")
                return None
            
            logger.info(f"تم إجراء طلب مجهول عبر {network_type.value}: {response.status_code}")
            return response
        
        except Exception as e:
            logger.error(f"فشل في الطلب المجهول: {e}")
            return None
    
    def generate_onion_address(self) -> Optional[str]:
        """إنشاء عنوان onion (محاكاة)"""
        # هذه دالة محاكاة - في التطبيق الحقيقي ستحتاج إلى Tor controller
        import random
        import string
        
        # إنشاء عنوان onion v3 وهمي
        onion_key = ''.join(random.choices(string.ascii_lowercase + string.digits, k=56))
        return f"{onion_key}.onion"

class DocumentManager:
    """مدير المستندات الآمنة"""
    
    def __init__(self, crypto_manager: CryptographyManager, ipfs_manager: IPFSManager):
        self.crypto_manager = crypto_manager
        self.ipfs_manager = ipfs_manager
        self.documents: Dict[str, SecureDocument] = {}
        self.fingerprints: Dict[str, DigitalFingerprint] = {}
    
    def create_secure_document(self, content: Union[str, bytes], 
                             access_level: SecurityLevel = SecurityLevel.MEDIUM,
                             expires_in_days: Optional[int] = None) -> SecureDocument:
        """إنشاء مستند آمن"""
        # تحويل المحتوى إلى bytes
        if isinstance(content, str):
            content_bytes = content.encode('utf-8')
        else:
            content_bytes = content
        
        # إنشاء معرف فريد
        doc_id = hashlib.sha256(content_bytes + str(time.time()).encode()).hexdigest()
        
        # إنشاء البصمة الرقمية
        fingerprint = self._create_digital_fingerprint(content_bytes, doc_id)
        
        # تحديد نوع التشفير بناءً على مستوى الأمان
        if access_level == SecurityLevel.CRITICAL:
            encryption_type = EncryptionType.HYBRID
        elif access_level == SecurityLevel.HIGH:
            encryption_type = EncryptionType.RSA_4096
        else:
            encryption_type = EncryptionType.AES_256
        
        # تشفير المحتوى
        encrypted_content = self.crypto_manager.encrypt_data(content_bytes, encryption_type)
        encrypted_bytes = json.dumps(encrypted_content).encode('utf-8')
        
        # رفع إلى IPFS
        ipfs_hash = self.ipfs_manager.add_content(encrypted_bytes)
        
        # تحديد تاريخ الانتهاء
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        # إنشاء المستند الآمن
        document = SecureDocument(
            id=doc_id,
            content=encrypted_bytes,
            fingerprint=fingerprint,
            encryption_type=encryption_type,
            access_level=access_level,
            ipfs_hash=ipfs_hash,
            created_at=datetime.now(),
            expires_at=expires_at
        )
        
        # حفظ المستند والبصمة
        self.documents[doc_id] = document
        self.fingerprints[fingerprint.id] = fingerprint
        
        logger.info(f"تم إنشاء مستند آمن: {doc_id}")
        
        return document
    
    def retrieve_secure_document(self, doc_id: str) -> Optional[bytes]:
        """استرجاع مستند آمن"""
        if doc_id not in self.documents:
            logger.error(f"المستند غير موجود: {doc_id}")
            return None
        
        document = self.documents[doc_id]
        
        # التحقق من انتهاء الصلاحية
        if document.expires_at and datetime.now() > document.expires_at:
            logger.error(f"انتهت صلاحية المستند: {doc_id}")
            return None
        
        try:
            # استرجاع المحتوى المشفر
            if document.ipfs_hash:
                encrypted_bytes = self.ipfs_manager.get_content(document.ipfs_hash)
                if not encrypted_bytes:
                    encrypted_bytes = document.content
            else:
                encrypted_bytes = document.content
            
            # فك التشفير
            encrypted_data = json.loads(encrypted_bytes.decode('utf-8'))
            decrypted_content = self.crypto_manager.decrypt_data(encrypted_data)
            
            logger.info(f"تم استرجاع المستند: {doc_id}")
            return decrypted_content
        
        except Exception as e:
            logger.error(f"فشل في استرجاع المستند {doc_id}: {e}")
            return None
    
    def verify_document_integrity(self, doc_id: str) -> bool:
        """التحقق من سلامة المستند"""
        if doc_id not in self.documents:
            return False
        
        document = self.documents[doc_id]
        fingerprint = document.fingerprint
        
        # استرجاع المحتوى الأصلي
        original_content = self.retrieve_secure_document(doc_id)
        if not original_content:
            return False
        
        # حساب hash جديد
        current_hash = hashlib.sha256(original_content).hexdigest()
        
        # مقارنة مع البصمة الأصلية
        if current_hash != fingerprint.content_hash:
            logger.error(f"تم تعديل المستند: {doc_id}")
            return False
        
        # التحقق من التوقيع الرقمي
        signature_data = f"{fingerprint.content_hash}{fingerprint.timestamp.isoformat()}{fingerprint.creator}".encode()
        is_signature_valid = self.crypto_manager.verify_digital_signature(
            signature_data, 
            fingerprint.signature
        )
        
        if not is_signature_valid:
            logger.error(f"التوقيع الرقمي غير صحيح: {doc_id}")
            return False
        
        logger.info(f"تم التحقق من سلامة المستند: {doc_id}")
        return True
    
    def _create_digital_fingerprint(self, content: bytes, doc_id: str) -> DigitalFingerprint:
        """إنشاء بصمة رقمية"""
        # حساب hash المحتوى
        content_hash = hashlib.sha256(content).hexdigest()
        
        # إنشاء معرف البصمة
        fingerprint_id = hashlib.sha256(f"{doc_id}{content_hash}".encode()).hexdigest()
        
        # إنشاء التوقيت
        timestamp = datetime.now()
        
        # إنشاء التوقيع الرقمي
        signature_data = f"{content_hash}{timestamp.isoformat()}system".encode()
        signature = self.crypto_manager.create_digital_signature(signature_data)
        
        # معلومات إضافية للتحقق
        verification_data = {
            "algorithm": "SHA-256",
            "signature_algorithm": "RSA-PSS",
            "key_size": 4096
        }
        
        return DigitalFingerprint(
            id=fingerprint_id,
            content_hash=content_hash,
            signature=signature,
            timestamp=timestamp,
            creator="system",
            metadata={"document_id": doc_id},
            verification_data=verification_data
        )

class InfrastructurePP:
    """البنية التحتية P&P الرئيسية"""
    
    def __init__(self, config_path: str = "infrastructure_config.json"):
        self.config = self._load_config(config_path)
        
        # تهيئة المكونات
        self.crypto_manager = CryptographyManager()
        self.ipfs_manager = IPFSManager(self.config.get("ipfs_api_url", "http://127.0.0.1:5001"))
        self.network_manager = AnonymousNetworkManager()
        self.document_manager = DocumentManager(self.crypto_manager, self.ipfs_manager)
        
        # سجل العقد
        self.network_nodes: Dict[str, NetworkNode] = {}
        
        # إحصائيات
        self.stats = {
            "documents_created": 0,
            "documents_retrieved": 0,
            "anonymous_requests": 0,
            "encryption_operations": 0
        }
        
        logger.info("تم تهيئة البنية التحتية P&P")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """تحميل التكوين"""
        default_config = {
            "ipfs_api_url": "http://127.0.0.1:5001",
            "enable_tor": True,
            "enable_i2p": False,
            "default_security_level": "medium",
            "document_expiry_days": 365,
            "max_document_size": 10485760  # 10MB
        }
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return {**default_config, **config}
        except FileNotFoundError:
            logger.warning(f"ملف التكوين غير موجود: {config_path}. استخدام التكوين الافتراضي.")
            return default_config
    
    # واجهات برمجية للتفاعل مع المكونات الأخرى
    
    def store_secure_data(self, data: Union[str, bytes], 
                         security_level: str = "medium") -> str:
        """تخزين بيانات آمنة"""
        try:
            level = SecurityLevel[security_level.upper()]
            document = self.document_manager.create_secure_document(
                data, 
                level, 
                self.config["document_expiry_days"]
            )
            
            self.stats["documents_created"] += 1
            self.stats["encryption_operations"] += 1
            
            return document.id
        
        except Exception as e:
            logger.error(f"فشل في تخزين البيانات الآمنة: {e}")
            raise
    
    def retrieve_secure_data(self, document_id: str) -> Optional[bytes]:
        """استرجاع بيانات آمنة"""
        try:
            data = self.document_manager.retrieve_secure_document(document_id)
            
            if data:
                self.stats["documents_retrieved"] += 1
                self.stats["encryption_operations"] += 1
            
            return data
        
        except Exception as e:
            logger.error(f"فشل في استرجاع البيانات الآمنة: {e}")
            return None
    
    def verify_data_integrity(self, document_id: str) -> bool:
        """التحقق من سلامة البيانات"""
        return self.document_manager.verify_document_integrity(document_id)
    
    def make_anonymous_request(self, url: str, network: str = "tor", 
                             method: str = "GET", data: Optional[Dict] = None) -> Optional[Dict]:
        """إجراء طلب مجهول"""
        try:
            network_type = NetworkType[network.upper()]
            response = self.network_manager.make_anonymous_request(url, network_type, method, data)
            
            if response:
                self.stats["anonymous_requests"] += 1
                return {
                    "status_code": response.status_code,
                    "content": response.text,
                    "headers": dict(response.headers)
                }
            
            return None
        
        except Exception as e:
            logger.error(f"فشل في الطلب المجهول: {e}")
            return None
    
    def encrypt_data(self, data: Union[str, bytes], encryption_type: str = "hybrid") -> Dict[str, Any]:
        """تشفير البيانات"""
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            enc_type = EncryptionType[encryption_type.upper()]
            result = self.crypto_manager.encrypt_data(data, enc_type)
            
            self.stats["encryption_operations"] += 1
            
            return result
        
        except Exception as e:
            logger.error(f"فشل في تشفير البيانات: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: Dict[str, Any]) -> bytes:
        """فك تشفير البيانات"""
        try:
            result = self.crypto_manager.decrypt_data(encrypted_data)
            
            self.stats["encryption_operations"] += 1
            
            return result
        
        except Exception as e:
            logger.error(f"فشل في فك تشفير البيانات: {e}")
            raise
    
    def create_digital_signature(self, data: Union[str, bytes]) -> str:
        """إنشاء توقيع رقمي"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return self.crypto_manager.create_digital_signature(data)
    
    def verify_digital_signature(self, data: Union[str, bytes], signature: str) -> bool:
        """التحقق من التوقيع الرقمي"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return self.crypto_manager.verify_digital_signature(data, signature)
    
    def get_system_status(self) -> Dict[str, Any]:
        """الحصول على حالة النظام"""
        return {
            "crypto_manager": {
                "keys_initialized": self.crypto_manager.system_private_key is not None
            },
            "ipfs_manager": {
                "available": self.ipfs_manager.is_available
            },
            "network_manager": {
                "tor_available": self.network_manager.tor_proxy is not None,
                "i2p_available": self.network_manager.i2p_proxy is not None
            },
            "statistics": self.stats,
            "documents_count": len(self.document_manager.documents),
            "fingerprints_count": len(self.document_manager.fingerprints)
        }

def main():
    """دالة الاختبار الرئيسية"""
    # إنشاء البنية التحتية
    infrastructure = InfrastructurePP()
    
    try:
        # اختبار تخزين واسترجاع البيانات الآمنة
        print("اختبار تخزين البيانات الآمنة...")
        
        test_data = "هذه بيانات سرية للغاية تحتاج إلى حماية قصوى"
        doc_id = infrastructure.store_secure_data(test_data, "high")
        print(f"تم تخزين البيانات: {doc_id}")
        
        # استرجاع البيانات
        retrieved_data = infrastructure.retrieve_secure_data(doc_id)
        if retrieved_data:
            print(f"تم استرجاع البيانات: {retrieved_data.decode('utf-8')}")
        
        # التحقق من السلامة
        is_valid = infrastructure.verify_data_integrity(doc_id)
        print(f"سلامة البيانات: {is_valid}")
        
        # اختبار التشفير المباشر
        print("\nاختبار التشفير المباشر...")
        
        encrypted = infrastructure.encrypt_data("رسالة سرية", "aes_256")
        print(f"تم التشفير: {encrypted['encryption_type']}")
        
        decrypted = infrastructure.decrypt_data(encrypted)
        print(f"تم فك التشفير: {decrypted.decode('utf-8')}")
        
        # اختبار التوقيع الرقمي
        print("\nاختبار التوقيع الرقمي...")
        
        message = "رسالة مهمة تحتاج توقيع"
        signature = infrastructure.create_digital_signature(message)
        print(f"تم إنشاء التوقيع: {signature[:50]}...")
        
        is_signature_valid = infrastructure.verify_digital_signature(message, signature)
        print(f"صحة التوقيع: {is_signature_valid}")
        
        # عرض حالة النظام
        print("\nحالة النظام:")
        status = infrastructure.get_system_status()
        for key, value in status.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        print(f"خطأ في الاختبار: {e}")

if __name__ == "__main__":
    main()

