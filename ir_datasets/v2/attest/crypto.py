"""Ed25519 signing for identities. Uses the ``cryptography`` package.

Keys travel as unpadded URL-safe base64 of the raw 32 bytes."""
import base64
import json
import os
from pathlib import Path


def _lib():
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives import serialization
    from cryptography.exceptions import InvalidSignature
    return ed25519, serialization, InvalidSignature


def _b64(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b'=').decode('ascii')


def _unb64(text):
    return base64.urlsafe_b64decode(text + '=' * (-len(text) % 4))


def generate_keypair():
    """``(private_key, public_key)`` as strings."""
    ed25519, ser, _ = _lib()
    key = ed25519.Ed25519PrivateKey.generate()
    raw = lambda k, **kw: k.public_bytes(ser.Encoding.Raw, ser.PublicFormat.Raw) if kw.get('pub') else \
        k.private_bytes(ser.Encoding.Raw, ser.PrivateFormat.Raw, ser.NoEncryption())
    return _b64(raw(key)), _b64(raw(key.public_key(), pub=True))


def sign(private_key, payload):
    ed25519, _, _ = _lib()
    return _b64(ed25519.Ed25519PrivateKey.from_private_bytes(_unb64(private_key)).sign(payload))


def verify(public_key, signature, payload):
    """Whether ``signature`` is valid for ``payload`` under ``public_key`` (never raises on bad input)."""
    ed25519, _, InvalidSignature = _lib()
    try:
        ed25519.Ed25519PublicKey.from_public_bytes(_unb64(public_key)).verify(_unb64(signature), payload)
        return True
    except (InvalidSignature, ValueError):
        return False


def attest_dir():
    """``<ir_datasets home>/attest`` (``~/.ir_datasets/attest``, or under ``$IR_DATASETS_HOME``)."""
    from ...util import home_path
    return home_path() / 'attest'


class Identity:
    """A registered identity: who you are on a given server, and the key that proves it.
    Stored as JSON in ``attest_dir()``, readable only by you."""

    def __init__(self, username, private_key, public_key, identity_id=None, server=None):
        self.username, self.private_key, self.public_key = username, private_key, public_key
        self.identity_id, self.server = identity_id, server

    @staticmethod
    def path():
        return attest_dir() / 'identity.json'

    def save(self):
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as f:
            json.dump(vars(self), f, indent=2)
        return path

    @classmethod
    def load(cls):
        """The saved identity, or ``None``."""
        try:
            return cls(**json.loads(cls.path().read_text()))
        except FileNotFoundError:
            return None

    def sign(self, payload):
        return sign(self.private_key, payload)
