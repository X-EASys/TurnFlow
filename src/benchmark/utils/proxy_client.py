"""
Proxy Client Manager

Manages httpx clients configured with HTTP proxy for direct Gemini API access
via SSH tunnel. Provides thread-safe client creation and SSH tunnel lifecycle
management.

SSH Tunnel Setup:
    ssh -p 20096 -R 7897:127.0.0.1:7897 root@10.120.17.48

This reverse tunnel makes a local HTTP proxy (e.g., Clash at 127.0.0.1:7897)
available on the remote server, enabling direct access to Google's Gemini API
without going through a third-party relay.
"""

import os
import socket
import subprocess
import threading
import time
from typing import Dict, Optional, Tuple

from loguru import logger

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    LITELLM_HTTP_HANDLER_AVAILABLE = True
except ImportError:
    LITELLM_HTTP_HANDLER_AVAILABLE = False


class SSHTunnelManager:
    """
    Manages an SSH reverse tunnel for proxy access.

    The tunnel makes a local HTTP proxy (e.g., Clash at 127.0.0.1:7897)
    available on the remote server where the benchmark runs.
    """

    def __init__(
        self,
        ssh_host: str = "10.120.17.48",
        ssh_port: int = 20096,
        ssh_user: str = "root",
        local_proxy_port: int = 7897,
        remote_bind_port: int = 7897,
    ):
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        self.ssh_user = ssh_user
        self.local_proxy_port = local_proxy_port
        self.remote_bind_port = remote_bind_port
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._should_monitor = False

    def is_tunnel_active(self) -> bool:
        """Check if the SSH tunnel is active (process running or port open)."""
        if self._process is not None and self._process.poll() is None:
            return True
        return self._check_port_open()

    def _check_port_open(self) -> bool:
        """Check if the proxy port is accepting connections."""
        try:
            with socket.create_connection(
                ("127.0.0.1", self.remote_bind_port), timeout=2
            ):
                return True
        except (ConnectionRefusedError, OSError, socket.timeout):
            return False

    def start(self, auto_reconnect: bool = True) -> bool:
        """
        Start the SSH tunnel.

        Returns True if tunnel started successfully or was already running.
        """
        with self._lock:
            if self.is_tunnel_active():
                logger.info("SSH tunnel already active")
                return True

            cmd = [
                "ssh",
                "-p",
                str(self.ssh_port),
                "-R",
                f"{self.remote_bind_port}:127.0.0.1:{self.local_proxy_port}",
                "-N",
                "-o",
                "ServerAliveInterval=30",
                "-o",
                "ServerAliveCountMax=3",
                "-o",
                "ExitOnForwardFailure=yes",
                "-o",
                "StrictHostKeyChecking=accept-new",
                f"{self.ssh_user}@{self.ssh_host}",
            ]

            try:
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                time.sleep(2)
                if self._process.poll() is not None:
                    stderr = (
                        self._process.stderr.read().decode()
                        if self._process.stderr
                        else ""
                    )
                    logger.error(f"SSH tunnel failed to start: {stderr}")
                    return False

                logger.info(f"SSH tunnel started (PID {self._process.pid})")

                if auto_reconnect:
                    self._start_monitor()

                return True
            except Exception as e:
                logger.error(f"Failed to start SSH tunnel: {e}")
                return False

    def stop(self):
        """Stop the SSH tunnel."""
        self._should_monitor = False
        with self._lock:
            if self._process is not None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                self._process = None
                logger.info("SSH tunnel stopped")

    def _start_monitor(self):
        """Start a background thread to monitor and reconnect the tunnel."""
        self._should_monitor = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True
        )
        self._monitor_thread.start()

    def _monitor_loop(self):
        """Background monitor that auto-reconnects if tunnel drops."""
        while self._should_monitor:
            time.sleep(30)
            if self._should_monitor and not self.is_tunnel_active():
                logger.warning("SSH tunnel appears down, reconnecting...")
                self.start(auto_reconnect=False)

    def get_status(self) -> Dict:
        """Get tunnel status info."""
        return {
            "active": self.is_tunnel_active(),
            "ssh_host": self.ssh_host,
            "ssh_port": self.ssh_port,
            "proxy_port": self.remote_bind_port,
            "pid": self._process.pid if self._process else None,
            "monitoring": self._should_monitor,
        }


class ProxyClientManager:
    """
    Manages proxy-configured httpx clients for direct Gemini API access.
    Thread-safe singleton: creates one HTTPHandler per proxy_url, reuses across calls.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._clients: Dict[str, "HTTPHandler"] = {}
        self._client_lock = threading.Lock()
        self._tunnel_manager: Optional[SSHTunnelManager] = None

    @classmethod
    def get_instance(cls) -> "ProxyClientManager":
        """Singleton access."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.cleanup()
            cls._instance = None

    def get_proxy_client(self, proxy_url: str = "http://127.0.0.1:7897"):
        """
        Get (or create) an HTTPHandler with proxy configured.

        This client is passed as the `client` kwarg to litellm.completion(),
        which forwards it to the Gemini VertexLLM handler.

        Args:
            proxy_url: HTTP proxy URL (e.g., "http://127.0.0.1:7897")

        Returns:
            HTTPHandler with proxy-configured httpx.Client
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for proxy client")
        if not LITELLM_HTTP_HANDLER_AVAILABLE:
            raise RuntimeError(
                "litellm HTTPHandler is required for proxy client"
            )

        with self._client_lock:
            if proxy_url not in self._clients:
                httpx_client = httpx.Client(
                    proxy=proxy_url,
                    timeout=httpx.Timeout(600.0, connect=10.0),
                    follow_redirects=True,
                )
                handler = HTTPHandler(client=httpx_client)
                self._clients[proxy_url] = handler
                logger.info(f"Created proxy HTTP client: {proxy_url}")

            return self._clients[proxy_url]

    def get_tunnel_manager(self, **kwargs) -> SSHTunnelManager:
        """Get the SSH tunnel manager (lazy init)."""
        if self._tunnel_manager is None:
            self._tunnel_manager = SSHTunnelManager(**kwargs)
        return self._tunnel_manager

    def check_proxy_health(
        self, proxy_url: str = "http://127.0.0.1:7897"
    ) -> Tuple[bool, float, str]:
        """
        Check if the proxy can reach Google's API.

        Returns:
            Tuple of (success, latency_ms, message)
        """
        if not HTTPX_AVAILABLE:
            return False, 0, "httpx not available"

        try:
            with httpx.Client(
                proxy=proxy_url, timeout=httpx.Timeout(10.0)
            ) as client:
                start = time.time()
                response = client.get(
                    "https://generativelanguage.googleapis.com/"
                )
                latency = (time.time() - start) * 1000

            if response.status_code < 500:
                return (
                    True,
                    latency,
                    f"OK (status={response.status_code}, {latency:.0f}ms)",
                )
            else:
                return (
                    False,
                    latency,
                    f"Server error: {response.status_code}",
                )
        except Exception as e:
            return False, 0, f"Connection failed: {e}"

    def cleanup(self):
        """Close all clients and stop tunnel."""
        with self._client_lock:
            for handler in self._clients.values():
                try:
                    handler.close()
                except Exception:
                    pass
            self._clients.clear()

        if self._tunnel_manager:
            self._tunnel_manager.stop()
