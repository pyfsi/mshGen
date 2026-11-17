'''
browser.py: headless Firefox driven through its Marionette protocol (no extra Python packages)

    with Browser(width, height) as b:
        b.open(url); b.js("return 1 + 1"); png = b.screenshot()
The Firefox profile lives in a temporary folder (removed afterwards); module library paths are removed from the
environment (they break Firefox on the cluster).
'''
import base64
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))


class Browser:

    def __init__(self, width=1600, height=1000, workdir=None):
        self.width, self.height = width, height
        self.prof = tempfile.mkdtemp(prefix="ffprof_", dir=workdir)
        self.port = 29000 + os.getpid() % 900
        with open(os.path.join(self.prof, "user.js"), "w") as f:
            f.write('user_pref("marionette.port", %d);\n' % self.port)
        env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
        env["HOME"] = self.prof
        self.proc = subprocess.Popen(["firefox", "--headless", "--marionette", "-no-remote", "-profile", self.prof],
                                     env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(90):
            try:
                self.sock = socket.create_connection(("127.0.0.1", self.port), timeout=600)
                break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError("Firefox (Marionette) did not start")
        self.id = 0
        self._read()
        self.cmd("WebDriver:NewSession", {"capabilities": {}})
        self.cmd("WebDriver:SetWindowRect", {"width": width, "height": height})
        self.page_js = open(os.path.join(HERE, "page.js")).read()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        try:
            self.proc.kill()
        finally:
            shutil.rmtree(self.prof, ignore_errors=True)

    def _read(self):
        buf = b""
        while b":" not in buf:
            buf += self.sock.recv(1)
        n, rest = buf.split(b":", 1)
        n = int(n)
        while len(rest) < n:
            rest += self.sock.recv(n - len(rest))
        return json.loads(rest)

    def cmd(self, name, params=None):
        self.id += 1
        msg = json.dumps([0, self.id, name, params or {}]).encode()
        self.sock.sendall(str(len(msg)).encode() + b":" + msg)
        r = self._read()
        if r[2]:
            raise RuntimeError("{}: {}".format(name, r[2]))
        return r[3]

    def open(self, url, wait=20):
        self.cmd("WebDriver:Navigate", {"url": url})
        time.sleep(wait)
        self.inject()

    def inject(self):
        self.js(self.page_js.replace("window.T = (function", "window.T = (function", 1) + "\nreturn true;")

    def js(self, code, *args):
        if "return" not in code:
            code = "return (" + code + ");"
        return self.cmd("WebDriver:ExecuteScript", {"script": code, "args": list(args)})["value"]

    def T(self, call):
        '''Run a call of the page helpers, e.g. T("T.set('Half-span (m)', 6)").'''
        return self.js("if (!window.T) {" + self.page_js + "} return " + call + ";")

    def screenshot(self):
        return base64.b64decode(self.cmd("WebDriver:TakeScreenshot", {"full": False})["value"])
