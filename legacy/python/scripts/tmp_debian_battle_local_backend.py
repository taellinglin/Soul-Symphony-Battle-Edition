from pathlib import Path

import paramiko

HOST = "192.168.0.2"
USER = "user"
PASSWORD = "Clefable64"
SSL_CONF = "/etc/apache2/sites-enabled/battle.linglin.art-le-ssl.conf"
LOCAL_WEBBUILD = Path(r"C:\Users\User\Programs\Soul Symphony 2\webbuild")


def run(ssh, cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def sudo_run(ssh, cmd):
    stdin, stdout, stderr = ssh.exec_command(f"sudo -S bash -lc {cmd!r}")
    stdin.write(PASSWORD + "\n")
    stdin.flush()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def must(code, out, err, label):
    print(f"$ {label}\nexit={code}")
    if out.strip():
        print(out)
    if err.strip():
        print(err)
    if code != 0:
        raise SystemExit(code)


ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASSWORD, timeout=12)

# 1) ensure directory exists and sync minimal webbuild files
code, out, err = run(ssh, "mkdir -p /home/user/soul_webbuild")
must(code, out, err, "mkdir -p /home/user/soul_webbuild")

sftp = ssh.open_sftp()
for fn in ["index.html", "app.prc", "preload_manifest.txt"]:
    local_file = LOCAL_WEBBUILD / fn
    if not local_file.exists():
        raise SystemExit(f"Missing local file: {local_file}")
    sftp.put(str(local_file), f"/home/user/soul_webbuild/{fn}")
sftp.close()
print("Uploaded webbuild files to /home/user/soul_webbuild")

# 2) start backend locally on Debian
cmd = "nohup python3 -m http.server 4747 --bind 127.0.0.1 --directory /home/user/soul_webbuild >/tmp/soul4747.log 2>&1 & sleep 1; ss -ltnp | grep ':4747 ' || true"
code, out, err = run(ssh, cmd)
print(f"$ {cmd}\nexit={code}")
if out.strip():
    print(out)
if err.strip():
    print(err)

code, out, err = run(ssh, "ss -ltnp | grep ':4747 ' || true; curl -I -m 8 http://127.0.0.1:4747/ | sed -n '1,8p'")
must(code, out, err, "verify debian local 4747")

# 3) point battle vhost to local backend
for cmd in [
    f"cp {SSL_CONF} {SSL_CONF}.bak_local_backend",
    f"sed -i '/^\\s*ProxyPass /c\\    ProxyPass / http://127.0.0.1:4747/ connectiontimeout=3000 timeout=3000' {SSL_CONF}",
    f"sed -i '/^\\s*ProxyPassReverse /c\\    ProxyPassReverse / http://127.0.0.1:4747/' {SSL_CONF}",
    f"grep -nE 'ServerName|ProxyPass|ProxyPassReverse' {SSL_CONF}",
    "apache2ctl -t",
    "systemctl reload apache2",
]:
    code, out, err = sudo_run(ssh, cmd)
    must(code, out, err, cmd)

# 4) verify
for cmd in [
    "curl -I -m 8 http://127.0.0.1:4747/ | sed -n '1,8p'",
    "curl -kI -m 8 -H 'Host: battle.linglin.art' https://127.0.0.1/ | sed -n '1,12p'",
]:
    code, out, err = run(ssh, cmd)
    print(f"$ {cmd}\nexit={code}")
    if out.strip():
        print(out)
    if err.strip():
        print(err)

ssh.close()
