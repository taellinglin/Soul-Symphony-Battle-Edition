import datetime
import paramiko

HOST = "192.168.0.2"
USER = "user"
PASSWORD = "Clefable64"
DOMAIN = "battle.linglin.art"
BACKEND = "http://192.168.0.3:4747/"


def run(ssh: paramiko.SSHClient, cmd: str):
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def sudo_run(ssh: paramiko.SSHClient, cmd: str):
    stdin, stdout, stderr = ssh.exec_command(f"sudo -S bash -lc {cmd!r}")
    stdin.write(PASSWORD + "\n")
    stdin.flush()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=10)

    # 1) Find files containing ServerName for domain
    code, out, err = run(
        ssh,
        "grep -RIl --include='*.conf' 'ServerName\\s\\+battle\\.linglin\\.art' /etc/apache2/sites-enabled /etc/apache2/sites-available 2>/dev/null || true",
    )
    files = [line.strip() for line in out.splitlines() if line.strip()]

    print("=== Matched vhost files ===")
    if not files:
        print("NO_MATCH")
        ssh.close()
        raise SystemExit(2)
    for path in files:
        print(path)

    # 2) Backup + update ProxyPass lines in each matched file
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in files:
        backup = f"{path}.bak_{stamp}"
        print(f"\n--- Updating {path} ---")
        cmds = [
            f"cp {path} {backup}",
            f"sed -i 's#^\\s*ProxyPass\\s\+/\\s\+http://[^ ]*#    ProxyPass / {BACKEND}#' {path}",
            f"sed -i 's#^\\s*ProxyPassReverse\\s\+/\\s\+http://[^ ]*#    ProxyPassReverse / {BACKEND}#' {path}",
            f"grep -nE 'ServerName|ProxyPass|ProxyPassReverse' {path}",
        ]
        for cmd in cmds:
            code, o, e = sudo_run(ssh, cmd)
            print(f"$ {cmd}\nexit={code}")
            if o.strip():
                print(o)
            if e.strip():
                print(e)
            if code != 0:
                ssh.close()
                raise SystemExit(code)

    # 3) Validate and reload apache
    for cmd in ["apache2ctl -t", "systemctl reload apache2"]:
        code, o, e = sudo_run(ssh, cmd)
        print(f"\n$ {cmd}\nexit={code}")
        if o.strip():
            print(o)
        if e.strip():
            print(e)
        if code != 0:
            ssh.close()
            raise SystemExit(code)

    # 4) Check from Pi side
    checks = [
        "curl -I -m 8 http://192.168.0.3:4747/ | sed -n '1,10p'",
        "curl -kI -H 'Host: battle.linglin.art' https://127.0.0.1/ | sed -n '1,20p'",
    ]
    for cmd in checks:
        code, o, e = run(ssh, cmd)
        print(f"\n$ {cmd}\nexit={code}")
        if o.strip():
            print(o)
        if e.strip():
            print(e)

    ssh.close()


if __name__ == "__main__":
    main()
