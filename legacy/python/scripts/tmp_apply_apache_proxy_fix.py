import datetime
import paramiko

HOST = "192.168.0.2"
USER = "user"
PASSWORD = "Clefable64"
TARGET = "/etc/apache2/sites-enabled/bank.linglin.art-le-ssl.conf"


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

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{TARGET}.bak_{stamp}"

    cmds = [
        f"cp {TARGET} {backup}",
        f"sed -i 's#ProxyPass / http://192.168.0.3:5001/#ProxyPass / http://192.168.0.3:4747/#g' {TARGET}",
        f"sed -i 's#ProxyPassReverse / http://192.168.0.3:5001/#ProxyPassReverse / http://192.168.0.3:4747/#g' {TARGET}",
        "apache2ctl -t",
        "systemctl reload apache2",
    ]

    for cmd in cmds:
        code, out, err = sudo_run(ssh, cmd)
        print(f"$ {cmd}")
        print(f"exit={code}")
        if out.strip():
            print(out)
        if err.strip():
            print(err)
        if code != 0:
            raise SystemExit(code)

    print("\n=== Updated file snippet ===")
    code, out, err = run(ssh, f"sed -n '1,220p' {TARGET}")
    print(out)

    print("\n=== Backend check from Pi ===")
    code, out, err = run(ssh, "curl -I -m 8 http://192.168.0.3:4747/ | sed -n '1,5p'")
    print(out)
    if err.strip():
        print(err)

    print("\n=== VHost check on Pi localhost ===")
    code, out, err = run(ssh, "curl -kI -H 'Host: bank.linglin.art' https://127.0.0.1/ | sed -n '1,10p'")
    print(out)
    if err.strip():
        print(err)

    ssh.close()


if __name__ == "__main__":
    main()
