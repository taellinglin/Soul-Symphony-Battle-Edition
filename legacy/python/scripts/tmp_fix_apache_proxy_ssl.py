import paramiko

HOST = "192.168.0.2"
USER = "user"
PASSWORD = "Clefable64"

FILES = [
    "/etc/apache2/sites-enabled/bank.linglin.art.conf",
    "/etc/apache2/sites-enabled/bank.linglin.art-le-ssl.conf",
]


def run(ssh: paramiko.SSHClient, cmd: str):
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=10)

    print("=== Current vhost files ===")
    for path in FILES:
        code, out, err = run(ssh, f"test -f {path} && echo FOUND || echo MISSING")
        print(path, out.strip())

    print("\n=== Current content (head) ===")
    for path in FILES:
        print(f"\n---- {path} ----")
        code, out, err = run(ssh, f"sed -n '1,220p' {path}")
        print(out)
        if err.strip():
            print("ERR:", err)

    ssh.close()


if __name__ == "__main__":
    main()
