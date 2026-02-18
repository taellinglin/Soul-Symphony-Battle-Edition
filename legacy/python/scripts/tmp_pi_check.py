import paramiko

HOST = "192.168.0.2"
USER = "user"
PASSWORD = "Clefable64"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASSWORD, timeout=10)

checks = [
    "curl -I -m 8 http://192.168.0.3:4747/ | sed -n '1,10p'",
    "curl -kI -H 'Host: bank.linglin.art' https://127.0.0.1/ | sed -n '1,20p'",
]

for cmd in checks:
    print(f"\\n$ {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    print(stdout.read().decode("utf-8", errors="replace"))
    err = stderr.read().decode("utf-8", errors="replace")
    if err.strip():
        print(err)

ssh.close()
