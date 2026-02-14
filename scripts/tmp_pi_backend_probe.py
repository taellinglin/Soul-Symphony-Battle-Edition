import paramiko

HOST = "192.168.0.2"
USER = "user"
PASSWORD = "Clefable64"

TARGETS = [
    "http://172.31.144.1:4747/",
    "http://172.31.146.164:4747/",
    "http://192.168.0.3:4747/",
    "http://192.168.0.32:4747/",
]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASSWORD, timeout=10)

for url in TARGETS:
    cmd = f"echo '=== {url} ==='; curl -I -m 6 {url} | sed -n '1,8p'"
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    print(out)
    if err.strip():
        print(err)

ssh.close()
