import paramiko

HOST='192.168.0.2'
USER='user'
PASSWORD='Clefable64'

ssh=paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASSWORD, timeout=10)

cmds=[
    "echo SHELL=$SHELL; whoami; python3 --version; command -v python3; ls -la /home/user/soul_webbuild | sed -n '1,20p'",
    "python3 -m http.server 4747 --bind 127.0.0.1 --directory /home/user/soul_webbuild >/tmp/soul4747.log 2>&1 & sleep 1; ss -ltnp | grep ':4747 ' || true; sed -n '1,40p' /tmp/soul4747.log",
    "curl -I -m 5 http://127.0.0.1:4747/ | sed -n '1,8p'",
]

for cmd in cmds:
    print(f"\n$ {cmd}")
    stdin,stdout,stderr=ssh.exec_command(cmd)
    out=stdout.read().decode('utf-8',errors='replace')
    err=stderr.read().decode('utf-8',errors='replace')
    code=stdout.channel.recv_exit_status()
    print(f"exit={code}")
    if out.strip(): print(out)
    if err.strip(): print(err)

ssh.close()
