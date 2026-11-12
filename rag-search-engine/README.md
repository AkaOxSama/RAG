To start the uv environment use:
    cd rag-search-engine && source .venv/bin/activate

Disable IPv6:
sudo sysctl -w net.ipv6.conf.all.disable_ipv6=1
sudo sysctl -w net.ipv6.conf.default.disable_ipv6=1
bootdev run

Enable IPv6:
sudo sysctl -w net.ipv6.conf.all.disable_ipv6=0
sudo sysctl -w net.ipv6.conf.default.disable_ipv6=0

Force install BootDev versions:
GOPROXY=direct go install github.com/bootdotdev/bootdev@latest
bootdev --version