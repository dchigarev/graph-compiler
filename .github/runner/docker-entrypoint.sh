#!/bin/bash
set -e

: "${GH_TOKEN:?Pass GH_TOKEN with --env GH_TOKEN}"
[ -f /opt/intel/oneapi/setvars.sh ] || {
    echo "oneAPI installation missing" >&2
    exit 1
}

export http_proxy="${http_proxy:-${HTTP_PROXY:-}}"
export https_proxy="${https_proxy:-${HTTPS_PROXY:-}}"
export no_proxy="${no_proxy:-${NO_PROXY:-}}"
if [ -n "$http_proxy" ] || [ -n "$https_proxy" ]; then
    {
        [ -z "$http_proxy" ] || echo "Acquire::http::Proxy \"$http_proxy\";"
        [ -z "$https_proxy" ] || echo "Acquire::https::Proxy \"$https_proxy\";"
    } > /etc/apt/apt.conf.d/80proxy
fi

for device in /dev/dri/renderD*; do
    [ -e "$device" ] || continue
    gid=$(stat -c '%g' "$device")
    group=$(getent group "$gid" | cut -d: -f1)
    if [ -z "$group" ]; then
        group="render$gid"
        groupadd -g "$gid" "$group"
    fi
    usermod -aG "$group" runner
done

exec runuser -u runner -- env \
    HOME=/home/runner \
    GH_TOKEN="$GH_TOKEN" \
    GH_RUNNER_NAME="${GH_RUNNER_NAME:-}" \
    GH_RUNNER_LABELS="${GH_RUNNER_LABELS:-}" \
    HTTP_PROXY="${HTTP_PROXY:-}" HTTPS_PROXY="${HTTPS_PROXY:-}" NO_PROXY="${NO_PROXY:-}" \
    http_proxy="$http_proxy" https_proxy="$https_proxy" no_proxy="$no_proxy" \
    GH_RUNNER_LAUNCHER=exec \
    bash -c '. /home/runner/.bash_login'
