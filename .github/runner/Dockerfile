# Find the token in the repo Settings > Actions > Runners > New self-hosted runner.
# GH_TOKEN=<Mandatory GH Token>
# IMG_NAME=gc-runner
# RUNNER_NAME=gc-$(hostname)
#
# docker build \
#   -t $IMG_NAME \
#   --build-arg HTTP_PROXY="$HTTP_PROXY" \
#   --build-arg HTTPS_PROXY="$HTTPS_PROXY" \
#   runner
#
# docker run -d \
#   --name $RUNNER_NAME \
#   --hostname $RUNNER_NAME \
#   --restart unless-stopped \
#   --device=/dev/dri:/dev/dri \
#   -e GH_TOKEN="$GH_TOKEN" \
#   -e HTTP_PROXY="$HTTP_PROXY" \
#   -e HTTPS_PROXY="$HTTPS_PROXY" \
#   -e NO_PROXY="$NO_PROXY" \
#   $IMG_NAME
#
# docker logs -f $RUNNER_NAME
# docker exec -it --user runner $RUNNER_NAME bash

FROM intel/oneapi-toolkit:latest

ENV DEBIAN_FRONTEND=noninteractive

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN export \
    http_proxy="${http_proxy:-${HTTP_PROXY:-}}" \
    https_proxy="${https_proxy:-${HTTPS_PROXY:-}}" \
    no_proxy="${no_proxy:-${NO_PROXY:-}}" \
    HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-}}" \
    HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-}}" \
    NO_PROXY="${NO_PROXY:-${no_proxy:-}}" \
 && apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    ocl-icd-libopencl1 \
    sudo \
 && curl -fsSL https://repositories.intel.com/gpu/intel-graphics.key \
    | gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg \
 && echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble unified' \
    > /etc/apt/sources.list.d/intel-gpu.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    intel-ocloc \
    intel-opencl-icd \
    libze-intel-gpu1 \
    libze1 \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash runner \
 && echo 'runner ALL=(ALL) NOPASSWD: ALL' > /etc/sudoers.d/runner \
 && chmod 0440 /etc/sudoers.d/runner

COPY .bash_login /home/runner/.bash_login
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
 && chown runner:runner /home/runner/.bash_login

WORKDIR /home/runner
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
