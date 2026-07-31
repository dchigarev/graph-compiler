#!/bin/sh
set -e

echo "Installing dependencies"
sudo apt-get update
sudo apt-get dist-upgrade -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -yq \
    binutils \
    ccache \
    clinfo \
    cmake \
    cmake-format \
    curl \
    gh \
    git \
    libigdfcl-dev \
    libomp-dev \
    libzstd-dev \
    ninja-build \
    ocl-icd-opencl-dev \
    psmisc \
    pkg-config \
    python-is-python3 \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-venv \
    screen \
    software-properties-common

export LLVM_VERSION=24
wget -qO- https://apt.llvm.org/llvm.sh \
  | sed -e "s|LLVM_VERSION_PATTERNS\[23\]|LLVM_VERSION_PATTERNS[${LLVM_VERSION}]|g" \
  | sudo bash -s -- "${LLVM_VERSION}" all
sudo apt purge -y clang lld llvm-14-linker-tools clang-format-14  libclang-cpp14 libclang-common-18-dev libllvm18 || true
sudo apt auto-remove -y
sudo apt-get install -y --no-install-recommends \
    libmlir-${LLVM_VERSION}-dev \
    mlir-${LLVM_VERSION}-tools

sudo update-alternatives --install /usr/bin/cc cc /usr/bin/clang-${LLVM_VERSION} 100
sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++-${LLVM_VERSION} 100
sudo update-alternatives --install /usr/bin/ld ld /usr/bin/lld-${LLVM_VERSION} 100
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-${LLVM_VERSION} 100
sudo update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-${LLVM_VERSION} 100
sudo update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-${LLVM_VERSION} 100
sudo update-alternatives --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-${LLVM_VERSION} 100
sudo update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-${LLVM_VERSION} 100

echo "Creating GitHub runner"
[ -z "$GH_TOKEN" ] && read -p "GitHub token: " GH_TOKEN
[ -z "$GH_REPO_PATH" ] && read -p "GitHub repository (owner/repo): " GH_REPO_PATH

: ${GH_RUNNER_DIR:="$HOME/actions-runner"}
: ${GH_RUNNER_NAME:="$(hostname)"}
: ${GH_RUNNER_LABELS:="$GH_RUNNER_NAME,main"}
GH_RUNNER_TOKEN="$(GH_TOKEN="$GH_TOKEN" gh api -X POST \
    "/repos/${GH_REPO_PATH}/actions/runners/registration-token" \
    --jq '.token')"

RUNNER_TMP_DIR="$(mktemp -d -p "$(dirname "$GH_RUNNER_DIR")")"
trap "rm -rf '$RUNNER_TMP_DIR'" EXIT KILL INT TERM

mkdir -p "$RUNNER_TMP_DIR"
cd "$RUNNER_TMP_DIR"
RUNNER_VERSION=$(curl -fsSL https://api.github.com/repos/actions/runner/releases/latest | grep -oP '"tag_name": "v\K[^"]+')
curl -o "actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" -L \
  "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
tar xzf "./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
rm "./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
./config.sh --url "https://github.com/$GH_REPO_PATH" --token "$GH_RUNNER_TOKEN" \
  --name "$GH_RUNNER_NAME" --labels "$GH_RUNNER_LABELS" \
  --replace --unattended

echo "GH_ACTIONS_CACHE_DIR=$GH_RUNNER_DIR/cache" >> .env
rm -rf "$GH_RUNNER_DIR"
mv "$RUNNER_TMP_DIR" "$GH_RUNNER_DIR"
