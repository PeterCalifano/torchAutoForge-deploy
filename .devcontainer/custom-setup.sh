#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

if ! command -v apt-get >/dev/null 2>&1; then
  echo "custom-setup.sh: apt-get not found, skipping package install."
  exit 0
fi

apt-get update
apt-get upgrade -y
apt-get install -y \
  build-essential \
  cmake \
  git \
  curl \
  wget \
  nano \
  pkg-config \
  gdb \
  python3 \
  python3-pip \
  libssl-dev \
  libffi-dev \
  software-properties-common \
  sudo \
  unzip \
  zip \
  man

apt-get install -y \
  gcc \
  g++ \
  gfortran \
  clang \
  gcc-mingw-w64

apt-get install -y \
  libboost-all-dev \
  libeigen3-dev \
  libsdl2-dev

valgrind_option="${INSTALL_VALGRIND:-on}"
case "${valgrind_option,,}" in
  false|off|0|no|disabled) build_valgrind=false ;;
  *) build_valgrind=true ;;
esac

if [[ "$build_valgrind" == true ]]; then
  valgrind_version="${VALGRIND_VERSION:-3.27.1}"
  valgrind_temp="$(mktemp -d)"
  set +e
  (
    set -e
    apt-get install -y bzip2 libc6-dbg
    curl -fsSL \
      "https://sourceware.org/pub/valgrind/valgrind-${valgrind_version}.tar.bz2" \
      -o "${valgrind_temp}/valgrind.tar.bz2"
    tar -xjf "${valgrind_temp}/valgrind.tar.bz2" -C "$valgrind_temp"
    cd "${valgrind_temp}/valgrind-${valgrind_version}"
    ./configure --prefix=/usr/local
    make -j"$(nproc)"
    make install
  )
  valgrind_status=$?
  set -e
  rm -rf "$valgrind_temp"
  if [[ "$valgrind_status" -ne 0 ]]; then
    echo "custom-setup.sh: valgrind build failed; continuing." >&2
  fi
fi

os_id=""
os_version=""
if [[ -r /etc/os-release ]]; then
  . /etc/os-release
  os_id="${ID:-}"
  os_version="${VERSION_ID:-}"
fi

if [[ "$os_id" == "ubuntu" && ( "$os_version" == "18.04" || "$os_version" == "20.04" ) ]]; then
  apt-get install -y qt5-default
else
  apt-get install -y qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools
fi

apt-get clean
rm -rf /var/lib/apt/lists/*
