#!/usr/bin/env bash
# Build helper for CMake-based C++ projects (Linux)
# - Created Jan 2024; updated Aug 2025
# - Uses GNU getopt for long options
# - Generator-agnostic build via `cmake --build`

set -Eeuo pipefail
IFS=$'\n\t' # Narrows word splitting to newlines and tabs (safe with spaces)

# --- Defaults ---
buildpath="build"

jobs="${JOBS:-$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)}"
jobs=$(( jobs < 6 ? jobs : 6 ))

rebuild_only=false
build_type="relwithdebinfo"   # debug|release|relwithdebinfo|minsizerel
run_tests=true
CXX_FLAGS=""
python_wrap=false
matlab_wrap=false
install=false
use_ninja=false
no_optim=false
clean_first=false
toolchain_file=""
cmake_defines=()

# Helper function to print instructions
usage() {
  cat <<'USAGE'
Usage: build_lib.sh [OPTIONS]

Options:
  -B, --buildpath <dir>       Build directory (default: ./build)
  -j, --jobs <N>              Parallel build jobs (default: $(nproc or 4))
  -r, --rebuild-only          Skip CMake configure; build existing tree only
  -t, --type|--type-build <t> Build type: debug|release|relwithdebinfo|minsizerel
  -c, --checks                Run tests (on by default). Alias of --run-tests
      --skip-tests            Do not run tests
  -f, --flagsCXX <flags>      Extra C++ flags (quoted). Appends warnings for
                              Debug/RelWithDebInfo/Release
  -D, --define <var[=val]>    Extra CMake cache definitions (repeatable)
  -p, --python-wrap           Enable Python wrapper defaults (-DGTWRAP_BUILD_PYTHON_DEFAULT=ON)
  -m, --matlab-wrap           Enable MATLAB wrapper defaults (-DGTWRAP_BUILD_MATLAB_DEFAULT=ON)
  -i, --install               Run "install" target after tests
  -N, --ninja-build           Use Ninja generator (requires `ninja`)
  -n, --no-optim              Set -DNO_OPTIMIZATION=ON in the CMake cache
      --toolchain <file>      Pass CMake toolchain file (-DCMAKE_TOOLCHAIN_FILE=<file>)
      --clean                 Delete build dir before configuring
  -h, --help                  Show this help and exit

Examples:
  # Configure + build (RelWithDebInfo) into ./build
  ./build_lib.sh

  # Debug build with warnings, 8 jobs, and Ninja
  ./build_lib.sh -t debug -j 8 -N

  # Custom build dir and flags, run tests then install
  ./build_lib.sh -B out/release -t release -f "-march=native" -i
./build_lib.sh -DOPENCV_DIR=/opt/opencv -DENABLE_SOMETHING=ON

Notes:
  * Short options with arguments use a separate value: "-B build", "-j 8".
    For CMake defines, use "-DVAR=ON" or "-D VAR=ON".
  * This script requires GNU getopt (standard on Debian/Ubuntu).
USAGE
}

# Auxiliary functions
die()  { echo -e "\e[31mError:\e[0m $*" >&2; echo; usage; exit 2; } # Stop execution due to error
info() { echo -e "\e[34m[INFO]\e[0m $*"; } # Print info
trap 'echo -e "\e[31mBuild failed (line $LINENO).\e[0m"' ERR # Exit condition

# --- argument parsing (GNU getopt) ---
if ! command -v getopt > /dev/null 2>&1; then
  die "GNU getopt is required. On macOS: brew install gnu-getopt and adjust PATH."
fi

OPTIONS=B:j:rt:c:f:D:pmhNni
LONGOPTIONS=buildpath:,jobs:,rebuild-only,type:,type-build:,checks,flagsCXX:,define:,python-wrap,matlab-wrap,help,ninja-build,no-optim,skip-tests,clean,install,toolchain:
PARSED=$(getopt -o "$OPTIONS" -l "$LONGOPTIONS" -- "$@") || { usage; exit 2; }
eval set -- "$PARSED"

while true; do
  case "$1" in
    -B|--buildpath)       buildpath="$2"; shift 2 ;;
    -j|--jobs)            jobs="$2";     shift 2 ;;
    -r|--rebuild-only)    rebuild_only=true; shift ;;
    -t|--type|--type-build) build_type="$2"; shift 2 ;;
    -c|--checks)          run_tests=true;  shift ;;
        --skip-tests|--no-checks) run_tests=false; shift ;;
    -f|--flagsCXX)        CXX_FLAGS="$2"; shift 2 ;;
    -D|--define)          cmake_defines+=( "-D$2" ); shift 2 ;;
    -p|--python-wrap)     python_wrap=true; shift ;;
    -m|--matlab-wrap)     matlab_wrap=true; shift ;;
    -i|--install)         install=true;    shift ;;
    -N|--ninja-build)     use_ninja=true;  shift ;;
    -n|--no-optim)        no_optim=true;   shift ;;
        --toolchain)      toolchain_file="$2"; shift 2 ;;
        --clean)          clean_first=true; shift ;;
    -h|--help)            usage; exit 0 ;;
    --) shift; break ;;
     *) die "Unknown option: $1" ;;
  esac
done

# --- normalize & validate build type ---
bt="${build_type,,}"
case "$bt" in
  debug)          cmake_bt="Debug" ;;
  release)        cmake_bt="Release" ;;
  relwithdebinfo) cmake_bt="RelWithDebInfo" ;;
  minsizerel)     cmake_bt="MinSizeRel" ;;
  *) die "Invalid build type: $build_type" ;;
esac

# For common types, enforce warnings unless user already provided them
if [[ "$bt" =~ ^(debug|relwithdebinfo|release)$ ]]; then
  CXX_FLAGS="${CXX_FLAGS:+$CXX_FLAGS }-Wall -Wextra -Wpedantic"
fi

# Enforce tests for Release
if [[ "$cmake_bt" == "Release" ]]; then
  run_tests=true
fi

# Validate toolchain file if provided
if [[ -n "$toolchain_file" && ! -f "$toolchain_file" ]]; then
  die "Toolchain file not found: $toolchain_file"
fi

# Pre-build checks
command -v cmake >/dev/null 2>&1 || die "cmake not found"
if [[ "$use_ninja" == true ]]; then
  command -v ninja >/dev/null 2>&1 || die "Requested Ninja but 'ninja' not found"
fi

# Print info
info "Buildpath          : $buildpath"
info "Jobs               : $jobs"
info "Build Type         : $cmake_bt"
info "Extra CXX flags    : ${CXX_FLAGS:-<none>}"
info "Extra CMake defines: ${cmake_defines[*]:-<none>}"
info "Python wrapper     : $python_wrap"
info "MATLAB wrapper     : $matlab_wrap"
info "Generator          : $([[ "$use_ninja" == true ]] && echo Ninja || echo 'Unix Makefiles')"
info "Toolchain file     : ${toolchain_file:-<none>}"
info "Run tests          : $run_tests"
info "Install after build: $install"

sleep 0.2

# --- Configure ---
if [[ "$rebuild_only" == false ]]; then
  if [[ "$clean_first" == true && -d "$buildpath" ]]; then
    info "Removing existing build dir '$buildpath'"
    rm -rf -- "$buildpath"
  fi

  cmake_args=(
    -S .
    -B "$buildpath"
    "-DCMAKE_BUILD_TYPE=$cmake_bt"
    "-DCMAKE_CXX_FLAGS=$CXX_FLAGS"
    "-DCMAKE_C_FLAGS=$CXX_FLAGS"
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  )
  [[ "$use_ninja"  == true ]] && cmake_args+=( -G Ninja )
  [[ "$python_wrap" == true ]] && cmake_args+=( -DGTWRAP_BUILD_PYTHON_DEFAULT=ON )
  [[ "$matlab_wrap" == true ]] && cmake_args+=( -DGTWRAP_BUILD_MATLAB_DEFAULT=ON )
  [[ "$no_optim"   == true ]] && cmake_args+=( -DNO_OPTIMIZATION=ON )
  [[ -n "$toolchain_file" ]] && cmake_args+=( "-DCMAKE_TOOLCHAIN_FILE=$toolchain_file" )
  [[ ${#cmake_defines[@]} -gt 0 ]] && cmake_args+=( "${cmake_defines[@]}" )

  info "Configuring with CMake...\n"
  cmake "${cmake_args[@]}"
elif [[ -n "$toolchain_file" ]]; then
  info "Toolchain file provided, but --rebuild-only skips configure."
fi

# --- Build ---
info "\nBuilding..."
cmake --build "$buildpath" --parallel "$jobs"

# --- Test ---
if [[ "$run_tests" == true || "$install" == true ]]; then
  info "\nRunning tests..."
  ctest --test-dir "$buildpath" --output-on-failure -j "$jobs"
fi

# --- Install ---
if [[ "$install" == true ]]; then
  info "Installing..."
  cmake --build "$buildpath" --parallel "$jobs" --target install
fi

info "Done."
