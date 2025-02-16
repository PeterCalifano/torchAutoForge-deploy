#!bin/bash
# Script to build cppprojects in GNU/Linux systems
# NOTE: this script assumes to be in the project root folder.
# CHANGELOG: 
# - Created January 2024,- modified May 2024 for Ubuntu 24.04 TLS by PeterC.
# - Last updated with shell parser by PeterC, July 2024
# - Major rework for cpp_template_project by PeterC, January 2025

# Default values
buildpath="build" 
use_default_buildpath=true
jobs=3
rebuild_only=false
build_type=relwithdebinfo # default build type, possible options: debug, release, relwithdebinfo, minsizerel
add_checks=true
add_legacy_tests=false
CXX_FLAGS=""
python_wrap=false
matlab_wrap=false
install=false
ninja_build=false

# Parse options using getopt
# NOTE: no ":" after option means no argument, ":" means required argument, "::" means optional argument
OPTIONS=B::,j::,i,r,t::,c,f::,p,m,n
LONGOPTIONS=buildpath::,jobs::,install,rebuild_only,type-build::,checks,flagsCXX::,python-wrap,matlab-wrap,ninja-build

# Parsed arguments list with getopt
PARSED=$(getopt --options ${OPTIONS} --longoptions ${LONGOPTIONS} --name "$0" -- "$@") 
# TODO check if this is where I need to modify something to allow things like -B build, instead of -Bbuild

# Check validity of input arguments 
if [[ $? -ne 0 ]]; then
  # e.g. $? == 1
  #  then getopt has complained about wrong arguments to stdout
  exit 2
fi

# Parse arguments
eval set -- "$PARSED"

# Process options (change default values if needed)
while true; do
  case "$1" in
    -B|-b|--buildpath)
      if [ -n "$2" ] && [ "$2" != "--" ]; then # Check how many args (if 2)
        buildpath="$2"
        use_default_buildpath=false
        shift 2 # Shift of two args, i.e. $1 will then point to the next argument
      else 
      # Handle the default case (no optional argument provided), thus shift of 1
        buildpath="build"
        use_default_buildpath=true
        shift
      fi
      ;;
    -j|--jobs)
      if [ -n "$2" ] && [ "$2" != "--" ]; then
        jobs="$2"
        shift 2
      else
        jobs=4
        shift
      fi
      ;;
    -r|--rebuild_only)
      rebuild_only=true
      shift
      ;;
    -t|--type-build)
      if [ -n "$2" ] && [ "$2" != "--" ]; then
        build_type="$2"
        if [ "${build_type}" == "debug" | "${build_type}" == "relwithdebinfo"]; then
          CXX_FLAGS="${CXX_FLAGS} -Wall -Wextra"
        fi
        shift 2
      else
        build_type=relwithdebinfo
        CXX_FLAGS="${CXX_FLAGS} -Wall -Wextra"
        shift
      fi
      ;;
    -c|--checks)
      add_checks=true
      shift
      ;;
    -f|--flagsCXX)
      if [ -n "$2" ] && [ "$2" != "--" ]; then
        CXX_FLAGS="$2"
        shift 2
      else 
        CXX_FLAGS="${CXX_FLAGS}"
        shift
      fi
      ;;
    -p|--python-wrap)  
        python_wrap=true
        shift
      ;;
    -m|--matlab-wrap)  
        matlab_wrap=true
        shift
      ;;
    -i|--install)
      install=true
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Not a valid option: $1" >&2
      exit 3
      ;;
  esac
done

# Export path to use GCC 11.4 instead of >13.0
#export CC=/usr/bin/gcc-11
#export CXX=/usr/bin/g++-11

# Enforce tests if build type is release
if [ "${build_type}" == "release" ]; then
  add_checks=true
fi

# Handle wrap requirements
#if [ "${python_wrap}" == true ] || [ "${matlab_wrap}" == true ]; #then
  # TODO add gtwrap repository/tools to the project or ensure they are present
#fi


if [ "${rebuild_only}" == false ]; then

  # Rebuild using existing buildpath
  echo "BUILDING project with options..."
  echo -e "\tBuildpath: $buildpath"
  echo -e "\tNum of jobs: $jobs"
  echo -e "\tBuild Type: ${build_type}"
  echo -e "\tEnforced compile flags: ${CXX_FLAGS}"
  echo -e "\tPython wrapper build: ${python_wrap}"
  echo -e "\tMATLAB wrapper build: ${matlab_wrap}"

  sleep 0.5

  BUILD_CONFIG_COMMANDS=" -DCMAKE_CXX_FLAGS=${CXX_FLAGS} \
                          -DCMAKE_C_FLAGS=${CXX_FLAGS} \
                          -DCMAKE_BUILD_TYPE=${build_type}"



  # Append additional build options
  if [ "${ninja_build}" == true ]; then
    BUILD_CONFIG_COMMANDS+=" -GNinja"
  fi

  if [ "$python_wrap" == true ]; then
    BUILD_CONFIG_COMMANDS+=" -DBUILD_PYTHON_WRAPPER=True"
  fi

  if [ "$matlab_wrap" == true ]; then
    BUILD_CONFIG_COMMANDS+=" -DBUILD_MATLAB_WRAPPER=True"
  fi

  #BUILD_CONFIG_COMMANDS+=""

  # Generate build configuration
    cmake -B ${buildpath} -S . ${BUILD_CONFIG_COMMANDS} 
    #-DGTSAM_BUILD_PYTHON=${python_wrap} \
    #-DGTSAM_INSTALL_MATLAB_TOOLBOX=${matlab_wrap} \

  # TODO introduce check to stop build script if configuration fails!
  
  # Build and optionally, install 
    make -j ${jobs} -C ${buildpath} 

    # Build and optionally, install 
    make -j ${jobs} -C ${buildpath} 

    if [ "${add_checks}" == true ] || [ "${install}" == true ]; then
      if [ "${add_legacy_tests}" == true ]; then
        make check -j ${jobs} -C ${buildpath} 
      fi
      make test -j ${jobs} -C ${buildpath} 
    fi
    
    if [ "${install}" == true ]; then
      sudo make install -j ${jobs} -C ${buildpath} 
    fi

else 

  # Rebuilding using existing buildpath
  echo "REBUILDING project with options..."
  echo -e "\tBuildpath: $buildpath"
  echo -e "\tNum of jobs: $jobs"
  if ! [ -d $buildpath ]; then
      echo "ERROR: NO PREVIOUS BUILD FOUND! EXITING..." >&2
      exit 1
  fi

    # Build and optionally, install 
    make -j ${jobs} -C ${buildpath} 

    if [ "${add_checks}" == true ] || [ "${install}" == true ]; then
      make check -j ${jobs} -C ${buildpath} 
      make test -j ${jobs} -C ${buildpath} 
    fi

    if [ "${install}" == true ]; then
      sudo make install -j ${jobs} -C ${buildpath} 
    fi

fi