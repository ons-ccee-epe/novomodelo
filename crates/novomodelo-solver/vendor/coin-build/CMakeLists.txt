# Superbuild for the vendored COIN-OR CoinUtils + Clp source trees.
#
# CoinUtils and Clp ship as autotools projects with no CMakeLists.txt. This
# file globs the upstream `src/*.cpp` sources, compiles two static libraries
# (CoinUtils, then Clp), and pulls in the hand-vendored config headers under
# `include/`. It is driven by `build.rs` through the `cmake` crate, the same
# mechanism the HiGHS build uses.
#
# Source layout note: the COIN-OR submodules use a nested directory layout, so
# the real source roots are one level deeper than the repository root:
#   CoinUtils sources -> ../CoinUtils/CoinUtils/src/*.cpp
#   Clp sources       -> ../Clp/Clp/src/*.cpp
#
# Optional dependencies (Osi, GLPK, the external Cholesky backends, the Abc
# vectorized simplex, and the AMPL/ASL command-line drivers) are deliberately
# excluded: enabling them would pull in absent third-party packages.

cmake_minimum_required(VERSION 3.15)
project(coin_build CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Honor BUILD_SHARED_LIBS=OFF for static-only output; default to static here.
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# Sibling source roots (nested COIN-OR layout) and the vendored config headers.
set(COINUTILS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../CoinUtils/CoinUtils/src")
set(CLP_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../Clp/Clp/src")
set(VENDOR_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")

# ---------------------------------------------------------------------------
# CoinUtils static library
# ---------------------------------------------------------------------------
file(GLOB COINUTILS_SOURCES "${COINUTILS_SRC}/*.cpp")

# Exclude optional GLPK and Abc sources (absent dependencies).
list(FILTER COINUTILS_SOURCES EXCLUDE REGEX ".*Glpk.*")
list(FILTER COINUTILS_SOURCES EXCLUDE REGEX ".*CoinAbc.*")

add_library(CoinUtils STATIC ${COINUTILS_SOURCES})
# Vendored include dir must precede the upstream src dir: the upstream public
# CoinUtilsConfig.h includes the private "config_coinutils.h" (quote form), and
# only the vendored copy of that private header carries the standard-library
# HAVE_* macros. HAVE_CONFIG_H selects that private-config path in the upstream
# public header (instead of config_coinutils_default.h, which omits them).
target_include_directories(CoinUtils PUBLIC
  "${VENDOR_INCLUDE}"
  "${COINUTILS_SRC}")
target_compile_definitions(CoinUtils PUBLIC HAVE_CONFIG_H)

# ---------------------------------------------------------------------------
# Clp static library
# ---------------------------------------------------------------------------
file(GLOB CLP_SOURCES "${CLP_SRC}/*.cpp")

# Exclude the Osi wrapper subdirectory sources (OsiClp*) — out of scope.
list(FILTER CLP_SOURCES EXCLUDE REGEX ".*OsiClp.*")
# Exclude optional GLPK interface sources.
list(FILTER CLP_SOURCES EXCLUDE REGEX ".*Glpk.*")
# Exclude external-Cholesky backends (Mumps/Wssmp/Taucs/Ufl) — absent deps.
list(REMOVE_ITEM CLP_SOURCES
  "${CLP_SRC}/ClpCholeskyMumps.cpp"
  "${CLP_SRC}/ClpCholeskyWssmp.cpp"
  "${CLP_SRC}/ClpCholeskyWssmpKKT.cpp"
  "${CLP_SRC}/ClpCholeskyTaucs.cpp"
  "${CLP_SRC}/ClpCholeskyUfl.cpp")
# Exclude the Abc vectorized simplex path: CoinAbc*, Clp*Abc*, and Abc* files.
list(FILTER CLP_SOURCES EXCLUDE REGEX ".*CoinAbc.*")
list(FILTER CLP_SOURCES EXCLUDE REGEX ".*Clp.*Abc.*")
list(FILTER CLP_SOURCES EXCLUDE REGEX ".*/Abc.*")
# Exclude the standalone `clp` command-line driver and its dependencies. These
# are not part of the Clp library or the C interface; they pull in optional
# packages absent from the vendored build (Osi, Cbc, GLPK, ASL/AMPL, readline,
# the Abc simplex) and were added to the exclusion list because they fail to
# compile without those dependencies:
#   - ClpSolver.cpp  : driver main(); includes glpk.h, Clp_ampl.h, Abc headers
#   - ClpMain.cpp    : driver entry; includes Abc simplex headers
#   - CbcOrClpParam.cpp : driver parameter parser; pulls Osi/Cbc/readline
#   - Clp_ampl.cpp   : AMPL/ASL solver driver; includes asl_pfgh.h (ASL)
#   - unitTest.cpp   : test harness; includes Abc factorization headers
list(REMOVE_ITEM CLP_SOURCES
  "${CLP_SRC}/ClpSolver.cpp"
  "${CLP_SRC}/ClpMain.cpp"
  "${CLP_SRC}/CbcOrClpParam.cpp"
  "${CLP_SRC}/Clp_ampl.cpp"
  "${CLP_SRC}/unitTest.cpp")

add_library(Clp STATIC ${CLP_SOURCES})
# Vendored include dir first (see CoinUtils note above): both ClpConfig.h and
# CoinUtilsConfig.h route through their vendored private configs under
# HAVE_CONFIG_H to pick up the standard-library HAVE_* macros.
target_include_directories(Clp PUBLIC
  "${VENDOR_INCLUDE}"
  "${CLP_SRC}"
  "${COINUTILS_SRC}")
target_compile_definitions(Clp PUBLIC HAVE_CONFIG_H)
target_link_libraries(Clp PUBLIC CoinUtils)

# ---------------------------------------------------------------------------
# Install rules
# ---------------------------------------------------------------------------
install(TARGETS CoinUtils Clp
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
install(FILES "${CLP_SRC}/Clp_C_Interface.h"
  DESTINATION include)
