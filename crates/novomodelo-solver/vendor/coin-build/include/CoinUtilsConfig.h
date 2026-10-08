/* CoinUtilsConfig.h — vendored public config for the Cobre CMake superbuild.
 *
 * CoinUtils is an autotools project. In a normal build, the configure script
 * generates this header (recording which optional dependencies were detected).
 * Cobre builds CoinUtils via an authored CMake superbuild without running
 * autotools, so this header is hand-vendored with every optional dependency
 * DISABLED (no GLPK, no external Cholesky/AMD/MUMPS/WSSMP/TAUCS, no Osi, no
 * MPI, no zlib, no Sample/Netlib data, no Abc).
 *
 * It is intentionally self-contained: it defines the version and integer-type
 * macros that CoinUtils/Clp sources require, and leaves every COIN_HAS_*
 * optional-dependency macro UNDEFINED so that no source guarded by
 * `#ifdef COIN_HAS_*` attempts to use an absent library.
 *
 * Version pinned to the CoinUtils releases/2.11.13 submodule tag.
 */

#ifndef COBRE_VENDOR_COINUTILSCONFIG_H_INCLUDED
#define COBRE_VENDOR_COINUTILSCONFIG_H_INCLUDED

/* Version number of project. */
#define COINUTILS_VERSION "2.11.13"

/* Major Version number of project. */
#define COINUTILS_VERSION_MAJOR 2

/* Minor Version number of project. */
#define COINUTILS_VERSION_MINOR 11

/* Release Version number of project. */
#define COINUTILS_VERSION_RELEASE 13

/* 64-bit integer types and the pointer-capturing integer type.
 *
 * On MSVC the Windows SDK types are used (matching upstream
 * config_coinutils_default.h); everywhere else the C++ built-in types are
 * used directly, which is why COINUTILS_HAS_STDINT_H / COINUTILS_HAS_CSTDINT
 * are deliberately left undefined (no <stdint.h>/<cstdint> include is needed
 * to name `long long`).
 */
#if defined(_MSC_VER) && _MSC_VER >= 1200
#include <BaseTsd.h>
/* Define to 64bit integer type. */
#define COIN_INT64_T INT64
/* Define to 64bit unsigned integer type. */
#define COIN_UINT64_T UINT64
/* Define to integer type capturing pointer. */
#define COIN_INTPTR_T ULONG_PTR
#else
/* Define to 64bit integer type. */
#define COIN_INT64_T long long
/* Define to 64bit unsigned integer type. */
#define COIN_UINT64_T unsigned long long
/* Define to integer type capturing pointer. */
#define COIN_INTPTR_T int *
#endif

/* Standard C++/C library header availability.
 *
 * Upstream CoinUtils/Clp sources guard the inclusion of standard headers
 * behind autotools-detected HAVE_* macros (e.g. CoinFinite.cpp only does
 * `#include <cfloat>` when HAVE_CFLOAT is defined). An autotools `configure`
 * run would detect all of these and record them in the generated config
 * header. Because the Cobre superbuild does not run autotools, we record them
 * here. Every macro below names a header that the C++11 standard guarantees on
 * every platform Cobre supports, so they are defined unconditionally — no
 * platform-detection logic is needed.
 *
 * Deliberately NOT defined here: non-standard / platform-specific headers
 * (<ieeefp.h>, <endian.h>, <unistd.h>, <sys/stat.h>, <sys/types.h>,
 * <strings.h>, <dlfcn.h>, <windows.h>, the obsolete <memory.h>) and optional
 * third-party headers (<zlib.h>, <bzlib.h>, <readline/readline.h>). Sources
 * that probe those fall back to their standard-header branch above.
 */
#define HAVE_CASSERT 1
#define HAVE_CCTYPE 1
#define HAVE_CFLOAT 1
#define HAVE_CMATH 1
#define HAVE_CINTTYPES 1
#define HAVE_CSTDARG 1
#define HAVE_CSTDDEF 1
#define HAVE_CSTDINT 1
#define HAVE_CSTDIO 1
#define HAVE_CSTDLIB 1
#define HAVE_CSTRING 1
#define HAVE_CTIME 1
#define HAVE_ASSERT_H 1
#define HAVE_CTYPE_H 1
#define HAVE_FLOAT_H 1
#define HAVE_INTTYPES_H 1
#define HAVE_MATH_H 1
#define HAVE_STDARG_H 1
#define HAVE_STDINT_H 1
#define HAVE_STDIO_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STRING_H 1
#define HAVE_TIME_H 1

/* The following optional-dependency macros are deliberately LEFT UNDEFINED.
 * Do not define them: doing so would make CoinUtils sources attempt to use
 * libraries that are not part of the Cobre vendored build.
 *
 *   COIN_HAS_GLPK     COIN_HAS_MUMPS    COIN_HAS_WSSMP
 *   COIN_HAS_TAUCS    COIN_HAS_AMD      COIN_HAS_CHOLMOD
 *   COIN_HAS_ASL      COIN_HAS_NETLIB   COIN_HAS_SAMPLE
 *   COIN_HAS_BLAS     COIN_HAS_LAPACK   COIN_HAS_ZLIB
 *   COIN_HAS_BZLIB    COIN_HAS_READLINE
 */

#endif /* COBRE_VENDOR_COINUTILSCONFIG_H_INCLUDED */
