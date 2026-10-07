/* ClpConfig.h — vendored public config for the Cobre CMake superbuild.
 *
 * Clp is an autotools project. In a normal build, the configure script
 * generates this header. Cobre builds Clp via an authored CMake superbuild
 * without running autotools, so this header is hand-vendored with every
 * optional dependency DISABLED and the Aboca (Abc) vectorized simplex path
 * excluded.
 *
 * It is intentionally self-contained: it defines only the Clp version macros
 * and leaves CLP_HAS_ABC plus every COIN_HAS_* optional-dependency macro
 * UNDEFINED so that no source guarded by `#ifdef CLP_HAS_ABC` / `#ifdef
 * COIN_HAS_*` attempts to use an absent feature or library.
 *
 * Version pinned to the Clp releases/1.17.11 submodule tag.
 */

#ifndef COBRE_VENDOR_CLPCONFIG_H_INCLUDED
#define COBRE_VENDOR_CLPCONFIG_H_INCLUDED

/* Version number of project. */
#define CLP_VERSION "1.17.11"

/* Major Version number of project. */
#define CLP_VERSION_MAJOR 1

/* Minor Version number of project. */
#define CLP_VERSION_MINOR 17

/* Release Version number of project. */
#define CLP_VERSION_RELEASE 11

/* Standard C++/C library header availability.
 *
 * Upstream Clp sources guard the inclusion of standard headers behind
 * autotools-detected HAVE_* macros. An autotools `configure` run would detect
 * these and record them in the generated config header; the Cobre superbuild
 * does not run autotools, so we record them here. Every macro below names a
 * header the C++11 standard guarantees on every platform Cobre supports, so
 * they are defined unconditionally. The `#ifndef` guards make this header
 * coexist cleanly with CoinUtilsConfig.h (which defines the same set) in
 * translation units that include both.
 *
 * Deliberately NOT defined: non-standard / platform-specific headers and
 * optional third-party headers — see the matching note in CoinUtilsConfig.h.
 */
#ifndef HAVE_CASSERT
#define HAVE_CASSERT 1
#endif
#ifndef HAVE_CCTYPE
#define HAVE_CCTYPE 1
#endif
#ifndef HAVE_CFLOAT
#define HAVE_CFLOAT 1
#endif
#ifndef HAVE_CMATH
#define HAVE_CMATH 1
#endif
#ifndef HAVE_CINTTYPES
#define HAVE_CINTTYPES 1
#endif
#ifndef HAVE_CSTDARG
#define HAVE_CSTDARG 1
#endif
#ifndef HAVE_CSTDDEF
#define HAVE_CSTDDEF 1
#endif
#ifndef HAVE_CSTDINT
#define HAVE_CSTDINT 1
#endif
#ifndef HAVE_CSTDIO
#define HAVE_CSTDIO 1
#endif
#ifndef HAVE_CSTDLIB
#define HAVE_CSTDLIB 1
#endif
#ifndef HAVE_CSTRING
#define HAVE_CSTRING 1
#endif
#ifndef HAVE_CTIME
#define HAVE_CTIME 1
#endif
#ifndef HAVE_ASSERT_H
#define HAVE_ASSERT_H 1
#endif
#ifndef HAVE_CTYPE_H
#define HAVE_CTYPE_H 1
#endif
#ifndef HAVE_FLOAT_H
#define HAVE_FLOAT_H 1
#endif
#ifndef HAVE_INTTYPES_H
#define HAVE_INTTYPES_H 1
#endif
#ifndef HAVE_MATH_H
#define HAVE_MATH_H 1
#endif
#ifndef HAVE_STDARG_H
#define HAVE_STDARG_H 1
#endif
#ifndef HAVE_STDINT_H
#define HAVE_STDINT_H 1
#endif
#ifndef HAVE_STDIO_H
#define HAVE_STDIO_H 1
#endif
#ifndef HAVE_STDLIB_H
#define HAVE_STDLIB_H 1
#endif
#ifndef HAVE_STRING_H
#define HAVE_STRING_H 1
#endif
#ifndef HAVE_TIME_H
#define HAVE_TIME_H 1
#endif

/* The following macros are deliberately LEFT UNDEFINED. Do not define them:
 * the Aboca vectorized simplex path is excluded, and the optional packages are
 * not part of the Cobre vendored build.
 *
 *   CLP_HAS_ABC
 *   COIN_HAS_GLPK     COIN_HAS_MUMPS    COIN_HAS_WSSMP
 *   COIN_HAS_TAUCS    COIN_HAS_AMD      COIN_HAS_CHOLMOD
 *   COIN_HAS_ASL      COIN_HAS_OSI      COIN_HAS_OSITESTS
 *   COIN_HAS_NETLIB   COIN_HAS_SAMPLE   COIN_HAS_WSMP
 *   COIN_HAS_BLAS     COIN_HAS_READLINE
 *
 * COIN_HAS_COINUTILS is also intentionally not defined here; CoinUtils is
 * always present in the vendored build, so the sources that need it are
 * compiled unconditionally by the superbuild.
 */

#endif /* COBRE_VENDOR_CLPCONFIG_H_INCLUDED */
