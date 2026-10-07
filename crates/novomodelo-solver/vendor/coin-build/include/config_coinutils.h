/* config_coinutils.h — vendored private autotools config for CoinUtils.
 *
 * This is the "private" config header that autotools would generate as
 * config_coinutils.h (the dependent-package variant of config.h). It records
 * the package version and the PACKAGE_* identification macros. Optional
 * dependencies are intentionally omitted so that nothing guarded by
 * `#ifdef COIN_HAS_*` is enabled.
 *
 * The version/integer-type macros are sourced from CoinUtilsConfig.h to keep a
 * single source of truth; only the autotools-private PACKAGE_* set is added
 * here. Version pinned to the CoinUtils releases/2.11.13 submodule tag.
 */

#ifndef COBRE_VENDOR_CONFIG_COINUTILS_H_INCLUDED
#define COBRE_VENDOR_CONFIG_COINUTILS_H_INCLUDED

/* Version and integer-type macros (COINUTILS_VERSION*, COIN_INT64_T, ...). */
#include "CoinUtilsConfig.h"

/* The generic autotools PACKAGE and VERSION macros are guarded with #ifndef
 * because a Clp translation unit may transitively include both this private
 * config and config_clp.h, which defines the same generic names with the Clp
 * values. In a native autotools build only one package config header is active
 * per translation unit; the guards reproduce that "first one wins" behavior and
 * avoid redefinition warnings. The package-specific COINUTILS_VERSION and
 * CLP_VERSION macros above remain distinct and unaffected. */

/* Name of package. */
#ifndef PACKAGE
#define PACKAGE "coinutils"
#endif

/* Define to the address where bug reports for this package should be sent. */
#ifndef PACKAGE_BUGREPORT
#define PACKAGE_BUGREPORT "coin-discuss@list.coin-or.org"
#endif

/* Define to the full name of this package. */
#ifndef PACKAGE_NAME
#define PACKAGE_NAME "CoinUtils"
#endif

/* Define to the full name and version of this package. */
#ifndef PACKAGE_STRING
#define PACKAGE_STRING "CoinUtils 2.11.13"
#endif

/* Define to the one symbol short name of this package. */
#ifndef PACKAGE_TARNAME
#define PACKAGE_TARNAME "coinutils"
#endif

/* Define to the version of this package. */
#ifndef PACKAGE_VERSION
#define PACKAGE_VERSION "2.11.13"
#endif

/* Version number of package. */
#ifndef VERSION
#define VERSION "2.11.13"
#endif

#endif /* COBRE_VENDOR_CONFIG_COINUTILS_H_INCLUDED */
