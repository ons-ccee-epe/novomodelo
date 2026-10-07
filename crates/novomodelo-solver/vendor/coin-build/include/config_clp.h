/* config_clp.h — vendored private autotools config for Clp.
 *
 * This is the "private" config header that autotools would generate as
 * config_clp.h (the dependent-package variant of config.h). It records the
 * package version and the PACKAGE_* identification macros. Optional
 * dependencies and the Aboca (Abc) path are intentionally omitted so that
 * nothing guarded by `#ifdef CLP_HAS_ABC` / `#ifdef COIN_HAS_*` is enabled.
 *
 * The version macros are sourced from ClpConfig.h to keep a single source of
 * truth; only the autotools-private PACKAGE_* set is added here. Version
 * pinned to the Clp releases/1.17.11 submodule tag.
 */

#ifndef COBRE_VENDOR_CONFIG_CLP_H_INCLUDED
#define COBRE_VENDOR_CONFIG_CLP_H_INCLUDED

/* Version macros (CLP_VERSION, CLP_VERSION_MAJOR, ...). */
#include "ClpConfig.h"

/* The generic autotools PACKAGE and VERSION macros are guarded with #ifndef
 * because a Clp translation unit may transitively include both this private
 * config and config_coinutils.h, which defines the same generic names with the
 * CoinUtils values. In a native autotools build only one package config header
 * is active per translation unit; the guards reproduce that "first one wins"
 * behavior and avoid redefinition warnings. The package-specific CLP_VERSION
 * and COINUTILS_VERSION macros above remain distinct and unaffected. */

/* Name of package. */
#ifndef PACKAGE
#define PACKAGE "clp"
#endif

/* Define to the address where bug reports for this package should be sent. */
#ifndef PACKAGE_BUGREPORT
#define PACKAGE_BUGREPORT "coin-discuss@list.coin-or.org"
#endif

/* Define to the full name of this package. */
#ifndef PACKAGE_NAME
#define PACKAGE_NAME "Clp"
#endif

/* Define to the full name and version of this package. */
#ifndef PACKAGE_STRING
#define PACKAGE_STRING "Clp 1.17.11"
#endif

/* Define to the one symbol short name of this package. */
#ifndef PACKAGE_TARNAME
#define PACKAGE_TARNAME "clp"
#endif

/* Define to the version of this package. */
#ifndef PACKAGE_VERSION
#define PACKAGE_VERSION "1.17.11"
#endif

/* Version number of package. */
#ifndef VERSION
#define VERSION "1.17.11"
#endif

#endif /* COBRE_VENDOR_CONFIG_CLP_H_INCLUDED */
