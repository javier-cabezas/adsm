/*
 * Copyright (C) 2010 Lluís Vilanova <vilanova@ac.upc.edu>
 *
 * This file is part of libparatrace.
 *
 * Libparatrace is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */

#ifndef GMAC_CONFIG_VISIBILITY_H_
#define GMAC_CONFIG_VISIBILITY_H_

/* Generic helper definitions for shared library support */
#if defined _WIN32 || defined __CYGWIN__
#  define GMAC_DLL_IMPORT __declspec(dllimport)
#  define GMAC_DLL_EXPORT __declspec(dllexport)
#  define GMAC_DLL_LOCAL
#  define CONSTRUCTOR
#  define DESTRUCTOR
#  define APICALL __stdcall
#else
#  if __GNUC__ >= 4
#    define GMAC_DLL_IMPORT __attribute__ ((visibility("default")))
#    define GMAC_DLL_EXPORT __attribute__ ((visibility("default")))
#    define GMAC_DLL_LOCAL  __attribute__ ((visibility("hidden")))
#  else
#    define GMAC_DLL_IMPORT
#    define GMAC_DLL_EXPORT
#    define GMAC_DLL_LOCAL
#  endif
#  define CONSTRUCTOR __attribute__((constructor))
#  define DESTRUCTOR  __attribute__((destructor))
#  define APICALL
#endif

/* Now we use the generic helper definitions above to define GMAC_API and
 * GMAC_LOCAL.
 * GMAC_API is used for the public API symbols. It either DLL imports or DLL
 * exports (or does nothing for static build).
 * GMAC_LOCAL is used for non-api symbols.
 */

#if defined(GMAC_DLL)
/* compiled as a non-debug shared library */
#  ifdef GMAC_DLL_EXPORTS           /* defined if we're building the library
                                    * instead of using it                     */
#    define GMAC_API GMAC_DLL_EXPORT
#  else
#    define GMAC_API GMAC_DLL_IMPORT
#  endif  /* GMAC_DLL_EXPORTS */
#  define GMAC_LOCAL GMAC_DLL_LOCAL
#else  /* GMAC_DLL is not defined, meaning we're in a static library */
#  define GMAC_API
#  define GMAC_LOCAL
#endif  /* GMAC_DLL */



#endif  /* GMAC_CONFIG_VISIBILITY_H */
