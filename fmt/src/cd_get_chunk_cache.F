      SUBROUTINE CD_GET_CHUNK_CACHE (cache_size, cache_nelems, 
     .    cache_preemption, status)

*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
* Get current chunk cache settings. 

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP

* revision 0.0 - 1/2010

      INCLUDE 'netcdf4_settings.cmn'
      INCLUDE 'tmap_errors.parm'

! Declare netcdf function names. 
! The ones without trailing underscore are declared in netcdf.inc
#include "netcdf_declareUnderscore.inc"

* argument declarations:
      INTEGER cache_size, cache_nelems, cache_preemption, status

* local variable declarations:
      INTEGER cdfstat

C Get the cache size for the files to be opened in this session.

C Get chunk cache sizes.
#ifdef usingDODSf2cUnderscore
      cdfstat = NF_GET_CHUNK_CACHE_(cache_size, cache_nelems, 
     .     cache_preemption)
#else
      cdfstat = NF_GET_CHUNK_CACHE(cache_size, cache_nelems, 
     .     cache_preemption)
#endif

      IF (cdfstat .NE. NF_NOERR) GOTO 5100 

C If it hasnt been done yet, save the default value
      IF (default_cache_size .EQ. 0) default_cache_size = cache_size

      status = merr_ok
 5000 RETURN
 5100 CALL TM_ERRMSG (cdfstat+pcdferr, status, 'CD_SET_CHUNK_CACHE',
     .     no_descfile, no_stepfile, 
     .    'Getting chunk cache settings',
     .     no_errstring, *5000 )
      END
