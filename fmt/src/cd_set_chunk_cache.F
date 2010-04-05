      SUBROUTINE CD_SET_CHUNK_CACHE (cache_size, cache_nelems, 
     .     cache_preemption, status)

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
* Set a new chunk cache size, nelems and/or preemption. User specifies 
* cache size in bytes, counts.  If any was unspecified it will come in with a negative
* value and should remain unchanged.
* If all arguments are -1, restore the default from the netcdf library.
* If any individual argument comes in as -1, use the current value;
* e.g. the user will probably set the cache size but leave other values 
* unchanged.

* From the netcdf-4 docs:
* sizep
*    The total size of the raw data chunk cache. 
* nelemsp
*    The number of chunk slots in the raw data chunk cache hash table 
* preemptionp
*    The preemtion value is between 0 and 1 inclusive and indicates 
*    how much chunks that have been fully read are favored for preemption. 
*    A value of zero means fully read chunks are treated no differently 
*    than other chunks (the preemption is strictly LRU) while a value of 
*    one means fully read chunks are always preempted before other chunks. 

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP

* revision 0.0 - 1/2010

* argument definitions:
*       cache_mbytes  - Cache size to set
*	status         - return status

      INCLUDE 'netcdf4_settings.cmn'
      INCLUDE 'tmap_errors.parm'

! Declare netcdf function names. 
! The ones without trailing underscore are declared in netcdf.inc
#include "netcdf_declareUnderscore.inc"

* argument declarations:
      REAL cache_mbytes
      INTEGER cache_size, cache_nelems, 
     .     cache_preemption, status

* local variable declarations:
      INTEGER cdfstat, c_size_in, c_nelems_in, c_preemp_in

C Set the cache size for the files created/opened in this session.

C Get current settings ( get cache_nelems, cache_preemption which 
C are not currently settable by the user.)

#ifdef usingDODSf2cUnderscore
      cdfstat = NF_GET_CHUNK_CACHE_(c_size_in, c_nelems_in, c_preemp_in)
#else
      cdfstat = NF_GET_CHUNK_CACHE(c_size_in, c_nelems_in, c_preemp_in)
#endif

      IF (cdfstat .NE. NF_NOERR) GOTO 5100 

C If it hasnt been done yet, save the default values
      IF (default_cache_size .EQ. 0) default_cache_size = c_size_in
      IF (default_nelems_size .EQ. 0) default_nelems_size = c_nelems_in
      IF (default_preemp_size .EQ. 0) default_preemp_size = c_preemp_in

C If this routine called with all the args set to -1, (cancel NCCACHE)
C then set to the library default sizes.

      IF (cache_size.LT.0 .AND. cache_nelems.LT.0 .AND. 
     .                          cache_preemption.LT.0) THEN
         cache_size = default_cache_size
         cache_nelems = default_nelems_size
         cache_preemption = default_preemp_size
      ENDIF

C If any individual argument comes in as -1, use the current value.

      IF (cache_size .LT. 0) cache_size = c_size_in
      IF (cache_nelems .LT. 0) cache_nelems = c_nelems_in
      IF (cache_preemption .LT. 0) cache_preemption = c_preemp_in

C Make the setting.

#ifdef usingDODSf2cUnderscore
      cdfstat = NF_SET_CHUNK_CACHE_(cache_size, cache_nelems, 
     .     cache_preemption)
#else 
      cdfstat = NF_SET_CHUNK_CACHE(cache_size, cache_nelems, 
     .     cache_preemption)
#endif

      IF (cdfstat .NE. NF_NOERR) GOTO 5100 

      status = merr_ok
 5000 RETURN
 5100 CALL TM_ERRMSG (cdfstat+pcdferr, status, 'CD_SET_CHUNK_CACHE',
     .     no_descfile, no_stepfile, 
     .    'Setting chunk cache size',
     .     no_errstring, *5000 )
      END

