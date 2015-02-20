	SUBROUTINE GET_UVARS_LIST_BY_ATTNAME_AND_VAL
     .			(attname, attstring, maxlist,
     .			 id_list, dset_list, nfound  )

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
* return a list of all user variables that have a particular given value for
* a given string attribute
* intended usage:  find variables that have _SIGMA_REF_'="DEPTH"
* Note: if the incoming string is unspecified_name4, then all attrib vals match

* V690: 2/14 *sh* 
* V693+ 11/14 *sh* renaming 'sigma' as 'layerz' throughout

        IMPLICIT NONE
#include "tmap_dset.parm"
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'gfdl_vms.parm'
	include 'xvariables.cmn'
	include 'xrisc.cmn'

* calling argument declaration:
	CHARACTER*(*) attname, attstring
	INTEGER	maxlist, id_list(maxlist), dset_list(maxlist), nfound

* internal parameter declarations
	INTEGER	maxlen
	PARAMETER (maxlen = 50)

* internal variable declarations:
	LOGICAL NC_GET_ATTRIB,
     .		found_one, no_match_needed
	INTEGER	STR_CASE_BLIND_COMPARE,
     .		ivar, dset, status, varid, attid, attlen, attoutflag,
     .		num_indices, vax_code, dsetm1
	REAL	attvals(maxlen)
	CHARACTER retstring*(maxlen)

* initialize
	no_match_needed = attstring .EQ. unspecified_name4
	nfound = 0
	dsetm1 = -1
        CALL deleted_list_get_undel(uvar_num_items_head, 
     .                          deleted_list_result(1),
     .                          max_uvar, num_indices )

* loop through the user variables
	DO ivar = 1, num_indices

	   CALL CD_GET_VAR_ID (dsetm1, uvar_name_code(ivar), varid, status)
           found_one = NC_GET_ATTRIB ( dsetm1, varid, patnam_layerz,
     .                            .FALSE., uvar_name_code(ivar), maxlen,
     .                            attlen, attoutflag, retstring,
     .                            attvals )
* ... attrib name matches.
*     if necessary now also check if the contents match
           IF (found_one) THEN
	      found_one = no_match_needed
	      IF (.NOT.no_match_needed) THEN
	         vax_code = STR_CASE_BLIND_COMPARE(attstring,retstring)
	         found_one = vax_code .EQ. vms_str_success
	      ENDIF
	      IF (found_one) THEN
	         nfound = nfound + 1
	         id_list  (nfound) = varid
	         dset_list(nfound) = pdset_uvars  ! needs to be uvar_dset(ivar) for LET/D
	         IF (nfound .EQ. maxlist) RETURN  ! hacky but good enuf
	      ENDIF
	   ENDIF
	ENDDO

	RETURN
	END

