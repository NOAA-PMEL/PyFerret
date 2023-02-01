	SUBROUTINE ERASE_DSET ( dset )

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
* delete all memory variables with data from a particular data set
* eliminate all context references to it, too

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/3/86
* V200:   6/9/89 - deleted variables flagged in mr_protected
*	10/25/89 - moved memory-resident purging to PURGE_DSET
* V411:  7/95 - Delete variables created with  LET/D=this_dset
* V601  11/06 - Also need to delete LET/D= variables from the linked-
*               list attribute structure.
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)
* V695+ *sh* 5/15 - free any c-cached arrays owned by the dataset
* V698  2/16 *acm* For ticket 2352: variables defined with LET/D are
*                  added to that dataset in the attribute structure. Extra
*                  argument to DELETE_USER_VAR

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xdset_info.cmn_text'

* calling argument declarations:
	INTEGER dset

* local variable declarations:
	INTEGER TM_LENSTR, mr, ii, slen, flen, dset_uvars

	PARAMETER (flen=128)      ! size of variable name string
	INTEGER*1 fhol(flen)      ! c-type Hollerith string buffer

* eliminate memory resident data
	CALL PURGE_DSET( dset )

* free any c-cached variables owned by this dataset
	DO ii = 1, maxvars
	   IF (ds_var_setnum(ii) .EQ. dset) THEN
	      IF (ds_var_c_cache_ptr(ii).NE.0.0) CALL FREE_CACHED_FULL_ARRAY
     .						(ds_var_c_cache_ptr(ii))
	   ENDIF
	ENDDO

* delete LET-defined variables that belong to this data set
	DO 100 ii = 1, max_uvar
*	   IF (uvar_dset(ii) .EQ. dset) uvar_num_items(ii) =
*     .					uvar_deleted
	   IF (uvar_dset(ii) .EQ. dset) THEN
	      CALL DELETE_USER_VAR(ii, dset)
              CALL deleted_list_modify(uvar_num_items_head, ii,
     .	            uvar_deleted)

* 6/16: This logic is now handled inside of DELETE_USER_VAR
!* Remove variable from linked-list structure, the user-variables dataset
!
!              slen = TM_LENSTR( uvar_name_code(ii)  )
!              IF (slen .GT. 0 .AND. uvar_name_code(ii)(1:3) .NE. 'EX#') THEN
!                 CALL TM_FTOC_STRNG( uvar_name_code(ii)(1:slen), fhol, flen )
!
!                 dset_uvars = pdset_uvars
!                 CALL NCF_DELETE_VAR (dset_uvars, fhol )
!              ENDIF

           ENDIF

 100	CONTINUE


* eliminate context information depending on this data set
	DO 200 ii = 1, max_context
	   IF ( cx_data_set( ii ) .EQ. dset ) THEN
	      cx_data_set( ii ) = unspecified_int4
	      cx_grid( ii ) = unspecified_int4
	   ENDIF
 200	CONTINUE

	RETURN
	END

