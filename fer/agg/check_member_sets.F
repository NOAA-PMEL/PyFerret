	SUBROUTINE CHECK_MEMBER_SETS (agg_dim, nsets, memb_dsetnames,
     .				      member_sets, status)
*
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
* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP 
*
* V680 5/12 *acm* 6D Ferret, changes for DEFINE DATA/AGGREGATE
* V685 *acm*  3/13 For Ensemble aggregations, call the ds_type 'ENS'
* V695 *acm*  2/15 For Forecast aggregations, call the ds_type 'FCT'
* V695+ *sh*  5/15 removed block of code from here to INIT_AGGREGATE_DSET
*                  notably, the call to GET_AXIS_SPEED
* V698+ *sh* 11/15 add ok_re_use_existing to TM_INIT_DSET call
*            12/15 pass dataset names and nsets *into* this routine
*	           if dataset not already open, then open it hidden
*	      2/16 allow "-" as an indicator of a dummy dataset
*	      3/16 check for illegal Unix filenames
*	      4/16 defer checking aggs build of aggs to OK_AS_AGG_VAR
* V741 *acm* 5/18 Promoting global attributes to a variable in ensembles:
*                 File extensions for member sets were not being stripped off 

* calling arguments
*   agg_dim             - axis of the aggregation being created
*   nsets		- number of datasets to be aggregated (input)
*   memb_dsetnames	- array of names of the datasets (input)
*   member_sets		- array of open dataset numbers (output)
*   status		- return status

        IMPLICIT NONE
        include 'tmap_errors.parm'
#include "tmap_dset.parm"
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
        include 'xprog_state.cmn'
	include 'xtm_grid.cmn_text'  
	include 'errmsg.parm'
	include 'netcdf.inc'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xfr_grid.cmn'
	include 'xdiag_ctrl.cmn_text'

* calling argument declarations:
	INTEGER agg_dim, nsets, member_sets(nsets), status
	CHARACTER*(*), memb_dsetnames(nsets)

* local variable declarations:
	LOGICAL TM_DIGIT, tregular, use_strict
	INTEGER TM_LENSTR, i, ich, buff_len, dset, iset, perm(nferdims)
	CHARACTER buff1*512

* local parameter definitions
	LOGICAL ok_re_use_existing
	PARAMETER ( ok_re_use_existing = .TRUE.)

	DO 100 iset = 1, nsets
	   member_sets(iset) = unspecified_int4
  100	CONTINUE

	DO 110 i = 1, nferdims
	   perm(i) = unspecified_int4
  110	CONTINUE

* Check on the aggregate-member datasets. (code taken from xeq_set)
* Open them if they're not already open.

	tregular = .FALSE.
	use_strict = .FALSE.

	DO iset = 1, nsets

	   buff1 = memb_dsetnames(iset)
	   buff_len = TM_LENSTR(buff1)

* check for legal Unix filename
* could use tm_legal_unix_name.F, but it is old and overly strict
* the netCDF library will refuse an illegal name, but it will not give a nice
* message
* can easily add more to this test as needed
	   IF (INDEX(buff1(:buff_len)," ") .GT. 0) GOTO 5050

	   IF (buff1 .EQ. "-") THEN
* dummy dataset indicator
	      IF (iset .EQ. 1) GOTO 5100
	      member_sets(iset) = pdset_dummy
	      CYCLE
	   ENDIF

* check to see if data set was referred to by number
* 9/95 - replaced "*" with "(I)" read from separate buffer
	   IF ( TM_DIGIT(buff1) ) THEN

* avoid crash due to single character "+" or "."
	      IF (buff_len.EQ.1
     .      .AND.(buff1(1:1).LT.'0' .OR. buff1(1:1).GT.'9')) GOTO 740

              DO ich = 1, buff_len
		IF ( .NOT. TM_DIGIT(buff1(ich:ich))) GOTO 740
	      ENDDO
	      READ (buff1,*) dset
	      READ (buff1,*,ERR=740) dset
* ... check for valid number given
	      IF   ( dset .LT. 1
     .	      .OR.   dset .GT. maxdsets )	  GOTO 5200
	      IF (ds_name(dset).EQ.char_init2048) GOTO 5200
	      GOTO 750
	   ENDIF

* no! it wasn't given by number so open the data set
 740       ds_dset_type = 'CDF'   ! used and cleared in tm_init_dset

* initialize a TMAP-supported data set
 	   CALL TM_INIT_DSET( buff1(:buff_len), ok_re_use_existing,
     .			      perm, dset, tregular, use_strict, mode_dsg, status )
	   
           IF ( status .NE. merr_no_action ) THEN  ! merr_no_action = already init'ed
	      IF ( status .NE. ferr_ok ) GOTO 5300 ! error in file or url
	      ds_hide(dset) = .TRUE.   ! hide unless previously opened
	   ENDIF

 750	   member_sets(iset) = dset

	ENDDO

	status = ferr_ok

 5000   RETURN	

* Error exits: 
 5050	CALL ERRMSG( ferr_invalid_command, status,
     .		'corrupted filename: '//buff1(:buff_len), *5000 )
 5100	CALL ERRMSG( ferr_invalid_command, status,
     .		'1st aggregate member may not be missing', *5000)
 5200	CALL ERRMSG( ferr_unknown_data_set, status,
     .		'dset '//buff1(:buff_len),         *5000 )
 5300	CALL ERRMSG( ferr_TMAP_error, status, ' ', *5000 )

	END
