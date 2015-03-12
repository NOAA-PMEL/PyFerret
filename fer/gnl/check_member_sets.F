	SUBROUTINE CHECK_MEMBER_SETS (nsets, member_sets, status)
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
	INTEGER nsets, member_sets(*), status

* local variable declarations:
	LOGICAL TM_DIGIT, tregular, use_strict
	INTEGER TM_LENSTR, item, ich, buff_len, dset, iset, perm(nferdims)
	CHARACTER buff1*512

	DO 100 iset = 1, maxdsets
	   member_sets(iset) = unspecified_int4
  100	CONTINUE

	DO 110 item = 1, nferdims
	   perm(item) = unspecified_int4
  110	CONTINUE

* Check on the aggregate-member datasets. (code taken from xeq_set)
* Open them if they're not already open.

	tregular = .FALSE.
	use_strict = .FALSE.

	nsets = 0
	DO 760 item = 1, num_items

           CALL EQUAL_STR_LC(
     .                   cmnd_buff(item_start(item):item_end(item)),
     .                   buff1, status ) 
	   IF (status .NE. ferr_ok)
     .        buff1 = cmnd_buff(item_start(item):item_end(item))
	   IF (TM_LENSTR(buff1) .EQ. 0)
     .        buff1 = cmnd_buff(item_start(item):item_end(item))

* check to see if data set was referred to by number
* 9/95 - replaced "*" with "(I)" read from separate buffer
	   IF ( TM_DIGIT(buff1) ) THEN

	      buff_len = TM_LENSTR(buff1)
              DO 730 ich = 1, buff_len
		IF ( .NOT. TM_DIGIT(buff1(ich:ich))) GOTO 740
 730	      CONTINUE
	      READ (buff1,*) dset
#ifdef MANDATORY_FORMAT_WIDTHS
	      READ (buff1,*,ERR=740) dset
#else
	      READ (buff1,'(I)',ERR=740) dset
#endif
* ... check for valid number given
	      IF   ( dset .LT. 1
     .	      .OR.   dset .GT. maxdsets )		GOTO 5200
	      IF ( ds_name( dset ) .EQ. char_init1024) GOTO 5200
	      IF (ds_type(dset).EQ.'ENS' .OR. ds_type(dset).EQ.'FCT') GOTO 5200
	      GOTO 750
	   ENDIF

* no! it wasn't given by number so open the data set
 740       ds_dset_type = 'CDF'   ! used and cleared in tm_init_dset

* initialize a TMAP-supported data set
 	   CALL TM_INIT_DSET( buff1,
     .			      perm, dset, tregular, use_strict, status )
	   
           IF ( status .NE. merr_no_action ) THEN  ! merr_no_action = already init'ed
	      IF ( status .NE. ferr_ok ) GOTO 5300 ! error in file or url

* interpret special parameters in the descriptor
	      CALL GET_DSET_PARMS( dset, status )
	      IF ( status .NE. ferr_ok ) GOTO 5300

* determine relative access speeds along each axis
	      CALL GET_AXIS_SPEED( dset )

* build completed ( with time axis ) grids for computable variables
	      CALL MAKE_CVAR_GRIDS( dset, status )
	      IF ( status .NE. ferr_ok ) GOTO 5300

	   ENDIF

 750	   CONTINUE
	   nsets = nsets + 1
	   member_sets(nsets) = dset

 760	CONTINUE

	status = ferr_ok

 5000   RETURN	

* Errors: too-many datasets open so aggregate dataset can't be opened,
* or error on a member dataset. 
* ?? If some member datasets were opened, close them??)
* Return no-dataset-number.

 9950	dset = unspecified_int4
	status = ferr_TMAP_error
	GOTO 5000

 5200	CALL ERRMSG( ferr_unknown_data_set, status,
     .		'dset '//cmnd_buff( item_start(item):item_end(item) )
     .		//' unknown or is already an aggregate dataset', *9950 )
 5300	CALL ERRMSG( ferr_TMAP_error, status, ' ', *9950 )

	END
