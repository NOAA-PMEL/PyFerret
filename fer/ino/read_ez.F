	SUBROUTINE READ_EZ (  cx, mr, status )

*
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
*
* read a file of EZ-type (ASCII or unformatted) data
* return only the requested variable in "mr"
* note: always read and store the entire grid limits of data

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  10/89/89
*	  4/24/90 - bug fix: when this routine determines data limits it must
*			    pass cx_hi_ww(idim,cx) as well as cx_hi_ss(cx,idim)
* V230:   7/15/92 - bug fix: was limited to 99999 record read
*         11/6/92 - increased variable length to 24 characters
* V300:   2/17/93 - bug fix: when requested lo limit exceeds received hi
*         3/12/93 - allow unlimited number of columns using FERRET mem mgmt
*	   5/5/93 - minor bug fix when file has no data and grid is only 1 pt
* V301:  12/21/93 - need to set separate bad value flag for each variable
* V312: 5/94 - array "memory" as a calling argument
* V320: 2/95 - increased number of readable vars from 10 to 20
*	3/95 - fixed bug - reserve "buff" of size "column" instead of "nrepeat"
* V500 *kob* 3/99- up VAR_CODE  to 64 chars
* V510: *sh* 3/00 - bug fix: needless (?) reference to cx_last
*		(see err500_2nd_try.jnl)
* V510: 6/00 *sh* - bug fix trap /COLUMNS= >grid size
* V552 *acm* 4/03- up VAR_CODE  to 128 chars
*       *acm* 3/12 cleanup old VMS ifdef
* V685 *acm* 1/13  Ticket 2030: if we run out of  issue a new message
* V702: *sh* 1/17 - bug fix: mr_protected must *end* at mr_not_in_use (=0)
*		  - for trac enhancement #2369 -- dynamic memory management 
* v767 6/22 *acm* See issue 112: Detect whether axis is too large for 4-byte integer indexing

        IMPLICIT NONE
	include	'tmap_errors.parm'
	include	'tmap_dims.parm'
	include	'xdset_info.cmn_text'
	external xdset_info_data
	include	'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xprog_state.cmn'
	include	'xtext_info.cmn'

* calling argument declarations:
	INTEGER	cx, mr, status

* internal variable declarations:
	INTEGER*8 	CGRID_SIZE8, gsize 
	INTEGER 	CX_DIM_LEN,
     .			idim, ask_var, category, data_set, grid, ivar,
     .			fvar, nvar20, the_var, nleft, nsize,
     .			cx0, cx1, cx20(20), mr20(20), nread,
     .			mr1, ndim, las_dim, fst_dim, dims(nferdims), rec1,
     .			rec_to_read, nrepeat, columns, num_names
	INTEGER*8	i8_size
	CHARACTER	VAR_CODE*128, errbuff*100

* --- end of introductory code ---

* nvar20  and the_var - value from 1 to 20      --> variable # within EZ set
* ask_var and fvar    - value from 1 to maxvars --> var # within DSET common

* initialize
	cx0 = cx_stack_ptr
	data_set = cx_data_set( cx )
	category = cx_category( cx )
	ask_var  = cx_variable( cx )
	grid = cx_grid( cx )
	fvar      = 0
	nvar20    = 0
	rec_to_read = 2**30      ! was 99999 - limited file reads

* how many values of each variable are repeated on each record ?
	columns = ds_ntuple(data_set)
	num_names = ds_basic_axes(1, data_set)
	nrepeat = columns / num_names

* purge from memory any variables from this data set to avoid duplication
* (1 variable may be deleted at any time by memory management but this routine
*  always reads all of 'em)
* note: problem arises if variable is in use
!	CALL PURGE_DSET( data_set )

* create a memory variable for each of up-to-20 variables in the file
	DO 400 ivar = 1, 20

* ... locate the next variable in this data set
	   DO 100 fvar = fvar+1, maxvars
	      IF ( ds_var_setnum( fvar ) .EQ. data_set ) GOTO 110
 100	   CONTINUE
* ... no more
	   GOTO 500

* ... is it a dummy '-' placeholder in the variable list ?
 110	   IF (INDEX( ds_var_code(fvar),'-') .NE. 0 ) GOTO 400

* save record of which is the desired var
	   nvar20 = nvar20 + 1
	   IF ( fvar .EQ. ask_var ) the_var = nvar20

* ... create a context describing this variable
	   CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   cx20(nvar20) = cx_stack_ptr
	   CALL TRANSFER_CONTEXT( cx, cx_stack_ptr )
	   cx_variable( cx_stack_ptr ) = fvar
	   cx_grid( cx_stack_ptr ) = ds_grid_number( fvar )
	   cx_bad_data( cx_stack_ptr ) = ds_bad_flag( fvar )	! 12/93 add'n
	   DO 200 idim = 1, nferdims
	      cx_lo_ss( cx_stack_ptr, idim ) = ds_grid_start( idim, fvar )
	      cx_hi_ss( cx_stack_ptr, idim ) = ds_grid_end  ( idim, fvar )
	      cx_by_ss( idim, cx_stack_ptr ) = .TRUE.
	      CALL CONFINE_AXIS( idim, cx_stack_ptr, status )
	      IF ( status .NE. ferr_ok ) RETURN
	      CALL FLESH_OUT_AXIS( idim, cx_stack_ptr, status )
	      IF ( status .NE. ferr_ok ) RETURN
 200	   CONTINUE

* read only what will fit into smallest grid

	   gsize = CGRID_SIZE8(cx_stack_ptr) 
	   IF (gsize/nrepeat .LT. arbitrary_large_int4) rec1 = gsize/ nrepeat
	   rec_to_read = MIN( rec_to_read, rec1 )
	   IF (rec_to_read .LE. 0 ) GOTO 5650

* reserve and catalog space for the result
	   CALL CREATE_MEM_VAR( cx_stack_ptr, mr20(nvar20), status )
	   IF ( status .NE. ferr_ok ) GOTO 5000

 400	CONTINUE

* point to asked-for variable and create dummy entries for the rest
 500	mr = mr20( the_var )
	DO 600 nleft = nvar20+1, 20
 600	mr20(nleft) = dummy_mr
	
* allocate a buffer big enough for one record
	i8_size = columns
        CALL GET_DYN_WORK_SPACE( i8_size, status )	! was "nrepeat" B4 3/95
	IF ( status .NE. merr_ok ) GOTO 5500

* diagnostic output
	IF (mode_diagnostic) CALL DIAGNOSTIC_OUT('reading',mr,point_to_mr)

* read the data via TMAP library routine
	nread = rec_to_read

	CALL EZ_READ( data_set, nread, nrepeat, columns,
     .          workmem(plegacy_work_buffer)%ptr, status,
     .		memry(mr20( 1))%ptr,
     .		memry(mr20( 2))%ptr,
     .		memry(mr20( 3))%ptr,
     .		memry(mr20( 4))%ptr,
     .		memry(mr20( 5))%ptr,
     .		memry(mr20( 6))%ptr,
     .		memry(mr20( 7))%ptr,
     .		memry(mr20( 8))%ptr,
     .		memry(mr20( 9))%ptr,
     .		memry(mr20(10))%ptr,
     .		memry(mr20(11))%ptr,
     .		memry(mr20(12))%ptr,
     .		memry(mr20(13))%ptr,
     .		memry(mr20(14))%ptr,
     .		memry(mr20(15))%ptr,
     .		memry(mr20(16))%ptr,
     .		memry(mr20(17))%ptr,
     .		memry(mr20(18))%ptr,
     .		memry(mr20(19))%ptr,
     .		memry(mr20(20))%ptr  )
	IF ( status .NE. merr_ok ) GOTO 5700

* free up the buffer space
        CALL  RELEASE_DYN_WORK_SPACE

* update the variables with info about the number of records actually read
* (for multi-dimensional grid reads just truncate the last axis best we can)
	IF ( nread .LT. rec_to_read ) THEN
	   DO 700 ivar = 1, nvar20
	      cx1 = cx20(ivar)
	      mr1 = mr20(ivar)
	      CALL GET_CX_DIMS( cx1, ndim, dims )
	      fst_dim = dims(1)
	      IF ( ndim .GT. 0 ) THEN        ! "IF" added 5/5/93
	         las_dim = dims(ndim)
	      ELSE
	         las_dim = fst_dim
	      ENDIF
	      IF ( ndim .EQ. 1 ) THEN
	         nsize = 1
	      ELSEIF( ndim .EQ. 2 ) THEN
	         nsize = CX_DIM_LEN(fst_dim,cx1)
	      ELSE
		 gsize = CGRID_SIZE8(cx1) 
		 IF (gsize .GT. arbitrary_large_int4) GOTO 5660
		 nsize =  gsize / CX_DIM_LEN(las_dim,cx1)
	      ENDIF
* ... last dim (each var) is
*     (total values read)/(prod. of all axis lengths except last)
	      cx_hi_ss( cx1, las_dim ) = MAX( 1, nrepeat*nread/nsize )
	      CALL FLESH_OUT_AXIS( las_dim, cx1, status )
	      IF ( status .NE. ferr_ok ) GOTO 6000
	      mr_hi_ss( mr1, las_dim ) = cx_hi_ss( cx1, las_dim )
	      mr_hi_ww( las_dim, mr1 ) = cx_hi_ww( las_dim, cx1 )
* ... also let the TMAP library know how much is there
	      ds_grid_end(las_dim, mr_variable(mr1)) = cx_hi_ss( cx1, las_dim )
 700	   CONTINUE
	ENDIF

* flag that all the data is available
	DO ivar = 1, nvar20
	   mr1 = mr20(ivar)
	   CALL MR_AVAILABLE(mr1)
	ENDDO

* the data region obtained may be inadequate for the request
* ("cx" is request, "cx1" is available)
	cx1 = cx20(the_var)
	DO 900 idim = 1, nferdims
	   IF ( cx_hi_ss(cx,idim) .GT. cx_hi_ss(cx1,idim) ) THEN
	      IF ( cx_given(idim,cx) ) THEN  ! 3/00 cut mystery cx_last ref.
* ... error: the user explicitly requested more than there is
	         GOTO 5600
              ELSEIF (cx_lo_ss(cx,idim) .GT. cx_hi_ss(cx1,idim)) THEN
* ... requested data is completely out of range of disk data (2/93 bug fix)
                 GOTO 5600
	      ELSE
* ... no error: the region was filled in by FERRET automatically
	         cx_hi_ss(cx,idim) = cx_hi_ss(cx1,idim)
	         cx_hi_ww(idim,cx) = cx_hi_ww(idim,cx1)
	      ENDIF
	   ENDIF
 900	CONTINUE

* successful completion
	cx_stack_ptr = cx0
	RETURN

* error exit
 4900	nvar20 = nvar20 + 1		! delete all
 5000	cx_stack_ptr = cx0
	IF (nvar20 .GT. 1) THEN
	   WRITE ( errbuff, 1000 ) nvar20
           CALL WARN (errbuff)
	ENDIF
 1000   FORMAT ('ASCII file reading: Insufficient memory reading variable ', i4)
	DO 5010 nleft = 1, nvar20-1	! delete all except last
 5010	CALL DELETE_VARIABLE(mr20(nleft))
        CALL  RELEASE_DYN_WORK_SPACE
 5500	RETURN
 5600	CALL ERRMSG( ferr_limits, status,
     .		ww_dim_name(idim)//' axis of '//VAR_CODE(cat_file_var,ask_var),
     .		*5500 )
 5650	CALL ERRMSG( ferr_limits, status,
     .		'/COLUMNS is impossibly large', *5500 )
 5660	CALL ERRMSG( ferr_limits, status,
     .		'/COLUMNS is impossibly large or total size too large', *5500 )
 5700	CALL ERRMSG( ferr_TMAP_error, status, ' ', *4900 )
 6000	CALL ERRMSG( ferr_internal, status, 'READ_EZ ??', *4900 )
	END	

