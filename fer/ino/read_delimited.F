	SUBROUTINE READ_DELIMITED( memory, cx, mr, status )

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
* read an ASCII delimited file
* return only the requested variable in "mr"
* note: always read and store the entire grid limits of data

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V530: 11/00 *sh*
* V533: 6/01 *sh* - check that data type is correct instead of asserting it
* V540: 10/01 *sh* - undo the erroneous change of V533 (why was V533 chg done?)
* V500 *kob* 3/99- up VAR_CODE  to 64 chars
* V552 *acm* 4/03- up VAR_CODE  to 128 chars

	include	'tmap_errors.parm'
	include	'tmap_dims.parm'
	include	'ez_lib.parm'
	include	'xstep_files.cmn_text'
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
	REAL	memory( mem_blk_size, max_mem_blks )

* internal variable declarations:
	INTEGER 	CGRID_SIZE, CX_DIM_LEN, TM_LENSTR1,
     .			idim, ask_var, category, dset, grid, ivar,
     .			fvar, nvar, the_var, nleft, nsize,
     .			cx0, cx1, nread, istep,
     .			cxlist(maxezvars), mrlist(maxezvars),
     .			field_type(maxezvars),
     .			mr1, ndim, las_dim, fst_dim, dims(4),
     .			rec_to_read, buffblk
	CHARACTER	VAR_CODE*128, delims*80

* --- end of introductory code ---

* nvar  and the_var - value from 1 to maxezvars --> variable # within EZ set
* ask_var and fvar    - value from 1 to maxvars --> var # within DSET common

* initialize
	cx0 = cx_stack_ptr
	dset = cx_data_set( cx )
	category = cx_category( cx )
	ask_var  = cx_variable( cx )
	grid = cx_grid( cx )
	fvar      = 0
	CALL GET_DELIMITED_INFO(nvar, field_type, delims, 
     .				   ds_file_info_ptr(dset) )

* locate the step file belonging to this data set

	DO 10 istep = 1,maxstepfiles
	  IF (sf_setnum(istep) .EQ. dset) GOTO 20
 10	CONTINUE
	CALL ERRMSG( ferr_internal, status, 'delim stepfile?', *5000)

* create a memory variable for each variable in the file
 20	DO 400 ivar = 1, nvar

* ... locate the next variable in this data set
	   DO 100 fvar = fvar+1, maxvars
	      IF ( ds_var_setnum(fvar) .EQ. dset ) GOTO 110
 100	   CONTINUE
* ... no more
	   GOTO 500

* ... is it a dummy '-' placeholder in the variable list ?
 110	   IF (field_type(ivar) .EQ. FTYP_MISSING) THEN
	      cxlist(ivar) = 0
	      mrlist(ivar) = 0
	   ELSE

* ... create a context describing this variable
	     CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	     IF ( status .NE. ferr_ok ) RETURN
	     cxlist(ivar) = cx_stack_ptr
	     CALL TRANSFER_CONTEXT( cx, cx_stack_ptr )
	     cx_variable( cx_stack_ptr ) = fvar
	     cx_grid    ( cx_stack_ptr ) = ds_grid_number( fvar )
	     cx_bad_data( cx_stack_ptr ) = ds_bad_flag( fvar )	! 12/93 add'n
	     cx_type    ( cx_stack_ptr ) = ds_var_type( fvar )
	     DO 200 idim = 1, 4
	        cx_lo_ss(cx_stack_ptr, idim) = ds_grid_start(idim, fvar)
	        cx_hi_ss(cx_stack_ptr, idim) = ds_grid_end  (idim, fvar)
	        cx_by_ss(idim, cx_stack_ptr) = .TRUE.
	        CALL CONFINE_AXIS( idim, cx_stack_ptr, status )
	        IF ( status .NE. ferr_ok ) RETURN
	        CALL FLESH_OUT_AXIS( idim, cx_stack_ptr, status )
	        IF ( status .NE. ferr_ok ) RETURN
 200	     CONTINUE

* reserve and catalog space for the result
	     CALL CREATE_MEM_VAR( cx_stack_ptr, mrlist(ivar), status )
	     IF ( status .NE. ferr_ok ) GOTO 5000
	   ENDIF

* save index of the desired var
	   IF ( fvar .EQ. ask_var ) the_var = ivar

 400	CONTINUE

* point to asked-for variable and create dummy entries for the rest
 500	rec_to_read = CGRID_SIZE(cxlist(the_var))
	mr = mrlist( the_var )
	
* diagnostic output
	IF (mode_diagnostic) CALL DIAGNOSTIC_OUT('reading',mr,point_to_mr)

* allocate a buffer big enough for one input ASCII record
        CALL GET_WORK_SPC( 1024, buffblk, status ) ! really 4x1024
        IF ( status .NE. merr_ok ) GOTO 5500

* create a null terminated string from the data set name (use record buffer)
	CALL TM_FTOC_STRNG( ds_des_name(dset)
     .		     (:TM_LENSTR1(ds_des_name(dset))),
     .		     memory(1,buffblk), 4*1024 )

* read the data via TMAP library routine
	CALL DECODE_FILE_JACKET (
     .		  memory(1,buffblk), memory(1,buffblk),
     .		  delims, sf_skip(istep), rec_to_read, 4*1024, nvar,
     .		  field_type, nread, mrlist, memory, mr_blk1,
     .		  mem_blk_size, mr_bad_data, mr_c_pointer)

* update the variables with info about the number of records actually read
* (for multi-dimensional grid reads just truncate the last axis best we can)
	IF ( nread .LT. rec_to_read ) THEN
	   DO 700 ivar = 1, nvar
	      IF (field_type(ivar) .NE. FTYP_MISSING) THEN
	        cx1 = cxlist(ivar)
	        mr1 = mrlist(ivar)
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
	           nsize =  CGRID_SIZE(cx1) / CX_DIM_LEN(las_dim,cx1)
	        ENDIF
* ... last dim (each var) is
*     (total values read)/(prod. of all axis lengths except last)
	        cx_hi_ss( cx1, las_dim ) = MAX( 1, nread/nsize )
	        CALL FLESH_OUT_AXIS( las_dim, cx1, status )
	        IF ( status .NE. ferr_ok ) GOTO 6000
	        mr_hi_ss( mr1, las_dim ) = cx_hi_ss( cx1, las_dim )
	        mr_hi_ww( las_dim, mr1 ) = cx_hi_ww( las_dim, cx1 )
* ... also let the TMAP library know how much is there
	        ds_grid_end(las_dim, mr_variable(mr1)) = cx_hi_ss( cx1, las_dim )
	      ENDIF
 700	   CONTINUE
	ENDIF

* flag that all the data is available
	DO 800 ivar = 1, nvar
	   IF (field_type(ivar) .NE. FTYP_MISSING) THEN
	     mr1 = mrlist(ivar)
	     mr_protected( mr1 ) = mr_not_protected
	     CALL MR_NOT_IN_USE( mr1 )
	   ENDIF
 800	CONTINUE

* the data region obtained may be inadequate for the request
* ("cx" is request, "cx1" is available)
	cx1 = cxlist(the_var)
	DO 900 idim = 1, 4
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

* set the data type based on the file
	cx_type(cx) = cx_type(cx1)

* successful completion
	cx_stack_ptr = cx0
	RETURN

* error exit
 4900	nvar = nvar + 1		! delete all
 5000	cx_stack_ptr = cx0
	DO 5010 nleft = 1, nvar-1	! delete all except last
 5010	CALL DELETE_VARIABLE(mrlist(nleft))
        CALL  RELEASE_WORK_SPC
 5500	RETURN
 5600	CALL ERRMSG( ferr_limits, status,
     .		ww_dim_name(idim)//' axis of '//VAR_CODE(cat_file_var,ask_var),
     .		*5500 )
 5650	CALL ERRMSG( ferr_limits, status,
     .		'/COLUMNS is impossibly large', *5500 )
 5700	CALL ERRMSG( ferr_TMAP_error, status, ' ', *4900 )
 6000	CALL ERRMSG( ferr_internal, status, 'READ_EZ ??', *4900 )
	END	
