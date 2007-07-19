	SUBROUTINE READ_BINARY ( memory, cx, mr, status )
* read a stream formatted file 
* stolen from read_ez

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* 7/98 *js*  First version hacked from read_ez
* 3/99 *kob* - nag F90 wants to use eqv for logical expressions
* V552 4/03 *acm* Up VAR_CODE to 128 characters
* V552 5/03 *acm* Use ds_des_name(data_set) when opening the data; it has
*                 the pathname as well as the filename
* v604 *acm* 7/07 increase length of ds_name

	include	'tmap_errors.parm'
	include	'tmap_dims.parm'
	include	'xdset_info.cmn_text'
	external xdset_info_data
	include	'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xstep_files.cmn_text'
	external xstep_files_data
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xprog_state.cmn'
	include	'xtext_info.cmn'

* calling argument declarations:
	INTEGER	cx, mr, status, fstatus
	REAL	memory( mem_blk_size, max_mem_blks ), fdummy(1)

* functions
	INTEGER BR_OPEN, BR_READ, BR_ADD_VAR

* internal variable declarations:
	INTEGER 	CGRID_SIZE, 
     .			idim, ask_var, category, data_set, grid, ivar,
     .			fvar, nvar20, the_var, nleft, 
     .			cx0, cx1, cx20(20), mr20(20),
     .			mr1, rec1, scnt,
     .			rec_to_read, nrepeat, columns, num_names, buffblk
	INTEGER         lengths(5), permutes(5), skip, doRead
	LOGICAL         firstTime
	CHARACTER	VAR_CODE*128
	CHARACTER*256   cerrmsg,ferrmsg
        CHARACTER*1024  cname

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
	firstTime = .true.
	skip = 0
	doRead = 0

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
	DO ivar = 1, 20

* ... locate the next variable in this data set
	   DO fvar = fvar+1, maxvars
	      IF ( ds_var_setnum( fvar ) .EQ. data_set ) then
* ... is it a dummy '-' placeholder in the variable list ?
 		 IF (INDEX( ds_var_code(fvar),'-') .EQ. 0 ) THEN

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
		    cx_bad_data( cx_stack_ptr ) = ds_bad_flag( fvar ) ! 12/93 add'n
		    DO idim = 1, 4
		       cx_lo_ss( cx_stack_ptr, idim ) = ds_grid_start( idim, fvar )
		       cx_hi_ss( cx_stack_ptr, idim ) = ds_grid_end  ( idim, fvar )
		       cx_by_ss( idim, cx_stack_ptr ) = .TRUE.
* 3/99 *kob* - nag F90 wants to use eqv for logical expressions
		       if (firstTime .eqv. .TRUE.) THEN
			  lengths(idim) = ds_grid_end(idim,fvar) - 
	1		       ds_grid_start(idim,fvar) + 1
		       end if
		       CALL CONFINE_AXIS( idim, cx_stack_ptr, status )
		       IF ( status .NE. ferr_ok ) RETURN
		       CALL FLESH_OUT_AXIS( idim, cx_stack_ptr, status )
		       IF ( status .NE. ferr_ok ) RETURN
		    END DO
		    firstTime = .FALSE.

*       read only what will fit into smallest grid
		    rec1 = CGRID_SIZE(cx_stack_ptr) / nrepeat
		    rec_to_read = MIN( rec_to_read, rec1 )

*       reserve and catalog space for the result
		    CALL CREATE_MEM_VAR( cx_stack_ptr, mr20(nvar20), status )
		    IF ( status .NE. ferr_ok ) GOTO 5000
		 END IF
	      END IF
	   END DO
	END DO

* point to asked-for variable and create dummy entries for the rest
 500	mr = mr20( the_var )
	DO nleft = nvar20+1, 20
	   mr20(nleft) = dummy_mr
	END DO
	
* allocate a buffer big enough for one record
        CALL GET_WORK_SPC( columns, buffblk, status )	! was "nrepeat" B4 3/95
	IF ( status .NE. merr_ok ) GOTO 5500

* Get the current permutation list
	CALL BR_GET_PERMUTES(permutes)

* Get the skip header
	do scnt = 1, maxstepfiles
	   if (sf_setnum(scnt) .EQ. data_set) then
	      skip = sf_skip(scnt)
	      goto 3000
	   end if
	end do
 3000	continue

* initialize the binary reader
!	call toChar(ds_name(data_set), cname)
	call toChar(ds_des_name(data_set), cname)
	if (BR_OPEN(cname, lengths, permutes,
     1      skip) .eq. 0) goto 6000

	fvar = 0
	nvar20 = 0
	DO ivar = 1, 20
	   DO fvar = fvar+1, maxvars
	      IF ( ds_var_setnum( fvar ) .EQ. data_set ) then
 		 IF (INDEX( ds_var_code(fvar),'-') .EQ. 0 ) THEN
		    doRead = 1
		    nvar20 = nvar20 + 1
		    fstatus = BR_ADD_VAR(memory(1, mr_blk1(mr20(nvar20))),
     1		                 doRead)
		 ELSE
		    doRead = 0
		    fstatus = BR_ADD_VAR(fdummy, doRead)
		 END IF
		 if (fstatus .EQ. 0) goto 6000
	      END IF
	   END DO
	END DO


* read the data
	if (BR_READ() .EQ. 0) goto 6000



* free up the buffer space
        CALL  RELEASE_WORK_SPC

* flag that all the data is available
	DO 800 ivar = 1, nvar20
	   mr1 = mr20(ivar)
	   mr_protected( mr1 ) = mr_not_protected
	   CALL MR_NOT_IN_USE( mr1 )
 800	CONTINUE

* the data region obtained may be inadequate for the request
* ("cx" is request, "cx1" is available)
	cx1 = cx20(the_var)
	DO 900 idim = 1, 4
	   IF ( cx_hi_ss(cx,idim) .GT. cx_hi_ss(cx1,idim) ) THEN
	      IF ( cx_hi_ss(cx_last,idim) .NE. unspecified_int4
     .	     .OR.  cx_given(idim,cx) ) THEN
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
	CALL BR_CLOSE()
	RETURN

* error exit
 4900	nvar20 = nvar20 + 1		! delete all
 5000	cx_stack_ptr = cx0
	DO 5010 nleft = 1, nvar20-1	! delete all except last
 5010	CALL DELETE_VARIABLE(mr20(nleft))
        CALL  RELEASE_WORK_SPC
	CALL BR_CLOSE()
 5500	RETURN
 5600	CALL ERRMSG( ferr_limits, status,
     .		ww_dim_name(idim)//' axis of '//VAR_CODE(cat_file_var,ask_var),
     .		*5500 )
 6000	CALL BR_GET_ERROR(cerrmsg)
	CALL TM_CTOF_STRNG(cerrmsg, ferrmsg, 255)
	CALL TM_SPLIT_MESSAGE(err_lun, ' **ERROR: '//ferrmsg(1:LEN(ferrmsg)))
        CALL ERRMSG( ferr_TMAP_error, status, ' ', *4900 )
	END	

c-------------------------------------------------------------------------------
c     Convert a Fortran string to a zero delimited C string

      subroutine toChar(fstring, cstring)
      character*(*) fstring, cstring

      integer TM_LENSTR1
      integer length
      
      length = TM_LENSTR1(fstring)
      cstring = fstring(1:length)
      cstring(length+1:length+1) = char(0)
      return
      end
