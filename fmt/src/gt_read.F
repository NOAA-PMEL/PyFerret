	SUBROUTINE GT_READ ( dset_num, varpos, time_step,
     .			     a_lo_x, a_lo_y, a_lo_z, a_lo_t,
     .			     a_hi_x, a_hi_y, a_hi_z, a_hi_t,
     .			     use_lo, use_hi,
     .			     data_grid, sf_num, status )
*
* Reads a 3-D block of data (2D being a degenerate case w/ 1 dimension being
* 1 level thick) from a TMAP direct access file.
*
* written 04/20/87 by M. Verschell for PMEL/TMAP
*
* revision 0.00 - 04/20/87 - initial incarnation
* revision 0.10	- 10/04/88 - changed actual dims to 4D to work with TM_READ
* revision 0.20 - 11/01/88 - Name changed from GT_GET_GRID; passed variable 
*			     position in common rather than variable name - this
*			     routine called from TM_CHOOSE_READ
* revision 0.30 - 12/06/89 - shanged arrays for dimensions to single variables
* Unix/RISC port 3/91: convert data if written on a different cpu type
* SUN port 2/92  -kob ----  changed sf_fp_convert to sf_cpu_type
* 10/96 *kob* - Linux port - had to move variable assignments off of
*		declaration line.

* arguments:
*	dset_num    - internal data set # (from GT_INIT_DSET)
*	varpos      - position in common of variable information
*	time_step   - desired time step value
*	actual_dims - "a_" full dimensions of data_grid array passed
*	use_dims    - indices of model data block desired
*	data_grid   - array containing model data (output)
*	sf_num      - stepfile number (returned from GT_FIND_TSTEP (output)
*	status      - error status upon completion (output)
*
	INTEGER		dset_num, varpos, sf_num, status
	INTEGER		a_lo_x, a_lo_y, a_lo_z, a_lo_t,
     .			a_hi_x, a_hi_y, a_hi_z, a_hi_t,
     .			use_lo(4), use_hi(4)
	REAL		data_grid(a_lo_x:a_hi_x,
     .				  a_lo_y:a_hi_y,
     .				  a_lo_z:a_hi_z,
     .				  a_lo_t:a_hi_t)
	REAL*8		time_step

* include files
#ifdef unix
	include 'tmap_format/tmap_dims.parm'		! data set dimension
	include 'tmap_format/tmap_errors.parm'		! err code parm defs
	include 'tmap_format/gt_lib.parm'		! axis setup parameters
	include 'tmap_format/xdset_info.cmn_text'	! data set info common
	external xdset_info_data
	include 'tmap_format/xstep_files.cmn_text'	! step file info common
	external xstep_files_data
	include 'tmap_format/xio.cmn_text'		!i/o channel common
	external xio_data
#ifdef sun
        include 'tmap_format/xmachine_byte.cmn'
#else
        include 'tmap_format/xmachine_int1.cmn'
#endif
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'		! Data set dimension
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'		! err code PARM defs
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'		! Axis setup parameters
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'		! Data set info common
	INCLUDE 'TMAP_FORMAT:XSTEP_FILES.CMN'		! Step file info common
	INCLUDE 'TMAP_FORMAT:XIO.CMN'			!I/O channel common
	INCLUDE 'TMAP_FORMAT:XMACHINE_BYTE.CMN'
#endif

* local definitions
* *kob* explicit definitions not allowed on decl line in linux...
*       moved it to parameter line 10/96
	CHARACTER*1	char_dim(3)
	DATA		char_dim/'I','J','K'/
	CHARACTER*4	rec_code
	CHARACTER*8	variable
	CHARACTER*13	TM_STRING
        LOGICAL         contig, cp_convert
	INTEGER 	i, j, k, l, xorder, yorder, zorder, ndim, lunit,
     .			bypoint, byline, byplane, pl_rec, l_rec, seq_num,
     .			file_seq, varstrt, full_grid, i_lo, i_hi,
     .			j_lo, j_hi, k_lo, k_hi, istrt, jstrt, kstrt,
     .			line_size, cptype
	INTEGER		GT_FULL_GRID
	REAL*4		dummy

	PARAMETER	(xorder=1, yorder=2, zorder=3)

* Initialize by locating correct file and starting record of variable
* in addition to variables name in common storage
	variable = ds_var_code(varpos)
	CALL GT_FIND_TSTEP ( dset_num, time_step, varpos,
     .			     sf_num, varstrt, status)
	IF (status .NE. merr_ok) GOTO 9900

* assign 3 integers to array showing axis positioning in data file
	bypoint = ds_ordering(1,varpos)
	byline  = ds_ordering(2,varpos)
	byplane = ds_ordering(3,varpos)

* check if requested grid dimensions are in correct order, and check requested
* grid to see if it lies in data set
	  DO 700 ndim = 1,3
	    IF (use_lo(ndim) .GT. use_hi(ndim)) CALL TM_ERRMSG
     .		(merr_badsubscr, status, 'GT_READ',
     .		 dset_num, sf_num,
     .		 'for requested '//char_dim(ndim)//' subscripts',
     .		 no_errstring, *9900)
	    IF ( use_lo(ndim) .LT. ds_grid_start(ndim,varpos) .OR.
     .	       use_hi(ndim) .GT. ds_grid_end(ndim,varpos) )
     .		 CALL TM_ERRMSG (merr_outofgrid, status, 'GT_READ',
     .		 dset_num, sf_num,
     .		 'TSTEP='//TM_STRING(time_step)//'VAR='//variable//
     .		 char_dim(ndim)//'='//TM_STRING(DBLE(use_lo(ndim)))//'-'//
     .		 TM_STRING(DBLE(use_hi(ndim))), no_errstring, *9900)
700	  CONTINUE
* check to see if asking for unsupported data ordering
	  IF (bypoint .LE. 0 .OR. byline .LE. 0 .OR. byplane .LE. 0)
     .		CALL TM_ERRMSG (merr_notsupport, status, 'GT_READ',
     .				dset_num, sf_num,
     .				'Descriptor requests backwards data '//
     .				'order for VAR='//variable,
     .				no_errstring, *9900)

* dimensions of full defining grid for axis stored by line
	full_grid = GT_FULL_GRID(ds_grid_number(varpos),byline)

* pull off lo and hi requested dimensions and start of grid in data file
	k_lo  = use_lo(3)
	k_hi  = use_hi(3) 
	kstrt = ds_grid_start(3,varpos)
	j_lo  = use_lo(2)
	j_hi  = use_hi(2)
	jstrt = ds_grid_start(2,varpos)
	i_lo  = use_lo(1)
	i_hi  = use_hi(1)
	istrt = ds_grid_start(1,varpos)

	line_size = (ds_grid_end(byline,varpos) -
     .		     ds_grid_start(byline,varpos)) + 1

* Get logical unit number from COMMON/XSTEP_FILES/
	lunit = sf_lunit(sf_num)
	cptype = sf_cpu_type(sf_num)
	cp_convert = cptype .NE. (active_cpu)

* Decide on loop and read depending upon data storage
	l = use_lo(4)
	IF (bypoint .EQ. xorder) THEN
	  IF (byline .EQ. yorder) THEN
	    DO 100 k = k_lo,k_hi
	    pl_rec = (varstrt-1) + (k-kstrt)*line_size
	    DO 100 j = j_lo,j_hi
	      l_rec = j-jstrt+1
	      READ (lunit, REC=pl_rec+l_rec, ERR=9000) rec_code, seq_num,
     .					(dummy,i=istrt,i_lo-1),
     .					(data_grid(i,j,k,l),i=i_lo,i_hi)
100	    CONTINUE
	  ELSE IF (byline .EQ. zorder) THEN
	    DO 200 j = j_lo,j_hi
	    pl_rec = (varstrt-1) + (j-jstrt)*line_size
	    DO 200 k = k_lo,k_hi
	      l_rec = k-kstrt+1
	      READ (lunit, REC=pl_rec+l_rec, ERR=9000) rec_code, seq_num,
     .					(dummy,i=istrt,i_lo-1),
     .					(data_grid(i,j,k,l),i=i_lo,i_hi)
200	    CONTINUE
	  ENDIF
	ELSE IF (bypoint .EQ. yorder) THEN
	  IF (byline .EQ. xorder) THEN
	    DO 300 k = k_lo,k_hi
	    pl_rec = (varstrt-1) + (k-kstrt)*line_size
	    DO 300 i = i_lo,i_hi
	      l_rec = i-istrt+1
	      READ (lunit, REC=pl_rec+l_rec, ERR=9000) rec_code, seq_num,
     .					(dummy,j=jstrt,j_lo-1),
     .					(data_grid(i,j,k,l),j=j_lo,j_hi)
300	    CONTINUE
	  ELSE IF (byline .EQ. zorder) THEN
	    DO 400 i = i_lo,i_hi
	    pl_rec = (varstrt-1) + (i-istrt)*line_size
	    DO 400 k = k_lo,k_hi
	      l_rec = k-kstrt+1
	      READ (lunit, REC=pl_rec+l_rec, ERR=9000) rec_code, seq_num,
     .					(dummy,j=jstrt,j_lo-1),
     .					(data_grid(i,j,k,l),j=j_lo,j_hi)
400	    CONTINUE
	  ENDIF
	ELSE IF (bypoint .EQ. zorder) THEN
	  IF (byline .EQ. xorder) THEN
	    DO 500 j = j_lo,j_hi
	    pl_rec = (varstrt-1) + (j-jstrt)*line_size
	    DO 500 i = i_lo,i_hi
	      l_rec = i-istrt+1
	      READ (lunit, REC=pl_rec+l_rec, ERR=9000) rec_code, seq_num,
     .					(dummy,k=kstrt,k_lo-1),
     .					(data_grid(i,j,k,l),k=k_lo,k_hi)
500	    CONTINUE
	  ELSE IF (byline .EQ. yorder) THEN
	    DO 600 i = i_lo,i_hi
	    pl_rec = (varstrt-1) + (i-istrt)*line_size
	    DO 600 j = j_lo,j_hi
	      l_rec = j-jstrt+1
	      READ (lunit, REC=pl_rec+l_rec, ERR=9000) rec_code, seq_num,
     .					(dummy,k=kstrt,k_lo-1),
     .					(data_grid(i,j,k,l),k=k_lo,k_hi)
600	    CONTINUE
	  ENDIF
	ENDIF

* Note: machines with a different endian will require seq_num to be repaired
* file written on a different machine ? convert machine words
	IF ( cp_convert ) CALL TM_INT4_SWAP( seq_num, cptype )

* check validity of last read as test of file positioning
	file_seq = (use_hi(byplane)-1)*full_grid+use_hi(byline)
	IF ( rec_code .NE. ds_var_code(varpos)
     .	  .OR. seq_num .NE. file_seq ) THEN
* no bad subscripts, asking for data out of bounds, or unsupported data
* ordering; so give general error
	  CALL TM_ERRMSG (merr_filpos, status, 'GT_READ',
     .		dset_num, sf_num,
     .		ds_var_code(varpos)//' SEQNUM='//TM_STRING(DBLE(file_seq)),
     .		rec_code//' SEQNUM='//TM_STRING(DBLE(seq_num)), *9900)
	ENDIF

* if file was written on a cpu with a different binary format fix things up
        IF ( cp_convert ) THEN
*    if data is contiguous in memory fix it with a single call
           contig = a_lo_x .EQ. use_lo(1)
     .        .AND. a_hi_x .EQ. use_hi(1)
     .        .AND. a_lo_y .EQ. use_lo(2)
     .        .AND. a_hi_y .EQ. use_hi(2)
           IF ( contig ) THEN
              CALL TM_FP_CONVERT(
     .             data_grid(a_lo_x,a_lo_y,use_lo(3),l),
     .             ( (use_hi(3)-use_lo(3)+1)
     .             * (use_hi(2)-use_lo(2)+1)
     .             * (use_hi(1)-use_lo(1)+1) ), cptype )

           ELSE
              DO 1000 k = use_lo(3), use_hi(3)
              DO 1000 j = use_lo(2), use_hi(2)
 1000         CALL TM_FP_CONVERT( data_grid(use_lo(1),j,k,l),
     .                           (use_hi(1)-use_lo(1)+1), cptype )
           ENDIF
        ENDIF

* successful completion
	status = merr_ok
	GOTO 9900
* ERROR branches
9000	CALL TM_ERRMSG( merr_erreq, status, 'GT_READ',
     .			dset_num, sf_num,
     .			no_errstring, no_errstring, *9900)

9100	CALL TM_ERRMSG( merr_eof, status, 'GT_READ',
     .			dset_num, sf_num,
     .			'Incorrect variable bounds in descriptor file',
     .			no_errstring, *9900)

* bye bye
9900	RETURN
	END
