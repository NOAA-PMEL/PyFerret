	SUBROUTINE TS_READ ( dset_num, varpos,
     .			     a_lo_x, a_lo_y, a_lo_z, a_lo_t,
     .			     a_hi_x, a_hi_y, a_hi_z, a_hi_t,
     .			     use_lo, use_hi,
     .			     data_grid, vf_num, status )
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
* Reads a 4-D block of data (3D, 2D being degenerate cases w/ 1 dimension being
* 1 time series) from a TMAP time-series direct access file.
*
* written 08/22/88 by M. Verschell for PMEL/TMAP
*
* revision 0.00 - 08/22/88 - initial incarnation
* revision 0.10 - 12/06/89 - shanged arrays for dimensions to single variables
* revision 0.20 - 02/05/90 - *sh* corrected bug in multiple-record time series
*			     accesses; also increased efficiency through not
*			     reading beyond required records in each
*			     multi-record group
* Sun port 2/92 *sh* - convert between various machine word types
* 10/96 *kob* - Linux port - had to move variable assignments off of
*		declaration line.

* arguments:
*	dset_num    - internal data set # (from GT_INIT_DSET)
*	varpos      - position in common of variable information
*	actual_dims - "a_" full dimensions of data_grid array passed
*	use_dims    - indices of model data block desired
*	data_grid   - array containing model data (output)
*	vf_num      - varfile number (returned from TS_FIND_VAR (output)
*	status      - error status upon completion (output)
*
	INTEGER		dset_num, varpos, vf_num, status
	INTEGER		a_lo_x, a_lo_y, a_lo_z, a_lo_t,
     .			a_hi_x, a_hi_y, a_hi_z, a_hi_t,
     .			use_lo(4), use_hi(4)
	REAL		data_grid(a_lo_x:a_hi_x,
     .				  a_lo_y:a_hi_y,
     .				  a_lo_z:a_hi_z,
     .				  a_lo_t:a_hi_t)

* include files
#ifdef unix
	include 'tmap_format/tmap_dims.parm'		! data set dimension
	include 'tmap_format/tmap_errors.parm'		! err code parm defs
	include 'tmap_format/gt_lib.parm'		! axis setup parameters
	include 'tmap_format/xdset_info.cmn_text'	! data set info common
	external xdset_info_data
	include 'tmap_format/xstep_files.cmn_text'	! step file info common
	external xstep_files_data
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
	INCLUDE 'TMAP_FORMAT:XMACHINE_BYTE.CMN'
#endif

* local definitions
* *kob* explicit definitions not allowed on decl line in linux...
*       moved it to parameter line 10/96
        LOGICAL         cp_convert
	CHARACTER*1	char_dim(4)
	DATA	        char_dim/'I','J','K','L'/
	CHARACTER*4	rec_code
	CHARACTER*8	variable
	CHARACTER*13	TM_STRING
	INTEGER*4	ndim, lunit, varstrt, posf, posn, nposn, cptype,
     .			rec0, rec1, nt_inrec, vgrid, first_t, last_t,
     .			start_time, end_time,
     .			i,     j,     k,     l,
     .			istrt, jstrt, kstrt, lstrt,
     .			iend,  jend,         lend,
     .			xfull, yfull,        tfull,
     .			irecs, jrecs, krecs,
     .			       jcnt,  kcnt,
     .			fst_pos, las_pos
	REAL*4		xw, yw, zw, dummy
	REAL*8		TM_WORLD

* Initialize by locating correct file and starting record of variable
* in addition to variables name in common
	variable = ds_var_code(varpos)
	CALL TS_FIND_VAR ( dset_num, varpos, vf_num,
     .			   varstrt, status )
	IF (status .NE. merr_ok) GOTO 9900

* check requested grid to see if it lies in data set
	  DO 100 ndim = 1,4
	    IF (use_lo(ndim) .GT. use_hi(ndim))
     .	       CALL TM_ERRMSG (merr_badsubscr,status,
     .	                       'TS_READ',dset_num,no_stepfile,
     .		               'for requested '//char_dim(ndim)//
     .	                       ' subscripts',no_errstring,*9900)
	    IF ( use_lo(ndim) .LT. ds_grid_start(ndim,varpos) .OR.
     .	         use_hi(ndim) .GT. ds_grid_end(ndim,varpos) )
     .	      CALL TM_ERRMSG (merr_outofgrid,status,
     .	                      'TS_READ',dset_num,no_stepfile,
     .	                      'VAR='//variable//char_dim(ndim)//'='//
     .	                      TM_STRING(DBLE(use_lo(ndim)))//'-'//
     .	                      TM_STRING(DBLE(use_hi(ndim))),
     .	                      no_errstring,*9900)
  100	  CONTINUE

* grid start and end in data file
	lstrt = ds_grid_start(t_dim,varpos)
	lend  = ds_grid_end(t_dim,varpos)
	kstrt = ds_grid_start(z_dim,varpos)
	jstrt = ds_grid_start(y_dim,varpos)
	jend  = ds_grid_end(y_dim,varpos)
	istrt = ds_grid_start(x_dim,varpos)
	iend  = ds_grid_end(x_dim,varpos)

* general file info
	nt_inrec = ds_ndataperrec(varpos)
	tfull = lend - lstrt + 1
	IF (MOD(tfull,nt_inrec) .NE. 0) THEN
	  nposn = 1 + tfull/nt_inrec
	ELSE
	  nposn = tfull/nt_inrec
	ENDIF
	vgrid = ds_grid_number(varpos)
	xfull = iend - istrt + 1
	yfull = jend - jstrt + 1
	kcnt  = nposn*xfull*yfull
	jcnt  = nposn*xfull
* Get stuff  from COMMON/XSTEP_FILES/
	lunit = vf_lunit(vf_num)
	cptype = vf_cpu_type(vf_num)
	cp_convert = cptype .NE. (active_cpu)

* determine first and last position to read from each time series
* (determine in advance to avoid re-computing at each XYZ grid location)
	fst_pos = (use_lo(t_dim)-lstrt)/nt_inrec + 1
	las_pos = (use_hi(t_dim)-lstrt)/nt_inrec + 1

* Read in data
	DO 140 k = use_lo(z_dim), use_hi(z_dim)
	  krecs = (k-kstrt)*kcnt
	  DO 130 j = use_lo(y_dim), use_hi(y_dim)
	    jrecs = krecs + (j-jstrt)*jcnt
	    DO 120 i = use_lo(x_dim), use_hi(x_dim)
	      irecs = jrecs + (i-istrt)*nposn
	      rec0 = varstrt + irecs
	      start_time = lstrt + (fst_pos-1)*nt_inrec
	      end_time = start_time + nt_inrec - 1
	      DO 110 posn = fst_pos, las_pos
	        first_t = MAX(use_lo(t_dim),start_time)
	        last_t = MIN(use_hi(t_dim),end_time)
	        rec1 = rec0+posn
	        READ (lunit, REC=rec1, ERR=9000) rec_code,xw,yw,zw,posf,
     .				   (dummy,l=1,use_lo(t_dim)-start_time),
     .				   (data_grid(i,j,k,l),l=first_t,last_t)

* Note: machines with a different endian will require seq_num to be repaired
	        IF ( cp_convert ) CALL TM_INT4_SWAP( posf, cptype )
	        IF ( rec_code .NE. 'POS ' .OR. posf .NE. posn) GOTO 9100

	        start_time = start_time + nt_inrec
	        end_time = MIN(end_time+nt_inrec,lend)
  110	      CONTINUE
              IF ( cp_convert ) CALL TM_FP_CONVERT( xw, 1, cptype )
	      IF (xw .NE. SNGL(TM_WORLD(i,vgrid,1,box_middle))) GOTO 9100
  120	    CONTINUE
            IF ( cp_convert ) CALL TM_FP_CONVERT( yw, 1, cptype )
	    IF (yw .NE. SNGL(TM_WORLD(j,vgrid,2,box_middle))) GOTO 9100
  130	  CONTINUE
          IF ( cp_convert ) CALL TM_FP_CONVERT( zw, 1, cptype )
	  IF (zw .NE. SNGL(TM_WORLD(k,vgrid,3,box_middle))) GOTO 9100
  140	CONTINUE

* if file was written on a cpu with a different binary format fix things up
        IF ( cp_convert ) THEN
           DO 1000 l = use_lo(4), use_hi(4)
           DO 1000 k = use_lo(3), use_hi(3)
           DO 1000 j = use_lo(2), use_hi(2)
 1000      CALL TM_FP_CONVERT( data_grid(use_lo(1),j,k,l),
     .                           (use_hi(1)-use_lo(1)+1), cptype )
        ENDIF

* successful completion
	status = merr_ok
	GOTO 9900

* ERROR branches
9000	CALL TM_ERRMSG( merr_erreq, status, 'TS_READ',
     .			dset_num, vf_num,
     .			no_errstring, no_errstring, *9900)

9100	CALL TM_ERRMSG (merr_filpos, status, 'TS_READ',
     .		dset_num, vf_num,
     .		ds_var_code(varpos)//' RECNUM='//TM_STRING(DBLE(rec1)),
     .		no_errstring, *9900)

* bye bye
9900	RETURN
	END
