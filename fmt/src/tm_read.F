	SUBROUTINE TM_READ ( dset_num, variable,
     .			     a_lo_x, a_lo_y, a_lo_z, a_lo_t,
     .			     a_hi_x, a_hi_y, a_hi_z, a_hi_t,
     .			     use_lo, use_hi,
     .			     data_grid, file_num, status )
*
* Determines if a read request for a 4-D block of data (3D, 2D being degenerate
* cases w/ 1 dimension being 1 time series) from TMAP data sets contains a 
* modulo axis. Arranges calls to TM_CHOOSE_READ accordingly.
*
* written 10/24/88 by M. Verschell for PMEL/TMAP
*
* revision 0.00	- 10/24/88 - initial incarnation
* revision 0.01 - 12/15/88 - changed variable to *(*) from *8
* revision 0.10 - 01/24/89 - Bug fixes to fully implement modulo axis
* revision 0.11 - 02/08/89 - Bug fix to 2nd modulo read IF-test
* revision 0.20 - 02/08/89 - TM_READ now finds modulo axis and checks for 1 only
* revision 0.30 - 12/06/89 - changed arrays for dimensions to single variables
* revision 0.40 - 02/27/90 - multiple modulo axis provided only 1 used in read
* revision 0.41 - 4/91 *sh* Unix/RISC port: upcase the variable code and use
*                           in error message
* revision 0.5  - 11/92 *sh* increased variable name lengths to 24 characters

* arguments:
*	dset_num    - internal data set # (from TM_INIT_DSET)
*	variable    - name of desired variable in data set
*	actual_dims - "a_" full dimensions of data_grid array passed
*	use_dims    - indices of model data block desired
*	data_grid   - array containing model data (output)
*	file_num    - file number (output)
*	status      - error status upon completion (output)
*
	CHARACTER*(*)	variable
	INTEGER		dset_num, file_num, status
	INTEGER		a_lo_x, a_lo_y, a_lo_z, a_lo_t,
     .                  a_hi_x, a_hi_y, a_hi_z, a_hi_t,
     .			use_lo(4), use_hi(4)
        REAL            data_grid(a_lo_x:a_hi_x,
     .                            a_lo_y:a_hi_y,
     .                            a_lo_z:a_hi_z,
     .                            a_lo_t:a_hi_t)
*
* include files
#ifdef unix
	include 'tmap_format/tmap_dims.parm'		! data set dimension
	include 'tmap_format/tmap_errors.parm'		! err code parm defs
	include 'tmap_format/gt_lib.parm'		! axis setup parameters
	include 'tmap_format/xtm_grid.cmn_text'		! grid info
	external xgt_grid_data
	include 'tmap_format/xdset_info.cmn_text'	! data set info common
	external xdset_info_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'		! Data set dimension
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'		! err code PARM defs
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'		! Axis setup parameters
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'		! Grid info
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'		! Data set info common
#endif

* local definitions
        INTEGER         STR_UPCASE, TM_LENSTR1
	INTEGER		lcnt, modline, ivar, varpos, pcnt, istat,
     .			i, j, k, l, in, jn, kn, ln, vargrid, thisline,
     .			Dstart, Dend, Drange,
     .			Rstart, Rend, Rrange,
     .			Tstart, Tend, Trange
	INTEGER		i_lo, j_lo, k_lo, l_lo,
     .			i_hi, j_hi, k_hi, l_hi,
     .			iadj, jadj, kadj, ladj
	INTEGER		tmp_lo(4), tmp_hi(4), adj_lo(4), adj_hi(4),
     .			adj(4), actual_lo(4), actual_hi(4)
        CHARACTER*24    upcase_var     ! (was *8 until 11/92 *sh*)
	EQUIVALENCE	(tmp_lo(1),i_lo),
     .			(tmp_lo(2),j_lo),
     .			(tmp_lo(3),k_lo),
     .			(tmp_lo(4),l_lo),
     .			(tmp_hi(1),i_hi),
     .			(tmp_hi(2),j_hi),
     .	  		(tmp_hi(3),k_hi),
     .			(tmp_hi(4),l_hi),
     .			(adj(1),iadj),
     .			(adj(2),jadj),
     .			(adj(3),kadj),
     .			(adj(4),ladj)

* initialize
	actual_lo(1) = a_lo_x
	actual_lo(2) = a_lo_y
	actual_lo(3) = a_lo_z
	actual_lo(4) = a_lo_t
	actual_hi(1) = a_hi_x
	actual_hi(2) = a_hi_y
	actual_hi(3) = a_hi_z
	actual_hi(4) = a_hi_t
        istat = STR_UPCASE( upcase_var, variable )     ! 4/91

* check for valid variable and get variable number
	DO 100 ivar = 1,maxvars
	  IF ((ds_var_setnum(ivar) .EQ. dset_num) .AND.
     .	    (ds_var_code(ivar) .EQ. upcase_var)) THEN
	    varpos = ivar
	    GOTO 110
	  ENDIF
  100	CONTINUE
	GOTO 9000

* Is it a Modulo set - If not call TM_CHOOSE_READ w/o modification
* if it is - get the modulo line, and make sure at this time there's only 1
* axis needing a modulo read call.
  110	vargrid = ds_grid_number(varpos)
	modline = 0
	DO 120 lcnt = 1,4
	  thisline = grid_line(lcnt,vargrid)
	  IF (thisline .EQ. mpsunkn .OR. thisline .EQ. mpsnorm) GOTO 120
	  IF (line_modulo(thisline)) THEN
	    IF (use_lo(lcnt) .GE. 1 .AND.
     .	        use_hi(lcnt) .LE. line_dim(thisline)) GOTO 120
	    IF (modline .NE. 0) GOTO 9100
	    modline = lcnt
	  ENDIF
  120	CONTINUE
	
	IF (modline .EQ. 0) THEN
	  CALL TM_CHOOSE_READ ( dset_num, varpos,
     .			 a_lo_x, a_lo_y, a_lo_z, a_lo_t,
     .			 a_hi_x, a_hi_y, a_hi_z, a_hi_t,
     .			 use_lo, use_hi,
     .			 data_grid, file_num, status )
	  IF (status .NE. merr_ok) GOTO 9999

* It's modulo : Set-up
* D = Defining Axis    : the axis that the data is stored on
* R = Requested Axis   : the axis requested by the user (not necessarily in the
*                        range of the Defining axis)
* T = Transformed Axis : the R axis translated by a MOD function so that it
*                        starts somewhere in the Defining axis
	ELSE
	  Dstart = 1
	  Drange = line_dim(grid_line(modline,vargrid))
	  Dend = Drange - Dstart + 1
	  Rstart = use_lo(modline)
	  Rend = use_hi(modline)
	  Rrange = Rend - Rstart + 1
	  Tstart = MOD(Rstart-1,Drange) + 1
	  IF (Tstart .LE. 0) Tstart = Tstart + Drange
	  Tend = Tstart + Rrange - 1
	  Trange = Rrange

* Read #1 - always done
	  DO 130 pcnt = 1,4
	    tmp_lo(pcnt) = use_lo(pcnt)
	    tmp_hi(pcnt) = use_hi(pcnt)
	    adj_lo(pcnt) = actual_lo(pcnt)
	    adj_hi(pcnt) = actual_hi(pcnt)
  130	  CONTINUE
	  tmp_lo(modline) = Tstart
	  tmp_hi(modline) = MIN(Dend,Tend)
	  adj_lo(modline) = actual_lo(modline) - (Rstart - Tstart)
	  adj_hi(modline) = actual_hi(modline) - (Rstart - Tstart)
	  CALL TM_CHOOSE_READ ( dset_num, varpos,
     .			 adj_lo(1),adj_lo(2),adj_lo(3),adj_lo(4),
     .			 adj_hi(1),adj_hi(2),adj_hi(3),adj_hi(4),
     .			 tmp_lo, tmp_hi,
     .			 data_grid, file_num, status )
	  IF (status .NE. merr_ok) GOTO 9999

* Read #2 - optional depending on length & position of Requested grid
	  IF (Tend .GT. Dend .AND. Tstart .NE. 1) THEN
	    tmp_lo(modline) = Dstart
	    tmp_hi(modline) = MIN(Tstart-1,Tend-Drange)
	    adj_lo(modline) = adj_lo(modline) - Drange
	    adj_hi(modline) = adj_hi(modline) - Drange
	    CALL TM_CHOOSE_READ ( dset_num, varpos,
     .			   adj_lo(1),adj_lo(2),adj_lo(3),adj_lo(4),
     .			   adj_hi(1),adj_hi(2),adj_hi(3),adj_hi(4),
     .			   tmp_lo, tmp_hi,
     .			   data_grid, file_num, status )
	    IF (status .NE. merr_ok) GOTO 9999
	  ENDIF

* Data duplication (NOTE: tmp_lo,tmp_hi are equivalenced with i,j,k,l _lo & _hi)
*                  (NOTE: adj is equivalenced with i,j,k,l _adj)
	  IF (Trange .GT. Drange) THEN
	    tmp_lo(modline) = Rstart+Drange
	    tmp_hi(modline) = Rend
	    DO 140 pcnt = 1,4
	      adj(pcnt) = 0
  140	    CONTINUE
	    adj(modline) = Drange
	    DO 180 ln = l_lo,l_hi
	      l = ln - ladj
	      DO 170 kn = k_lo,k_hi
	        k = kn - kadj
	        DO 160 jn = j_lo,j_hi
	          j = jn - jadj
	          DO 150 in = i_lo,i_hi
	            i = in - iadj
	            data_grid(in,jn,kn,ln) = data_grid(i,j,k,l)	
  150	          CONTINUE
  160	        CONTINUE
  170	      CONTINUE
  180	    CONTINUE
	  ENDIF
	ENDIF
	status = merr_ok
	GOTO 9999

* ERRORS
* note: use upcase_var instead of "variable" to avoid RISC // prob. w/ CHAR*(*)
 9000   l = TM_LENSTR1(upcase_var)
	CALL TM_ERRMSG (merr_unkvar, status, 'TM_READ',
     .			dset_num, no_stepfile,
     .			'VAR='//upcase_var(:l), no_errstring, *9999)
Cf77     .			'VAR='//variable, no_errstring, *9999)
 9100	CALL TM_ERRMSG (merr_notsupport, status, 'TM_READ',
     .			dset_num, no_stepfile,
     .			'only 1 axis at a time can have a modulo read',
     .			no_errstring, *9999)

 9999	RETURN
	END
