	SUBROUTINE TM_READ ( dset_num, variable,
     .		a_lo_x, a_lo_y, a_lo_z, a_lo_t, a_lo_e, a_lo_f,
     .		a_hi_x, a_hi_y, a_hi_z, a_hi_t, a_hi_e, a_hi_f,
     .		use_lo, use_hi, stride,
     .		data_grid, file_num, n2feat, status )
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
*
* Determines if a read request for a nferdims-D block of data (3D, 2D being degenerate
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
* *sh* 7/98 V4.91 - want to allow case-sensitive variables in single quotes
*       so dont upcase the variable name when comparing (why was it
*       upcased, anyway??)
* V500 4/99 *kob*: increase upcase_var 24 --> 64 chars matching passed vname
*      5/99 *sh* - added strides to CD_READ call
* V552 5/03 *acm*: increase size of upcase_var from 64 to 128 chars
* V602 9/06 *jli*  Changes for native strides
* V674 2/12 *acm* 6D Ferret: use nferdims rather than 4 for dimension indexing
*
* include files
* include files
        include 'tmap_dims.parm'                ! data set dimension
#include "gt_lib.parm"
        include 'tmap_errors.parm'              ! err code parm defs
	include 'tmap_dtypes.parm'
        include 'xtm_grid.cmn_text'             ! grid info
                include 'xdset_info.cmn_text'   ! data set info common
        external xdset_info_data

        include 'xdyn_linemem.cmn_text'

* arguments:
*	dset_num    - internal data set # (from TM_INIT_DSET)
*	variable    - name of desired variable in data set
*	actual_dims - "a_" full dimensions of data_grid array passed
*	use_dims    - indices of model data block desired
*	data_grid   - array containing model data (output)
*	file_num    - file number (output)
*	n2feat	    - IF > 0, then we are reading station 
*			or traj info as instance data for 
*			timeseriesprofile or trajectoryprofile
*	status      - error status upon completion (output)
*
	CHARACTER*(*)	variable
	INTEGER		dset_num, file_num, n2feat, status
	INTEGER		a_lo_x, a_lo_y, a_lo_z, a_lo_t, a_lo_e, a_lo_f, 
     .                  a_hi_x, a_hi_y, a_hi_z, a_hi_t, a_hi_e, a_hi_f,
     .			use_lo(*), use_hi(*), stride(*)
        REAL            data_grid(a_lo_x:a_hi_x,
     .                            a_lo_y:a_hi_y,
     .                            a_lo_z:a_hi_z,
     .                            a_lo_t:a_hi_t,
     .                            a_lo_e:a_hi_e,
     .                            a_lo_f:a_hi_f)

* local definitions
        INTEGER         STR_UPCASE, TM_LENSTR1, n, num_indices

	INTEGER		lcnt, modline, ivar, varpos, pcnt, istat,
     .			i, j, k, l, in, jn, kn, ln, vargrid, thisline,
     .			Dstart, Dend, Drange,
     .			Rstart, Rend, Rrange,
     .			Tstart, Tend, Trange,
     .			nprof, index_lm, ista, iprof 

	INTEGER		i_lo, j_lo, k_lo, l_lo, m_lo, n_lo, 
     .			i_hi, j_hi, k_hi, l_hi, m_hi, n_hi,
     .			iadj, jadj, kadj, ladj, madj, nadj
	INTEGER		tmp_lo(nferdims), tmp_hi(nferdims), 
     .			adj_lo(nferdims), adj_hi(nferdims),
     .			adj(nferdims), actual_lo(nferdims), 
     .			actual_hi(nferdims)
        CHARACTER*128    upcase_var
	EQUIVALENCE	(tmp_lo(1),i_lo),
     .			(tmp_lo(2),j_lo),
     .			(tmp_lo(3),k_lo),
     .			(tmp_lo(4),l_lo),
     .			(tmp_lo(5),m_lo),
     .			(tmp_lo(6),n_lo)
	EQUIVALENCE	(tmp_hi(1),i_hi),
     .			(tmp_hi(2),j_hi),
     .	  		(tmp_hi(3),k_hi),
     .			(tmp_hi(4),l_hi),
     .			(tmp_hi(5),m_hi),
     .			(tmp_hi(6),n_hi)
	EQUIVALENCE	(adj(1),iadj),
     .			(adj(2),jadj),
     .			(adj(3),kadj),
     .			(adj(4),ladj),
     .			(adj(5),madj),
     .			(adj(6),nadj)


* initialize


	actual_lo(1) = a_lo_x
	actual_lo(2) = a_lo_y
	actual_lo(3) = a_lo_z
	actual_lo(4) = a_lo_t
	actual_lo(5) = a_lo_e
	actual_lo(6) = a_lo_f
	actual_hi(1) = a_hi_x
	actual_hi(2) = a_hi_y
	actual_hi(3) = a_hi_z
	actual_hi(4) = a_hi_t
	actual_hi(5) = a_hi_e
	actual_hi(6) = a_hi_f
!        istat = STR_UPCASE( upcase_var, variable )     ! 4/91
        upcase_var = variable     ! 7/98

* check for valid variable and get variable number
*	DO 100 ivar = 1,maxvars
*	  IF ((ds_var_setnum(ivar) .EQ. dset_num) .AND.
*     .	    (ds_var_code(ivar) .EQ. upcase_var)) THEN
*	    varpos = ivar
*	    GOTO 110
*	  ENDIF
*  100	CONTINUE

        CALL string_array_find_exact(ds_var_code_head,
     .                               upcase_var,
     .                               LEN(upcase_var),
     .                               string_array_result1(1),
     .                               maxvars,
     .                               num_indices)
	DO 100 n = 1, num_indices
            ivar = string_array_result1(n)
            IF ( ds_var_setnum(ivar) .EQ. dset_num) THEN
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
	DO 120 lcnt = 1,nferdims
	  thisline = grid_line(lcnt,vargrid)
	  IF (thisline .EQ. mpsunkn .OR. thisline .EQ. mpsnorm) GOTO 120
	  IF (line_modulo(thisline)) THEN

	    IF (use_lo(lcnt) .GE. 1 .AND.
!     .	        use_hi(lcnt) .LE. line_dim(thisline)) GOTO 120
     .         use_hi(lcnt) .LE. ds_grid_end(lcnt,varpos)) GOTO 120

	    IF (modline .NE. 0) GOTO 9100
	    modline = lcnt
	  ENDIF
  120	CONTINUE
	
	IF (modline .EQ. 0) THEN

	  nprof = a_hi_e
	  IF (n2feat .GT. 0 .AND. a_hi_f.EQ.1) a_hi_e = n2feat

	  CALL TM_CHOOSE_READ ( dset_num, varpos,
     .		a_lo_x, a_lo_y, a_lo_z, a_lo_t, a_lo_e, a_lo_f,
     .		a_hi_x, a_hi_y, a_hi_z, a_hi_t, a_hi_e, a_hi_f,
     .		use_lo, use_hi, stride,
     .		data_grid, file_num, status )
 

* If the trajectoryProfile or timeseriesProfile data is being
* handle as simple profile data,  put the 1..nstation or 
* 1..ntrajectory data into 1..profile

	  IF (n2feat.GT.0  .AND.  dsg_orientation(dset_num).EQ.z_dim) THEN
	     index_lm = dsg_loaded_lm(dsg_index_var(dset_num))

	     IF (ds_var_type(ivar) .EQ. ptype_string) THEN

* Read in the last slots of data to allocate string memory
* 
	        a_hi_e = nprof-n2feat
		CALL TM_CHOOSE_READ ( dset_num, varpos,
     .	 	  a_lo_x, a_lo_y, a_lo_z, a_lo_t, a_lo_e, a_lo_f,
     .		  a_hi_x, a_hi_y, a_hi_z, a_hi_t, a_hi_e, a_hi_f,
     .		  use_lo, use_hi, stride,
     .		  data_grid(a_lo_x,a_lo_y,a_lo_z,a_lo_t,n2feat+1,a_lo_f), 
     .		  file_num, status )

	        DO iprof = nprof, 2, -1

	           ista = dsg_linemem(index_lm)%ptr(iprof) + 1
		   CALL COPY_C_STRING(
     .		   data_grid(a_lo_x,a_lo_y,a_lo_z,a_lo_t,ista,a_lo_f), 
     .		   data_grid(a_lo_x,a_lo_y,a_lo_z,a_lo_t,iprof,a_lo_f) )

	        ENDDO

	     ELSE

	        DO iprof = nprof, 2, -1

	           ista = dsg_linemem(index_lm)%ptr(iprof) + 1
	           data_grid(a_lo_x,a_lo_y,a_lo_z,a_lo_t,iprof,a_lo_f) = 
     .		   data_grid(a_lo_x,a_lo_y,a_lo_z,a_lo_t,ista,a_lo_f)
	        
		ENDDO

	     ENDIF

	     a_hi_e = nprof

	  ENDIF


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
	  DO 130 pcnt = 1,nferdims
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
     .		adj_lo(1),adj_lo(2),adj_lo(3),adj_lo(4),adj_lo(5),adj_lo(6),
     .		adj_hi(1),adj_hi(2),adj_hi(3),adj_hi(4),adj_hi(5),adj_hi(6),
     .		tmp_lo, tmp_hi, stride,
     .		data_grid, file_num, status )
	  IF (status .NE. merr_ok) GOTO 9999

* Read #2 - optional depending on length & position of Requested grid
	  IF (Tend .GT. Dend .AND. Tstart .NE. 1) THEN
	    tmp_lo(modline) = Dstart
	    tmp_hi(modline) = MIN(Tstart-1,Tend-Drange)
	    adj_lo(modline) = adj_lo(modline) - Drange
	    adj_hi(modline) = adj_hi(modline) - Drange
	    CALL TM_CHOOSE_READ ( dset_num, varpos,
     .		adj_lo(1),adj_lo(2),adj_lo(3),adj_lo(4),adj_lo(5),adj_lo(6),
     .		adj_hi(1),adj_hi(2),adj_hi(3),adj_hi(4),adj_hi(5),adj_hi(6),
     .		tmp_lo, tmp_hi, stride,
     .		data_grid, file_num, status )
	    IF (status .NE. merr_ok) GOTO 9999
	  ENDIF

* Data duplication (NOTE: tmp_lo,tmp_hi are equivalenced with i,j,k,l _lo & _hi)
*                  (NOTE: adj is equivalenced with i,j,k,l _adj)
	  IF (Trange .GT. Drange) THEN
	    tmp_lo(modline) = Rstart+Drange
	    tmp_hi(modline) = Rend
	    DO 140 pcnt = 1,nferdims
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
	            data_grid(in,jn,kn,ln,-999,-999) = 
     .			   data_grid(i,j,k,l,-999,-999)	
  150	          CONTINUE
  160	        CONTINUE
  170	      CONTINUE
  180	    CONTINUE
	  ENDIF
	ENDIF



*	n2feat	    - IF > 0, then we are reading station 
*			or traj info as instance data for 
*			timeseriesprofile or trajectoryprofile

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
