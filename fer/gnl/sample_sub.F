	SUBROUTINE SAMPLE_SUB(  memory, cx_list, mr_list, npts, cxpts,
     .				field, mfield, cxfield, fill_value,
     .				coords, out_buf,
     .				fmt, fix_holes, ssorww,
     .				out_lun, binary, columns, status )

* sample the given field at the specified XYZT coordinates

* Each output record will consist of
*	value xcoord ycoord zcoord tcoord code
* where "code" is 0=fully successful
*		  1=successful with extrapolation
*              -1-4=failed due to beyond input data limits on axis -code
*		 -9=failed due to hole in input data field
*		-99=failed due to hole in input coordinates

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V301:  1/94
* V312: 5/94 - array "memory" as a calling argument

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xmem_subsc.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN_TEXT'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XMEM_SUBSC.CMN'
#endif

* calling argument declarations:
	LOGICAL fix_holes, binary
	INTEGER	cx_list(4), mr_list(4), npts, cxpts(4), columns,
     .		mfield, cxfield, ssorww(4), out_lun, status
	REAL	memory( mem_blk_size, max_mem_blks )
	REAL	coords(npts, 4), out_buf(columns), fill_value,
     .		field( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit )
	CHARACTER fmt*(*)

* internal parameter definitions
	INTEGER     by_ss, by_ww, by_nothin
	PARAMETER ( by_ss=1, by_ww=2, by_nothin=0 )

* internal variable declarations:
	LOGICAL	extrapolate, condensed
	INTEGER	ii(4), di(4), ss(4,2), grid, idim, mr, mv_temp, ipt,
     .		i, j, k, l, ix, jy, kz, lt, hi_surrnd(4), nbuf,
     .		nfix, ssfix(4,-1:1), ssp1(4), ssm1(4), line, lohi
	REAL lin_fact(4), surrnd(2,2,2,2), result, missing, res_flag,
     .		coord, missing_coord(4), sumfix
	
* equivalences
	INTEGER ss_lo(4), ss_hi(4), ihi, jhi, khi, lhi
	EQUIVALENCE (ss(1,1),ss_lo), (ss(1,2),ss_hi),
     .		    (hi_surrnd(1), ihi), (hi_surrnd(2), jhi),
     .		    (hi_surrnd(3), khi), (hi_surrnd(4), lhi)

* *** INITIALIZE ***
* background quantities
	condensed = columns .GT. 0
	nbuf = 0
	missing = mr_bad_data(mfield)
	grid = mr_grid(mfield)
	DO 10 idim = 1, 4
 10	missing_coord(idim) = mr_bad_data(mr_list(idim))

* extract the 4 coordinate variables into a single 4D array
	DO 100 idim = 1, 4
	   mr = mr_list(idim)
	   CALL CREATE_TEMP_MEM_VAR( cx_list(idim), mv_temp, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   CALL COPY_GRID(memory(1,mr_blk1(mr)),mr,
     .		          coords(1,idim), mv_temp )
	   CALL DELETE_VARIABLE( mv_temp )
 100	CONTINUE

* input sampling coordinates are a mix of single point and n point arrays
* ... ii - sample coordinate pointers (increment by delta each loop)
* ... di - delta for sample coordinate pointer (0 or 1)
	DO 200 idim = 1, 4
	   ii(idim) = 1
	   IF ( cxpts(idim) .EQ. 1 ) THEN
	      di(idim) = 0
	   ELSE
	      di(idim) = 1
	   ENDIF
 200	CONTINUE

* prepare those axes on which there will be no interpolating
* ... hi_surrnd - # of pts surrounding requested coord on each axis (1 or 2)
* ... also preload the subscript limits to avoid redoing it npts times below
	DO 210 idim = 1, 4
	   IF ( ssorww(idim) .EQ. by_nothin ) THEN
	      hi_surrnd(idim) = 1		! no interp on this axis
	      ss_lo(idim) = cx_lo_ss(cxfield, idim)
	      ss_hi(idim) = ss_lo(idim)			! needed to automate
	      lin_fact(idim) = unspecified_int4		! not needed
	   ELSE
	      hi_surrnd(idim) = 2		! interpolate on this axis
	      ss_lo(idim) = unspecified_int4
	   ENDIF
 210	CONTINUE

* *** START OF LOOP OVER ALL SAMPLING POINTS ***
	DO 2000 ipt = 1, npts

* initialize next sample point
	   res_flag = 0.

* determine the lo and hi subscript bounds of the surrounding points
* and the interpolation cooeficients for each axis
	   DO 500 idim = 1, 4
	      coord = coords(ii(idim), idim)
	      IF ( ssorww(idim) .NE. by_nothin
     .	     .AND. coord .EQ. missing_coord(idim) ) THEN
	         res_flag = -idim
	         result = fill_value
	         GOTO 1000
	      ELSEIF ( ssorww(idim) .EQ. by_ss ) THEN   ! e.g. j=55.2
	         ss_lo(idim) = INT( coord )
	         ss_hi(idim) = INT( coord + .99999 )
	         lin_fact(idim) = ss_hi(idim) - coord
	      ELSEIF ( ssorww(idim) .EQ. by_ww ) THEN
	         CALL SS_NEIGHBORS( coord, grid, idim, extrapolate,
     .				ss_lo(idim), ss_hi(idim), lin_fact(idim) )
	         IF ( extrapolate ) THEN
	            IF ( fix_holes ) THEN
	               res_flag = 1.
	            ELSE
	               res_flag = -idim
	               result = fill_value
	               GOTO 1000
	            ENDIF
	         ENDIF
	      ENDIF
 500	   CONTINUE

* check that all subscripts are within the given range of the field
	   DO 600 idim = 1, 4
	      IF ( ss_lo(idim) .LT. cx_lo_ss(cxfield, idim)
     .	      .OR. ss_hi(idim) .GT. cx_hi_ss(cxfield, idim)
     .        .OR. ss_lo(idim) .GT. ss_hi(idim)  ) THEN
	         res_flag = -idim
	         result = fill_value
	         GOTO 1000
	      ENDIF
 600	   CONTINUE

* pre-cleanse array if diagnostic testing in progress
	   IF ( mode_diagnostic ) THEN
	      DO 650 i = 1, ihi
	      DO 650 j = 1, jhi
	      DO 650 k = 1, khi
	      DO 650 l = 1, lhi
 650	      surrnd(i,j,k,l) = bad_val4
	   ENDIF

* assemble the 4-dimensional collection of surrounding points
	   DO 700 i = 1, ihi
	   DO 700 j = 1, jhi
	   DO 700 k = 1, khi
	   DO 700 l = 1, lhi
 700	   surrnd(i,j,k,l) = field(ss(x_dim,i),
     .				      ss(y_dim,j),
     .				      ss(z_dim,k),
     .				      ss(t_dim,l) )
! diag	WRITE (6,'(A/,4(1PG15.6))') ' Surrounding: ',surrnd

* check for missing values - try to fill them in if requested
	   DO 850 l = 1, lhi
	   DO 850 k = 1, khi
	   DO 850 j = 1, jhi
	   DO 850 i = 1, ihi
	      IF ( surrnd(i,j,k,l) .EQ. missing ) THEN
	         IF ( fix_holes ) THEN
* ... Try to fill hole with the average of all surrounding values
*     Procedure is to locate the subscripts surrounding the missing point
*     that may contain valid data and to average all of the data found.
*     The axes may be modulo - in which case wrap around the subscripts.
*     The supplied data (context) may or may not provide the surrounding pts   
	            nfix = 0
	            sumfix = 0.0		
	            DO 800 idim =  1, 4
	            DO 800 lohi = -1, 1
 800	            ssfix(idim,lohi) = ss(idim,i) + lohi
	            DO 810 idim = 1, 4    	 ! modulo-based subscripts
	               line = grid_line(idim,grid)
	               IF ( line.EQ.mnormal .OR. line.EQ.munknown ) GOTO 810
	               IF ( line_modulo(line) ) THEN
		          IF ( ssfix(idim,-1) .EQ. 0 )
     .					ssfix(idim,-1) = line_dim(line)
		          IF ( ssfix(idim,+1) .GT. line_dim(line) )
     .					ssfix(idim,+1) = 1
	               ENDIF
 810	            CONTINUE
	            DO 820 idim = 1, 4		! check context limits
	               IF (ssfix(idim,-1).LT.cx_lo_ss(cxfield,idim)
     .		       .OR.ssfix(idim,-1).GT.cx_hi_ss(cxfield,idim) ) THEN
	                  ssm1(idim) = 0
	               ELSE
	                  ssm1(idim) = -1
	               ENDIF
	               IF (ssfix(idim,+1).LT.cx_lo_ss(cxfield,idim)
     .		       .OR.ssfix(idim,+1).GT.cx_hi_ss(cxfield,idim) ) THEN
	                  ssp1(idim) = 0
	               ELSE
	                  ssp1(idim) = +1
	               ENDIF
 820	            CONTINUE
	            DO 830 lt = ssfix(t_dim,ssm1(t_dim)),
     .				ssfix(t_dim,ssp1(t_dim))
	            DO 830 kz = ssfix(z_dim,ssm1(z_dim)),
     .				ssfix(z_dim,ssp1(z_dim))
	            DO 830 jy = ssfix(y_dim,ssm1(y_dim)),
     .				ssfix(y_dim,ssp1(y_dim))
	            DO 830 ix = ssfix(x_dim,ssm1(x_dim)),
     .				ssfix(x_dim,ssp1(x_dim))
	               IF ( field(ix,jy,kz,lt) .NE. missing ) THEN
	                  nfix = nfix + 1
	                  sumfix = sumfix +  field(ix,jy,kz,lt)
	               ENDIF
 830	            CONTINUE
	            IF ( nfix .GT. 0 ) THEN
	               surrnd(i,j,k,l) = sumfix/nfix
	               res_flag = 1.
	            ELSE
	               res_flag = -9.
	               result = fill_value
	               GOTO 1000  
	            ENDIF
	         ELSE
* ... dont even try to fill the hole
	            res_flag = -9.
	            result = fill_value
	            GOTO 1000
	         ENDIF
	      ENDIF
 850	   CONTINUE

* compute the interpolated value from the fully-specified surrounding points
* ... interpolate on T axis 
	   IF (ssorww(t_dim) .NE. by_nothin) THEN
	      DO 910 i = 1,ihi
	      DO 910 j = 1,jhi
	      DO 910 k = 1,khi
 910	      surrnd(i,j,k,1) =   lin_fact(t_dim)  * surrnd(i,j,k,1)
     .			    + (1.-lin_fact(t_dim)) * surrnd(i,j,k,2)
	   ENDIF
* ... interpolate on Z axis 
	   IF (ssorww(z_dim) .NE. by_nothin) THEN
	      DO 920 i = 1,ihi
	      DO 920 j = 1,jhi
 920	      surrnd(i,j,1,1) =   lin_fact(z_dim)  * surrnd(i,j,1,1)
     .			    + (1.-lin_fact(z_dim)) * surrnd(i,j,2,1)
	   ENDIF
* ... interpolate on Y axis 
	   IF (ssorww(y_dim) .NE. by_nothin) THEN
	      DO 930 i = 1,ihi
 930	      surrnd(i,1,1,1) =   lin_fact(y_dim)  * surrnd(i,1,1,1)
     .			    + (1.-lin_fact(y_dim)) * surrnd(i,2,1,1)
	   ENDIF
* ... interpolate on X axis 
	   IF (ssorww(x_dim) .NE. by_nothin) THEN
	      surrnd(1,1,1,1) =   lin_fact(x_dim)  * surrnd(1,1,1,1)
     .			    + (1.-lin_fact(x_dim)) * surrnd(2,1,1,1)
	   ENDIF
	   result = surrnd(1,1,1,1)

* output the result
 1000	   IF ( binary ) THEN
	      IF ( condensed ) THEN
	         nbuf = nbuf + 1
	         out_buf(nbuf) = result
	         IF ( nbuf .EQ. columns ) THEN
	            WRITE ( out_lun ) out_buf
	            nbuf = 0
	         ENDIF
	      ELSE
	         WRITE ( out_lun ) result,
     .				(coords(ii(i),i),i=1,4), res_flag
	      ENDIF
	   ELSE
	       WRITE ( out_lun, fmt ) result,
     .				(coords(ii(i),i),i=1,4), res_flag
	   ENDIF
	
! diag	   WRITE (6,*) 'Result: ',result
! diag	   WRITE (6,*) 'lo/hi subscripts: ',((ss(idim,i),i=1,2),idim=1,4)
! diag	   WRITE (6,*) 'factors: ',lin_fact

* prepare for the next sample point
	   DO 1100 idim = 1, 4
 1100	   ii(idim) = ii(idim) + di(idim)

 2000	CONTINUE

* check for any output remaining in the output record buffer
	IF ( condensed .AND. nbuf.GT.0 ) 
     .		CALL WARN( 'Output columns didnt match sampling coordinates')

* success!
	RETURN

* error exits
 5000	RETURN

	END
