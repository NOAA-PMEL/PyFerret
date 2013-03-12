	SUBROUTINE READ_REMOTE ( memory, baddat, cx, mr, status )

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
* read a grid of data from an F-TDS dataset, having been defined with LET/REMOTE

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Thermal Modeling and Analysis Program
*
* revision 0.0 - 2/2012 - based on READ_TM

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
	REAL	memory( mem_blk_size, max_mem_blks ), baddat

* internal variable declarations:
	LOGICAL	has_a_stride, has_strides(nferdims)
	INTEGER MGRID_SIZE,
     .			idim, uvar, category, dset, grid,
     .			sf_num, tmap_status, idir, ndim,
     .			lo_ss(nferdims), hi_ss(nferdims),
     .			lo_s1, lo_s2, lo_s3, lo_s4, lo_s5, lo_s6, 
     .			hi_s1, hi_s2, hi_s3, hi_s4, hi_s5, hi_s6,
     .			stride(nferdims), lo_ss_ds,hi_ss_ds, line, 
     .			gridlines(nferdims), grid_start(nferdims),
     .			grid_end(nferdims), grid_order(nferdims)
* V500 *kob* 3/99- up VAR_CODE and local var vname to 64 chars
	CHARACTER*128	VAR_CODE, vname

	LOGICAL		reprimand
	DATA       	reprimand/.TRUE./
* *kob* 2/99 - add declaration for new c function

	INTEGER REPLACEABLE_BAD_FLAGS

	EQUIVALENCE	(lo_ss(1),lo_s1), (lo_ss(2),lo_s2),
     .			(lo_ss(3),lo_s3), (lo_ss(4),lo_s4),
     .			(lo_ss(5),lo_s5), (lo_ss(6),lo_s6),
     .			(hi_ss(1),hi_s1), (hi_ss(2),hi_s2),
     .			(hi_ss(3),hi_s3), (hi_ss(4),hi_s4),
     .			(hi_ss(5),hi_s5), (hi_ss(6),hi_s6)

* --- end of introductory code ---

* initialize
	dset = cx_data_set( cx )
	category = cat_user_var
	uvar = cx_variable( cx )
	vname = VAR_CODE( category, uvar )

	grid = cx_grid( cx )

	DO 50 idim = 1, nferdims
	   gridlines(idim) = grid_line(idim, grid)
  50	CONTINUE
	CALL CD_NF_GET_VAR_GRID ( dset, vname, gridlines, status )

	ndim = 0
	DO 70 idim = 1, nferdims
	   line = gridlines(idim)
	   grid_start(idim) = 1
	   grid_end(idim) = 1
	   grid_order(idim) = 0
	   IF (line .NE. mnormal) THEN
	      grid_end(idim) = line_dim(line)

	      DO 60 idir = 1, nferdims+1
                 IF ( line_direction(line) .EQ. axis_orients( idir ) ) THEN
		    ndim = ndim + 1
		    grid_order(ndim) = idir
		    IF (grid_order(ndim) .EQ. 7) grid_order(ndim) = 3
		    GOTO 65
		 ENDIF
 60	      CONTINUE
 65	      CONTINUE
	   ENDIF
 70     CONTINUE

* ensure that all needed limits are known and correct for the remote var

	DO 100 idim = 1, nferdims

	   IF ( gridlines(idim) .EQ. mnormal) THEN
	      cx_lo_ss( cx, idim ) = unspecified_int4
	      cx_hi_ss( cx, idim ) = unspecified_int4
	   ENDIF

	   IF ( cx_lo_ss( cx, idim ) .NE. unspecified_int4 ) GOTO 100
	   IF ( gridlines(idim).EQ.mnormal) GOTO 100

	   IF ( reprimand ) THEN
	      CALL WARN
     .		('FERRET is more efficient if regions are fully specified' )
	      CALL WARN( ww_dim_name(idim)//
     .		' axis limits are unspecified - possibly others, too' )
	      reprimand = .FALSE.		! only remind them once
	   ENDIF
	   CALL FLESH_OUT_AXIS( idim, cx, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000		! should never happen
!	   CALL ERRMSG( ferr_limits, status,
!     .			ww_dim_name(idim)//' axis limits are unknown for '//
!     .			VAR_CODE( cat_file_var, cx_variable(cx) ), *5000 )
 100	CONTINUE


* jli: replace the following cx_delta logic by the line_delta

* strides indicated in cx_delta e.g. SST[I=110:180:10] ?
* If so, transfer this info to the stride array
	has_a_stride = .FALSE.

* reserve and catalog space for the result
	CALL CREATE_MEM_VAR( cx, mr, status )
	IF ( status .NE. ferr_ok ) GOTO 5900
	IF (ds_var_type(uvar) .EQ. ptype_string)
     .		CALL INIT_C_STRING_ARRAY( MGRID_SIZE(mr),
     .			                  memory(1,mr_blk1(mr)),
     .					  mr_c_pointer(mr)      )

* un-cache it?
* Turns out that uncaching the result here causes all of its descendants to
* become uncached, as well, and lowers the cache hit rate. Instead, we will
* handle invalid mr limits left by strides in IS_REGRID
!	IF ( has_a_stride ) CALL UN_CACHE( mr )   ! since lo:hi are "fake"

* set up array of subscripts
	DO 300 idim = 1, nferdims
*<jli
           line = gridlines(idim)

*          For strides indicated in cx_delta e.g. SST[I=110:180:10]
*          use cx_delta logic

           IF(line_parent(line) .EQ. 0) THEN
              has_strides(idim) =  cx_delta(idim,cx) .NE. unspecified_val8
     .                  .AND.  cx_delta(idim,cx) .NE. 1 
              IF (has_strides(idim)) THEN
                stride(idim) = NINT(cx_delta(idim,cx))
                has_a_stride = .TRUE.
              ELSE
                stride(idim) = 1
              ENDIF
     
              IF ( gridlines(idim) .EQ. mnormal ) THEN
                 lo_ss( idim ) = 1
                 hi_ss( idim ) = 1
              ELSE
                 lo_ss( idim ) = cx_lo_ss( cx, idim )
                 hi_ss( idim ) = cx_hi_ss( cx, idim )
              ENDIF
           
           ELSE

*             For strides specified in SET AXIS/STRIDE
*             Use line_delta logic

              has_strides(idim) = .TRUE.

	      IF ( gridlines(idim) .EQ. mnormal ) THEN
	         lo_ss( idim ) = 1
	         hi_ss( idim ) = 1
                 stride( idim ) = 1

              ELSEIF ( line_regular(line) ) THEN
              
                 IF (has_strides(idim)) THEN
                    stride(idim) = line_delta(line)/line_delta(line_parent(line))+0.5
                    has_a_stride = .TRUE.
                 ELSE
                    stride(idim) =1 
                 ENDIF

!                get size of the native axis
                 CALL VAR_SS_LIMS(idim, cx,lo_ss_ds,hi_ss_ds)

                 lo_ss(idim) = (cx_lo_ss( cx, idim )-1)*stride(idim) + 
     .                         line_offset(line)

                 hi_ss(idim) = (cx_hi_ss( cx, idim )-1)*stride(idim) +
     .                         line_offset(line) 

!                for a reversed axis
                 IF( line_reversed(line)) THEN
!                   must compute hi_ss first and then lo_ss
                    hi_ss(idim) = hi_ss_ds - 
     .                         (line_dim(line)-cx_hi_ss( cx, idim ))*stride(idim) -
     .                         line_offset(line) + 1
  
                    lo_ss(idim) = hi_ss(idim) - 
     .                         (cx_hi_ss(cx,idim)-cx_lo_ss(cx,idim))*stride(idim)
                 ENDIF

!                for nested stride
                 IF( (cx_delta(idim,cx) .NE. unspecified_val8)
     .                  .AND.  (cx_delta(idim,cx) .NE. 1) ) THEN
                    stride(idim) = stride(idim)*cx_delta(idim,cx)
                 ENDIF


              ELSE
*             Irregular line

                 IF (has_strides(idim)) THEN
                    has_a_stride = .TRUE.
                    stride(idim) = line_delta(line)

!                   Nested strided, e.g.,i=1:20:2      
                    IF( (cx_delta(idim,cx) .NE. unspecified_val8)
     .                  .AND.  (cx_delta(idim,cx) .NE. 1) ) THEN
                       stride(idim) = stride(idim)*cx_delta(idim,cx)
                    ENDIF

                    lo_ss(idim)  = line_start(line) +  
     .                             (cx_lo_ss( cx, idim )-1)*stride(idim)
                    hi_ss(idim)  = line_start(line) +  
     .                             (cx_hi_ss( cx, idim )-1)*stride(idim)
                 ELSE
                    lo_ss( idim ) = cx_lo_ss( cx, idim )
                    hi_ss( idim ) = cx_hi_ss( cx, idim )
                    stride(idim) = 1 
                 ENDIF 

	      ENDIF
           
           ENDIF
 300	CONTINUE

* diagnostic output
	IF (mode_diagnostic) THEN
	   IF ( has_a_stride) THEN
	     CALL DIAGNOSTIC_OUT('rdstride',cx,point_to_cx)
	   ELSE
	     CALL DIAGNOSTIC_OUT('reading',mr,point_to_mr)
	   ENDIF
	ENDIF

* read the data

	CALL REMOTE_READ  ( dset,
     .			vname,
     .			rvar_varid(uvar),
     .			lo_s1, lo_s2, lo_s3, lo_s4, lo_s5, lo_s6, 
     .			hi_s1, hi_s2, hi_s3, hi_s4, hi_s5, hi_s6,
     .			lo_ss,
     .			hi_ss,
     .			stride,
     .			gridlines,
     .			grid_start,
     .			grid_end,
     .			grid_order,
     .			memory( 1, mr_blk1( mr ) ),
     .			baddat, 
     .			sf_num,
     .			tmap_status		)
	IF ( tmap_status .NE. merr_ok ) THEN
	   CALL DELETE_VARIABLE( mr )
	   GOTO 5010
	ENDIF

* baddat attribute is read from the f-tds dataset; if the 
* _FillValue attribute and the bad-value with in the data
* are corrupted by e.g. a conversion from float to double,
* then this way both are corrupted in the same way.

	mr_bad_data(mr) = baddat

* replace multiple bad data flags in some variables
* IF ( ds_bad_flag( variable ) .NE. ds_missing_flag( variable ) )
* *kob* 2/99 - call new routine to check for usage of NaN
	IF (REPLACEABLE_BAD_FLAGS(ds_bad_flag(uvar), 
     .                            ds_missing_flag(uvar)) .EQ. 1)
     .	   CALL REPLACE_BAD_DATA
     .		(memory(1,mr_blk1(mr)),mr,cx,ds_missing_flag(uvar))

* successful completion
 	RETURN

* error exit
 5000	cx_category(cx) = cat
 	RETURN

* error exit
 5010	CALL ERRMSG( ferr_TMAP_error, status, ' ', *5900 )   ! 7/92
 5900	RETURN
	END	
