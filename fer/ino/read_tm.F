	SUBROUTINE READ_TM (  cx, mr, status )

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
* read a grid of data from a TMAP data set

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 11/28/88 - based on IS_READ_GT v0.3 and READ_GT_XYZ v1.1
* V200:  5/19/89 - reordered lo/hi_ss args
*	 12/8/89 - changed call arguments in TM_READ for new compiler
* V230:  7/30/92 - must CALL ERRMSG to stop command files, etc. following error
*        11/6/92 - increased variable length to 24 characters
* V312: 5/94 - array "memory" as a calling argument
* V500: 2/99 - *kob* add call to new routine REPLACEABLE_BAD_FLAGS to allow
*              and check for NaN as missing values
*	3/99 - *kob* up VAR_CODE and local var variable_code to 64 chars
*	5/99 - *sh* added strides to TMAP read calls
* V533: 6/01 - *sh* initialize string input variable
* v552: 4/03 - *acm* up VAR_CODE and local var variable_code to 128 chars
* V556  4/04 - *acm* stride/modulo fixes: use NINT to convert REAL*8 to integer
* v602  8/06 - *jli* native strides  replace cx_delta logic by the line_delta
*       *acm* 3/12 cleanup old VMS ifdef
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V702 1/17 *sh* as a part of conversion to dynamic memory fixed bug that 
*                REPLACE_BAD_DATA was being called on string variables


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

* calling argument declarations:
	INTEGER	cx, mr, status

* internal variable declarations:
	LOGICAL	TM_ITSA_DSG_RAGGED,
     .		has_a_stride, has_strides(nferdims), 
     .		its_dsg, its_dsg_promote
	INTEGER MGRID_SIZE_DELTA, TM_DSG_NF2FEATURES,
     .			idim, variable, category, data_set, grid,
     .			sf_num, tmap_status, 
     .			lo_ss(nferdims), hi_ss(nferdims),
     .			lo_s1, lo_s2, lo_s3, lo_s4, lo_s5, lo_s6, 
     .			hi_s1, hi_s2, hi_s3, hi_s4, hi_s5, hi_s6,
     .			stride(nferdims), lo_ss_ds,hi_ss_ds, line, n2feat,
     .			nvdims, vdims(2), npts
* V500 *kob* 3/99- up VAR_CODE and local var variable_code to 64 chars
	CHARACTER*128	VAR_CODE, variable_code, buff

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
	data_set = cx_data_set( cx )
	category = cx_category( cx )
	variable = cx_variable( cx )
	grid = cx_grid( cx )
	variable_code = VAR_CODE( category, variable )
	its_dsg = TM_ITSA_DSG_RAGGED(grid)

* check data type
	IF ( cx_type(cx) .NE. ds_var_type(variable) ) THEN
	   CALL WARN('Internal err: read data type mismatch')
	   cx_type(cx) = ds_var_type(variable)
	ENDIF

* jli: replace the following cx_delta logic by the line_delta

* strides indicated in cx_delta e.g. SST[I=110:180:10] ?
* If so, transfer this info to the stride array
	has_a_stride = .FALSE.

* reserve and catalog space for the result
	CALL CREATE_MEM_VAR( cx, mr, status )
	IF ( status .NE. ferr_ok ) GOTO 5900
	IF (ds_var_type(variable) .EQ. ptype_string)
     .		CALL INIT_C_STRING_ARRAY( MGRID_SIZE_DELTA(mr),
     .			                  memry(mr)%ptr,
     .					  mr_c_pointer(mr)      )

* If this is a DSG ragged array variable, that is a separate path
	IF (its_dsg) GOTO 500

* un-cache it?
* Turns out that uncaching the result here causes all of its descendants to
* become uncached, as well, and lowers the cache hit rate. Instead, we will
* handle invalid mr limits left by strides in IS_REGRID
!	IF ( has_a_stride ) CALL UN_CACHE( mr )   ! since lo:hi are "fake"

* set up array of subscripts

	DO 300 idim = 1, nferdims
*<jli
           line = grid_line(idim, grid)

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
     
              IF ( grid_line( idim, grid ) .EQ. mnormal ) THEN
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

	      IF ( grid_line( idim, grid ) .EQ. mnormal ) THEN
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


                 lo_ss(idim) = (cx_lo_ss( cx, idim )-1)*stride(idim) + 
     .                         line_offset(line)

                 hi_ss(idim) = (cx_hi_ss( cx, idim )-1)*stride(idim) +
     .                         line_offset(line) 

!                for a reversed axis
                 IF( line_reversed(line)) THEN

! get size of the native axis 
! w/ fix to issue 1843, var_ss_lims returns strided index values
		 
                    CALL VAR_SS_LIMS(idim, cx,lo_ss_ds,hi_ss_ds)
                    hi_ss_ds = hi_ss_ds* stride(idim)

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
 500	IF (mode_diagnostic) THEN
	   IF ( has_a_stride) THEN
	     CALL DIAGNOSTIC_OUT('rdstride',cx,point_to_cx)
	   ELSE
	     CALL DIAGNOSTIC_OUT('reading',mr,point_to_mr)
	   ENDIF
	ENDIF

* read the data

	IF (its_dsg) THEN
	
	   CALL CD_DSG_READ(data_set,
     .			    variable,
     .			    memry(mr)%ptr,
     .			    mr_type(mr),
     .			    tmap_status	)
	ELSE
	   
	   n2feat = TM_DSG_NF2FEATURES (data_set)
	   IF (n2feat .GT. 0) THEN
	   
	      CALL CD_GET_VAR_DIMS_INFO (data_set, variable, 
     .        buff, nvdims, vdims, status)
	      IF (nvdims.EQ.2) vdims(1) = vdims(2) 
	      CALL CD_GET_DS_DIMS (data_set, vdims(1), buff, npts, status)
	      IF (n2feat .NE. npts) n2feat = 0
	   ENDIF

	   CALL TM_READ( data_set,
     .			 variable_code,
     .			 lo_s1, lo_s2, lo_s3, lo_s4, lo_s5, lo_s6, 
     .			 hi_s1, hi_s2, hi_s3, hi_s4, hi_s5, hi_s6,
     .			 lo_ss,
     .			 hi_ss,
     .			 stride,
     .			 memry( mr )%ptr,
     .			 sf_num,
     .			 n2feat,
     .			 tmap_status		)
	ENDIF 
	IF ( tmap_status .NE. merr_ok ) THEN
	   CALL DELETE_VARIABLE( mr )
	   GOTO 5010
	ENDIF

* replace multiple bad data flags in some variables
* IF ( ds_bad_flag( variable ) .NE. ds_missing_flag( variable ) )
* *kob* 2/99 - call new routine to check for usage of NaN
! 1/17 *sh* check mr_type is needed here because missing data flags are not
! properly initialized on string variables
	IF ( mr_type(mr) .NE. ptype_string
     . .AND. REPLACEABLE_BAD_FLAGS(ds_bad_flag(variable), 
     .                             ds_missing_flag(variable)) .EQ. 1)
     .	   CALL REPLACE_BAD_DATA
     .		(memry(mr)%ptr,mr,cx,ds_missing_flag(variable))

* successful completion
	RETURN

* error exit
 5010	CALL ERRMSG( ferr_TMAP_error, status, ' ', *5900 )   ! 7/92
 5900	RETURN
	END	
