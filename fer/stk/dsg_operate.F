      SUBROUTINE DSG_OPERATE (	operation, ncom, ifv, res_cx, com_cx,
     .                          dset,
     .                          nfeatures, fmask,
     .				com1, mr1,
     .				com2, mr2,
     .				com3, mr3,
     .				com4, mr4,
     .				res, mres )

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
* perform an operation on a contiguous ragged array DSG
* possibly involving a mixture of instance and obs arguments 

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* V7.2 *sh* 5/17 - patterned on OPERATE.F
* V7.5 *acm* 7/19 -fixes to indexing for some operations
* V7.51 *acm* 8/19 - more fixes to indexing for some operations
* V7.5  *acm* 4/20 - more fixes to indexing for some operations, atan2,...
* 12/2022 *acm* Code cleanup: Argument to RANDN2 is of type integer
*  1/2023 *acm* Code cleanup: Replace the computed GOTO statement 
*                  "GOTO ( 100, 200, 300..) op"  with IF blocks

	include 'tmap_dims.parm'
	include	'ferret.parm'
        include 'xdset_info.cmn_text'
        include 'xdyn_linemem.cmn_text'
	include	'xmem_subsc.cmn'
	include 'xvariables.cmn'
	include 'xcontext.cmn'

* calling argument declarations:
	LOGICAL ifv, fmask(nfeatures)
	INTEGER	operation, ncom, res_cx, mr1, mr2, mr3, mr4, mres,
     .          com_cx(ncom), dset, nfeatures
	REAL    com1( m1lox:m1hix ),  ! need lo:hi cuz of -999:-999
     .		com2( m2lox:m2hix ),
     .		com3( m3lox:m3hix ),
     .		com4( m4lox:m4hix ),
     .		 res( * )

!DSGTBD: if it becomes desirable to implement a free (gridded) F axis ...
!	REAL    com1( m1hix, m1lof:m1hif ),
!     .		com2( m2hix, m2lof:m2hif ),
!     .		com3( m3hix, m3lof:m3hif ),
!     .		com4( m4hix, m4lof:m4hif ),
!     .		 res( m5hix, m5lof:m5hif )

* internal variable declarations:
	LOGICAL	first, is_obs_res, is_inst(4), is_obs(4), is_const(4)
	INTEGER CX_DIM_LEN, TM_DSG_DSET_FROM_GRID, TM_DSG_NFEATURES,
     .          row_size_lm, op, inseed, i, ifeature, iobs, base, flen,
     .          nobs
	REAL	RANDU, RANDN, RANDN2, RHO_UNESCO, THETA_FOFF,
     .		DAYS_FROM_DAY0,
     .          bad1, bad2, bad3, bad4, bad_res, dummy,
     .          op_repl, args(4)
	REAL*4  R_IN

* convenience equivalences
        LOGICAL
     .          is_inst_1 , is_inst_2 , is_inst_3 , is_inst_4,
     .          is_obs_1  , is_obs_2  , is_obs_3  , is_obs_4,
     .          is_const_1, is_const_2, is_const_3, is_const_4
        REAL    arg1, arg2, arg3, arg4
        EQUIVALENCE
     .          (is_inst (1), is_inst_1 ), (is_inst (2), is_inst_2 ),
     .          (is_inst (3), is_inst_3 ), (is_inst (4), is_inst_4 ),
     .          (is_obs  (1), is_obs_1  ), (is_obs  (2), is_obs_2  ),
     .          (is_obs  (3), is_obs_3  ), (is_obs  (4), is_obs_4  ),
     .          (is_const(1), is_const_1), (is_const(2), is_const_2),
     .          (is_const(3), is_const_3), (is_const(4), is_const_4),
     .          (args(1), arg1), (args(2), arg2),
     .          (args(3), arg3), (args(4), arg4)

* internal parameter declarations:
	REAL*8          pdays_by_1900
	PARAMETER     ( pdays_by_1900 = 59958230400.0 / (60.*60.*24.) )

* get feature and obs masks
        CALL MAKE_DSG_FEATURE_MASK(dset, res_cx, fmask, nfeatures)

* initialize
	row_size_lm = dsg_loaded_lm(dsg_row_size_var(dset))
        nobs  = dsg_obs_dimlen(dset)

* differentiate obs, instance and constant component vars
        DO i = 1, ncom
           is_inst(i)  = hi_ss(i,1) .EQ. nfeatures
           is_obs(i)   = hi_ss(i,1) .EQ. nobs
           is_const(i) = cx_category(com_cx(i)) .EQ. cat_constant
	   is_const(i) = (hi_ss(i,1) .EQ. unspecified_int4) 
        ENDDO

* set constant values, just once, now
        IF (                is_const_1) THEN
	   IF (hi_ss(1,1) .EQ. unspecified_int4) THEN 
	      arg1 = com1(unspecified_int4)
	   ELSE
	      arg1 = com1(m1lox) 
	   ENDIF
	ENDIF
        IF (ncom.GE.2 .AND. is_const_2)  THEN
	   IF (hi_ss(2,1) .EQ. unspecified_int4) THEN 
	      arg2 = com2(unspecified_int4)
	   ELSE
	      arg2 = com2(m1lox) 
	   ENDIF
	ENDIF
        IF (ncom.GE.3 .AND. is_const_3)   THEN
	   IF (hi_ss(3,1) .EQ. unspecified_int4) THEN 
	      arg3 = com3(unspecified_int4)
	   ELSE
	      arg3 = com3(m1lox) 
	   ENDIF 
	ENDIF
        IF (ncom.GE.4 .AND. is_const_4)   THEN
	   IF (hi_ss(4,1) .EQ. unspecified_int4) THEN 
	      arg4 = com4(unspecified_int4)
	   ELSE
	      arg4 = com4(m1lox) 
	   ENDIF
	ENDIF

* is the result a ragged array or a simple line of instance data?
* note that the result is only an instance when all components are too
        is_obs_res = m5hix .GT. nfeatures

* flag(s) for bad or missing values
	bad1 = mr_bad_data ( mr1 )
	bad2 = mr_bad_data ( mr2 )
	bad3 = mr_bad_data ( mr3 )
	bad4 = mr_bad_data ( mr4 )
	bad_res = mr_bad_data( mres )

* For operations, e.g.  IF var GT a THEN b, create a mask of 1's and 0's.
* but for expressions using IFV, the mask needs to be 1's and bad-values.
* If its an IFV expression then use the corresponding THEN and ELSE.

        op = operation
	op_repl = 0.
	IF (ifv) THEN
	   op_repl = bad_res
	   IF (op .EQ. 45) op = 47
	   IF (op .EQ. 46) op = 48
        ENDIF

* operators, functions, logic structures according to the value of op

* ----------------------------------------------------------------------------
* ------- OPERATORS -------------
*
* "+"
      IF (op .EQ. 1) THEN

	IF (is_const_1 .AND. is_const_2) THEN  ! 2 constants may have been sent in
	   IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2) THEN
	      res(1) = bad_res
	   ELSE
	      res(1) = arg1 + arg2
	   ENDIF
	   RETURN
	ENDIF
	
	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance arguments
           IF (is_inst_1) arg1 = com1(ifeature)
           IF (is_inst_2) arg2 = com2(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)
              IF (is_obs_2) arg2 = com2(iobs)

	      IF ( arg1.EQ.bad1 .OR. arg2.EQ.bad2 ) THEN
	        res(iobs) = bad_res
	      ELSE
	        res(iobs) = arg1 + arg2       !   "+"
	      ENDIF
           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF

* "-"
      IF (op .EQ. 2) THEN

	IF (is_const_1 .AND. is_const_2) THEN  ! 2 constants may have been sent in
	   IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2) THEN
	      res(1) = bad_res
	   ELSE
	      res(1) = arg1 - arg2
	   ENDIF
	   RETURN
	ENDIF
	
	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance arguments
           IF (is_inst_1) arg1 = com1(ifeature)
           IF (is_inst_2) arg2 = com2(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)
              IF (is_obs_2) arg2 = com2(iobs)

	      IF ( arg1.EQ.bad1 .OR. arg2.EQ.bad2 ) THEN
	        res(iobs) = bad_res
	      ELSE
	        res(iobs) = arg1 - arg2       !   "-"
	      ENDIF
           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF

* "*"
      IF (op .EQ. 3) THEN
	IF (is_const_1 .AND. is_const_2) THEN  ! 2 constants may have been sent in
	   IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2) THEN
	      res(1) = bad_res
	   ELSE
	      res(1) = arg1* arg2
	   ENDIF
	   RETURN
	ENDIF
	
	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance arguments
           IF (is_inst_1) arg1 = com1(ifeature)
           IF (is_inst_2) arg2 = com2(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)
              IF (is_obs_2) arg2 = com2(iobs)

	      IF ( arg1.EQ.bad1 .OR. arg2.EQ.bad2 ) THEN
	        res(iobs) = bad_res
	      ELSE
	        res(iobs) = arg1 * arg2       !   "*"
	      ENDIF
           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF

* "/"
      IF (op .EQ. 4) THEN

	IF (is_const_1 .AND. is_const_2) THEN  ! 2 constants may have been sent in
	   IF (arg2.EQ.0 .OR. arg1.EQ.bad1 .OR. arg2.EQ.bad2) THEN
	      res(1) = bad_res
	   ELSE
	      res(1) = arg1/ arg2
	   ENDIF
	   RETURN
	ENDIF
	
	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance arguments
           IF (is_inst_1) arg1 = com1(ifeature)
           IF (is_inst_2) arg2 = com2(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)
              IF (is_obs_2) arg2 = com2(iobs)

	      IF (arg2.EQ.0 .OR. arg1.EQ.bad1 .OR. arg2.EQ.bad2 ) THEN
	        res(iobs) = bad_res
	      ELSE
	        res(iobs) = arg1 / arg2       !   "/"
	      ENDIF
           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF
 
* "^"

      IF (op .EQ. 5) THEN

	IF (is_const_1 .AND. is_const_2) THEN  ! 2 constants may have been sent in
	   IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2) THEN
	      res(1) = bad_res
	   ELSE
	      res(1) = arg1**arg2
	   ENDIF
	   RETURN
	ENDIF
	
	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance arguments
           IF (is_inst_1) arg1 = com1(ifeature)
           IF (is_inst_2) arg2 = com2(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)
              IF (is_obs_2) arg2 = com2(iobs)

	      IF (arg2.EQ.0 .OR. arg1.EQ.bad1 .OR. arg2.EQ.bad2 ) THEN
	        res(iobs) = bad_res
	      ELSE
	        res(iobs) = arg1 ** arg2       !   "^"
	      ENDIF
           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF

* "AND" - logical AND of 2 masks (any non-zero data regarded as .TRUE.)

      IF (op .EQ. 6) THEN

	IF (is_const_1 .AND. is_const_2) THEN  ! 2 constants may have been sent in
	   IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2) THEN
	      res(1) = bad_res
	   ELSE
	      IF ( arg1.NE.0.0 .AND. arg2.NE.0.0 ) THEN
	  	 res(1) = 1.0
	      ELSE
	  	 res(1) = op_repl
	      ENDIF
	   ENDIF
	   RETURN
	ENDIF
	
	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance arguments
           IF (is_inst_1) arg1 = com1(ifeature)
           IF (is_inst_2) arg2 = com2(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)
              IF (is_obs_2) arg2 = com2(iobs)

              IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2) THEN
	         res(iobs) = bad_res
              ELSE
	         IF ( arg1.NE.0.0 .AND. arg2.NE.0.0 ) THEN
	  	    res(iobs) = 1.0
	         ELSE
	  	    res(iobs) = op_repl
	         ENDIF
              ENDIF

           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF

* "OR" - logical OR of 2 masks (any non-zero data regarded as .TRUE.)

      IF (op .EQ. 7) THEN

	IF (is_const_1 .AND. is_const_2) THEN  ! 2 constants may have been sent in
	   IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2) THEN
	      res(1) = bad_res
	   ELSE
	      IF ( arg1.NE.0.0 .OR. arg2.NE.0.0 ) THEN
	  	 res(1) = 1.0
	      ELSE
	  	 res(1) = op_repl
	      ENDIF
	   ENDIF
	   RETURN
	ENDIF
	
	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance arguments
           IF (is_inst_1) arg1 = com1(ifeature)
           IF (is_inst_2) arg2 = com2(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)
              IF (is_obs_2) arg2 = com2(iobs)

              IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2) THEN
	         res(iobs) = bad_res
              ELSE
	         IF ( arg1.NE.0.0 .OR. arg2.NE.0.0 ) THEN
	  	    res(iobs) = 1.0
	         ELSE
	  	    res(iobs) = op_repl
	         ENDIF
              ENDIF

           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF

* "GT" - mask of 1's where grid 1 exceeds grid 2, 0's elsewhere
* "GT" - mask of 1's where grid 1 exceeds grid 2, bad-val elsewhere

      IF (op .EQ. 8) THEN

	IF (is_const_1 .AND. is_const_2) THEN  ! 2 constants may have been sent in
	   IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2) THEN
	      res(1) = bad_res
	   ELSE
	      IF ( arg1 .GT. arg2 ) THEN
	  	 res(1) = 1.0
	      ELSE
	  	 res(1) = op_repl
	      ENDIF
	   ENDIF
	   RETURN
	ENDIF
	
	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance arguments
           IF (is_inst_1) arg1 = com1(ifeature)
           IF (is_inst_2) arg2 = com2(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)
              IF (is_obs_2) arg2 = com2(iobs)

              IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2) THEN
	         res(iobs) = bad_res
              ELSE
	         IF ( arg1 .GT. arg2 ) THEN
	  	    res(iobs) = 1.0
	         ELSE
	  	    res(iobs) = op_repl
	         ENDIF
              ENDIF

           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF

* "GE" - mask of 1's where grid 1 exceeds or equals grid 2, bad elsewhere

      IF (op .EQ. 9) THEN

	IF (is_const_1 .AND. is_const_2) THEN  ! 2 constants may have been sent in
	   IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2) THEN
	      res(1) = bad_res
	   ELSE
	      IF ( arg1 .GE. arg2 ) THEN
	  	 res(1) = 1.0
	      ELSE
	  	 res(1) = op_repl
	      ENDIF
	   ENDIF
	   RETURN
	ENDIF
	
	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance arguments
           IF (is_inst_1) arg1 = com1(ifeature)
           IF (is_inst_2) arg2 = com2(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)
              IF (is_obs_2) arg2 = com2(iobs)

              IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2) THEN
	         res(iobs) = bad_res
              ELSE
	         IF ( arg1 .GE. arg2 ) THEN
	  	    res(iobs) = 1.0
	         ELSE
	  	    res(iobs) = op_repl
	         ENDIF
              ENDIF

           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF

* "LT" - mask of 1's where grid 1 is less than grid 2, bad elsewhere

      IF (op .EQ. 10) THEN

	 IF (is_const_1 .AND. is_const_2) THEN  ! 2 constants may have been sent in
	   IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2) THEN
	      res(1) = bad_res
	   ELSE
	      IF ( arg1 .LT. arg2 ) THEN
	  	 res(1) = 1.0
	      ELSE
	  	 res(1) = op_repl
	      ENDIF
	   ENDIF
	   RETURN
	ENDIF

	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance arguments
           IF (is_inst_1) arg1 = com1(ifeature)
           IF (is_inst_2) arg2 = com2(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)
              IF (is_obs_2) arg2 = com2(iobs)

	      IF ( arg1.EQ.bad1 .OR. arg2.EQ.bad2 ) THEN
	        res(iobs) = bad_res
	      ELSE
	         IF ( arg1 .LT. arg2 ) THEN
		    res(iobs) = 1.0
	         ELSE
		    res(iobs) = op_repl
	         ENDIF
	      ENDIF
           ENDDO
           base = base + flen
        ENDDO
	RETURN
	
      ENDIF

* "LE" - mask of 1's where grid 1 is less than or equal to grid 2, bad elsewher

      IF (op .EQ. 11) THEN

	IF (is_const_1 .AND. is_const_2) THEN  ! 2 constants may have been sent in
	   IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2) THEN
	      res(1) = bad_res
	   ELSE
	      IF ( arg1 .LE. arg2 ) THEN
	  	 res(1) = 1.0
	      ELSE
	  	 res(1) = op_repl
	      ENDIF
	   ENDIF
	   RETURN
	ENDIF
	
	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance arguments
           IF (is_inst_1) arg1 = com1(ifeature)
           IF (is_inst_2) arg2 = com2(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)
              IF (is_obs_2) arg2 = com2(iobs)

              IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2) THEN
	         res(iobs) = bad_res
              ELSE
	         IF ( arg1 .LE. arg2 ) THEN
	  	    res(iobs) = 1.0
	         ELSE
	  	    res(iobs) = op_repl
	         ENDIF
              ENDIF

           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF

* "EQ" - mask of 1's where grids are equal, 0's elsewhere
* "EQ" - mask of 1's where grids are equal, bad elsewhere

      IF (op .EQ. 12) THEN

	IF (is_const_1 .AND. is_const_2) THEN  ! 2 constants may have been sent in
	   IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2) THEN
	      res(1) = bad_res
	   ELSE
	      IF ( arg1 .EQ. arg2 ) THEN
	  	 res(1) = 1.0
	      ELSE
	  	 res(1) = op_repl
	      ENDIF
	   ENDIF
	   RETURN
	ENDIF
	
	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance arguments
           IF (is_inst_1) arg1 = com1(ifeature)
           IF (is_inst_2) arg2 = com2(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)
              IF (is_obs_2) arg2 = com2(iobs)

              IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2) THEN
	         res(iobs) = bad_res
              ELSE
	         IF ( arg1 .EQ. arg2 ) THEN
	  	    res(iobs) = 1.0
	         ELSE
	  	    res(iobs) = op_repl
	         ENDIF
              ENDIF

           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF


* "NE" - mask of 1's where grids are not equal, 0's elsewhere

      IF (op .EQ. 13) THEN

	IF (is_const_1 .AND. is_const_2) THEN  ! 2 constants may have been sent in
	   IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2) THEN
	      res(1) = bad_res
	   ELSE
	      IF ( arg1 .NE. arg2 ) THEN
	  	 res(1) = 1.0
	      ELSE
	  	 res(1) = op_repl
	      ENDIF
	   ENDIF
	   RETURN
	ENDIF
	
	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance arguments
           IF (is_inst_1) arg1 = com1(ifeature)
           IF (is_inst_2) arg2 = com2(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)
              IF (is_obs_2) arg2 = com2(iobs)

              IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2) THEN
	         res(iobs) = bad_res
              ELSE
	         IF ( arg1 .NE. arg2 ) THEN
	  	    res(iobs) = 1.0
	         ELSE
	  	    res(iobs) = op_repl
	         ENDIF
              ENDIF

           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF

      
      IF (op.EQ.14 .OR.op.EQ.15 .OR.op.EQ.16 .OR.op.EQ.17 .OR.op.EQ.18) THEN
	STOP 'OPERATOR NOT IMPLEMENTED'
      ENDIF

* -----------------------------------------------------------------------------
* -------- FUNCTIONS ------------


* "EXP" - exponential function (single argument)

      IF (op .EQ. 19) THEN

	IF (is_const_1) THEN  
	   IF (arg1.EQ.bad1 ) THEN
	      res(1) = bad_res
	   ELSE
	      res(1) = EXP(arg1)
	   ENDIF
	   RETURN
	ENDIF

	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance argument
           IF (is_inst_1) arg1 = com1(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)

	      IF ( arg1 .EQ. bad1 ) THEN
	        res(iobs) = bad_res
	      ELSE
	        res(iobs) =  EXP(arg1)
	      ENDIF
           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF

* "LOG"  (single argument)

      IF (op .EQ. 20) THEN

	IF (is_const_1) THEN  
	   IF (arg1.EQ.bad1 .OR. arg1.LE.0.0 ) THEN
	      res(1) = bad_res
	   ELSE
	      res(1) = LOG10(arg1)
	   ENDIF
	   RETURN
	ENDIF

	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance argument
           IF (is_inst_1) arg1 = com1(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)

	      IF ( arg1.EQ.bad1 .OR. arg1.LE.0.0 ) THEN
	        res(iobs) = bad_res
	      ELSE
	        res(iobs) =  LOG10(arg1)
	      ENDIF
           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF

* "MAX" 2 arguments

      IF (op .EQ. 21) THEN

	IF (is_const_1 .AND. is_const_2) THEN  ! 2 constants may have been sent in
	   IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2) THEN
	      res(1) = bad_res
	   ELSE
	      res(1) = MAX( arg1, arg2 )
	   ENDIF
	   RETURN
	ENDIF
	
	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance arguments
           IF (is_inst_1) arg1 = com1(ifeature)
           IF (is_inst_2) arg2 = com2(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)
              IF (is_obs_2) arg2 = com2(iobs)

	      IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2 ) THEN
	        res(iobs) = bad_res
	      ELSE
	        res(iobs) = MAX(arg1, arg2)
	      ENDIF
           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF

* "MIN" 2 arguments

      IF (op .EQ. 22) THEN

	IF (is_const_1 .AND. is_const_2) THEN  ! 2 constants may have been sent in
	   IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2) THEN
	      res(1) = bad_res
	   ELSE
	      res(1) = MIN( arg1, arg2 )
	   ENDIF
	   RETURN
	ENDIF
	
	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance arguments
           IF (is_inst_1) arg1 = com1(ifeature)
           IF (is_inst_2) arg2 = com2(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)
              IF (is_obs_2) arg2 = com2(iobs)

	      IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2 ) THEN
	        res(iobs) = bad_res
	      ELSE
	        res(iobs) = MIN(arg1, arg2)
	      ENDIF
           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF


* "INT"  (single argument)

      IF (op .EQ. 23) THEN

	IF (is_const_1) THEN  
	   IF (arg1 .EQ. bad1 ) THEN
	      res(1) = bad_res
	   ELSE
	      res(1) = INT(arg1)
	   ENDIF
	   RETURN
	ENDIF

	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance argument
           IF (is_inst_1) arg1 = com1(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)

	      IF ( arg1.EQ.bad1 .OR. arg2.EQ.bad2 ) THEN
	        res(iobs) = bad_res
	      ELSE
	        res(iobs) =  INT(arg1)
	      ENDIF
           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF


* "ABS"  (single argument)

      IF (op .EQ. 24) THEN

	IF (is_const_1) THEN  
	   IF (arg1 .EQ. bad1 ) THEN
	      res(1) = bad_res
	   ELSE
	      res(1) = ABS(arg1)
	   ENDIF
	   RETURN
	ENDIF

	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance argument
           IF (is_inst_1) arg1 = com1(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)

	      IF ( arg1.EQ.bad1 .OR. arg2.EQ.bad2 ) THEN
	        res(iobs) = bad_res
	      ELSE
	        res(iobs) =  ABS(arg1)
	      ENDIF
           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF


* "SIN"  (single argument)

      IF (op .EQ. 25) THEN

	IF (is_const_1) THEN  
	   IF (arg1 .EQ. bad1 ) THEN
	      res(1) = bad_res
	   ELSE
	      res(1) = SIN(arg1)
	   ENDIF
	   RETURN
	ENDIF

	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance argument
           IF (is_inst_1) arg1 = com1(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)

	      IF ( arg1.EQ.bad1 .OR. arg2.EQ.bad2 ) THEN
	        res(iobs) = bad_res
	      ELSE
	        res(iobs) =  SIN(arg1)
	      ENDIF
           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF

* "COS"  (single argument)

      IF (op .EQ. 26) THEN

	IF (is_const_1) THEN  
	   IF (arg1 .EQ. bad1 ) THEN
	      res(1) = bad_res
	   ELSE
	      res(1) = COS(arg1)
	   ENDIF
	   RETURN
	ENDIF

	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance argument
           IF (is_inst_1) arg1 = com1(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)

	      IF ( arg1.EQ.bad1 .OR. arg2.EQ.bad2 ) THEN
	        res(iobs) = bad_res
	      ELSE
	        res(iobs) =  COS(arg1)
	      ENDIF
           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF

* "TAN"  (single argument)
	
      IF (op .EQ. 27) THEN
	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance argument
           IF (is_inst_1) arg1 = com1(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)

	      IF ( arg1.EQ.bad1 .OR. arg2.EQ.bad2 ) THEN
	        res(iobs) = bad_res
	      ELSE
	        res(iobs) =  TAN(arg1)
	      ENDIF
           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF


* "LN"  (single argument)

      IF (op .EQ. 28) THEN

	IF (is_const_1) THEN  
	   IF (arg1.EQ.bad1 .OR. arg1.LE.0.0 ) THEN
	      res(1) = bad_res
	   ELSE
	      res(1) = LOG(arg1)
	   ENDIF
	   RETURN
	ENDIF

	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance argument
           IF (is_inst_1) arg1 = com1(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)

	      IF ( arg1.EQ.bad1 .OR. arg2.EQ.bad2 ) THEN
	        res(iobs) = bad_res
	      ELSE
	        res(iobs) =  LOG(arg1)
	      ENDIF
           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF

* "MOD"  2 argument

      IF (op .EQ. 29) THEN

	IF (is_const_1 .AND. is_const_2) THEN  ! 2 constants may have been sent in
	   IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2) THEN
	      res(1) = bad_res
	   ELSE
	      res(1) = MOD( arg1, arg2 )
	   ENDIF
	   RETURN
	ENDIF
	
	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance arguments
           IF (is_inst_1) arg1 = com1(ifeature)
           IF (is_inst_2) arg2 = com2(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)
              IF (is_obs_2) arg2 = com2(iobs)

	      IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2 ) THEN
	        res(iobs) = bad_res
	      ELSE
	        res(iobs) = MOD(arg1, arg2)
	      ENDIF
           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF

* "MISSING" - replace missing value flag with values in grid 2

      IF (op .EQ. 30) THEN

	IF (is_const_1 .AND. is_const_2) THEN  ! 2 constants may have been sent in
	   IF (arg1 .NE. bad1 ) THEN
	      res(1) = arg1
	   ELSEIF (arg2 .NE. bad2) THEN 
	      res(1) = arg1
	   ELSE
	      res(1) = bad_res
	   ENDIF
	   RETURN
	ENDIF
	
	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance arguments
           IF (is_inst_1) arg1 = com1(ifeature)
           IF (is_inst_2) arg2 = com2(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)
              IF (is_obs_2) arg2 = com2(iobs)

              IF (arg1 .NE. bad1 ) THEN
	         res(iobs) = arg1
              ELSEIF (arg2 .NE. bad2) THEN 
	         res(iobs) = arg1
              ELSE
	         res(iobs) = bad_res
              ENDIF
           ENDDO

           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF

* "IGNORE0" - replace zeros with bad flags  (single argument)

      IF (op .EQ. 31) THEN

	IF (is_const_1) THEN  
	   IF (arg1.EQ.bad1 .OR. arg1.EQ.0.) THEN
	      res(1) = bad_res
	   ELSE
	      res(1) = arg1
	   ENDIF
	   RETURN
	ENDIF

	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance argument
           IF (is_inst_1) arg1 = com1(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)

	      IF (arg1.EQ.bad1 .OR. arg1.EQ.0.) THEN
	         res(iobs) = bad_res
	      ELSE
	         res(iobs) = arg1
	      ENDIF
           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF

* "ATAN"  (single argument)

      IF (op .EQ. 32) THEN

	IF (is_const_1) THEN  
	   IF (arg1.EQ.bad1) THEN
	      res(1) = bad_res
	   ELSE
	      res(1) = ATAN(arg1)
	   ENDIF
	   RETURN
	ENDIF

	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance argument
           IF (is_inst_1) arg1 = com1(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)

	      IF (arg1.EQ.bad1) THEN
	         res(iobs) = bad_res
	      ELSE
	         res(iobs) = ATAN(arg1)
	      ENDIF
           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF


* "ATAN2" - 2 argument arc tangent

      IF (op .EQ. 33) THEN

	IF (is_const_1 .AND. is_const_2) THEN  ! 2 constants may have been sent in
	   IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2 ) THEN
	      res(1) = bad_res
	   ELSEIF (arg1.EQ.0. .AND. arg2.EQ.0. ) THEN 
	      res(1) = bad_res
	   ELSE
	      res(1) = ATAN2 (arg1, arg2)
	   ENDIF
	   RETURN
	ENDIF
	
	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance arguments
           IF (is_inst_1) arg1 = com1(ifeature)
           IF (is_inst_2) arg2 = com2(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)
              IF (is_obs_2) arg2 = com2(iobs)
	      
	      IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2 ) THEN
	         res(iobs) = bad_res
	      ELSEIF (arg1.EQ.0. .AND. arg2.EQ.0. ) THEN 
	         res(iobs) = bad_res
	      ELSE
	         res(iobs) = ATAN2 (arg1, arg2)
	      ENDIF
	   ENDDO

           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF


* "ASIN"  (single argument)

      IF (op .EQ. 34) THEN

	IF (is_const_1) THEN  
	   IF (arg1.EQ.bad1) THEN
	      res(1) = bad_res
	   ELSE
	      res(1) = ASIN(arg1)
	   ENDIF
	   RETURN
	ENDIF

	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance argument
           IF (is_inst_1) arg1 = com1(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)

	      IF (arg1.EQ.bad1) THEN
	         res(iobs) = bad_res
	      ELSE
	         res(iobs) = ASIN(arg1)
	      ENDIF
           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF


* "ACOS"

      IF (op .EQ. 35) THEN

	IF (is_const_1) THEN  
	   IF (arg1.EQ.bad1) THEN
	      res(1) = bad_res
	   ELSE
	      res(1) = ACOS(arg1)
	   ENDIF
	   RETURN
	ENDIF

	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance argument
           IF (is_inst_1) arg1 = com1(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)

	      IF (arg1.EQ.bad1) THEN
	         res(iobs) = bad_res
	      ELSE
	         res(iobs) = ACOS(arg1)
	      ENDIF
           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF

* "RANDU" - random uniform  (single argument)
* use the first value as the seed

      IF (op .EQ. 36) THEN

	first = .TRUE.
	IF (is_const_1) THEN  
	   IF (arg1.EQ.bad1) THEN
	      res(1) = bad_res
	   ELSE
	      R_IN = REAL(arg1)
	      res(1) = RANDU( R_IN )
	   ENDIF
	   RETURN
	ENDIF

	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance argument
           IF (is_inst_1 .AND. first) arg1 = com1(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1 .AND. first) arg1 = com1(iobs)

	      IF (arg1.EQ.bad1) THEN
	         res(iobs) = bad_res
	      ELSEIF ( first ) THEN
	         R_IN = REAL(arg1)
		 res(iobs) = RANDU( R_IN )
		 first = .FALSE.
	      ELSE
	         R_IN = 0.
		 res(iobs) = RANDU( R_IN )
	      ENDIF

           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF

* "RANDN" - random normal  (single argument)
* use the first value as the seed

      IF (op .EQ. 37) THEN

	first = .TRUE.
	IF (is_const_1) THEN  
	   IF (arg1.EQ.bad1) THEN
	      res(1) = bad_res
	   ELSE
	      R_IN = REAL(arg1) 
	      res(1) = RANDN( R_IN )
	   ENDIF
	   RETURN
	ENDIF

	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance argument
           IF (is_inst_1 .AND. first) arg1 = com1(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1 .AND. first) arg1 = com1(iobs)

	      IF (arg1.EQ.bad1) THEN
	         res(iobs) = bad_res
	      ELSEIF ( first ) THEN
	         R_IN = REAL(arg1)
		 res(iobs) = RANDN( R_IN )
		 first = .FALSE.
	      ELSE
	         R_IN = 0.
		 res(iobs) = RANDN( R_IN )
	      ENDIF

           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF

* "RHO" - UNESCO equation of state:
* rho = rho(salinity, temperature, pressure)
* three components - each may be a different XYZT shape

      IF (op .EQ. 38) THEN

	IF (is_const_1 .AND. is_const_2 .AND. is_const_3) THEN  ! all constants may have been sent in
	   IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2) THEN
	      res(1) = bad_res
	   ELSE
	      res(1) = RHO_UNESCO( arg1, arg2, arg3 )
	   ENDIF
	   RETURN
	ENDIF
	
	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance arguments
           IF (is_inst_1) arg1 = com1(ifeature)
           IF (is_inst_2) arg2 = com2(ifeature)
           IF (is_inst_3) arg3 = com3(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)
              IF (is_obs_2) arg2 = com2(iobs)
              IF (is_obs_3) arg3 = com3(iobs)

	      IF ( arg1.EQ.bad1 .OR. arg2.EQ.bad2  .OR. arg3.EQ.bad3) THEN
	        res(iobs) = bad_res
	      ELSE
	        res(iobs) = RHO_UNESCO( arg1, arg2, arg3 )
	      ENDIF
           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF

* "THETA" - potential temperature from BRYDEN,H.,1973,DEEP-SEA RES
*           and FOFONOFF,N,M,1977,DEEP-SEA RES
* theta = theta(salinity, temperature, pressure,reference_pressure)
* three components - each may be a different XYZT shape

      IF (op .EQ. 39) THEN

	IF (is_const_1 .AND. is_const_2 .AND. is_const_3 .AND. is_const_4) THEN  ! all constants?
	   IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2) THEN
	      res(1) = bad_res
	   ELSE
	      res(1) = THETA_FOFF( arg1, arg2, arg3, arg4)
	   ENDIF
	   RETURN
	ENDIF
	
	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance arguments
           IF (is_inst_1) arg1 = com1(ifeature)
           IF (is_inst_2) arg2 = com2(ifeature)
           IF (is_inst_3) arg3 = com3(ifeature)
           IF (is_inst_4) arg4 = com4(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)
              IF (is_obs_2) arg2 = com2(iobs)
              IF (is_obs_3) arg3 = com3(iobs)
              IF (is_obs_4) arg4 = com4(iobs)

	      IF ( arg1.EQ.bad1 .OR. arg2.EQ.bad2 .OR.
     .             arg3.EQ.bad3 .OR. arg4.EQ.bad4) THEN
	        res(iobs) = bad_res
	      ELSE
	        res(iobs) = THETA_FOFF( arg1, arg2, arg3, arg4)
	      ENDIF
           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF


* "DAYS1900(year, month, day)" - number of days since 1900
* three components - each may be a different XYZT shape
* *** note: returns days since 1-jan-1900, on the STANDARD CALENDAR

      IF (op .EQ. 40) THEN

	IF (is_const_1 .AND. is_const_2 .AND. is_const_3) THEN  ! all constants?
	   IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2 .OR. arg3.EQ.bad3 .OR.
     .         arg2.LT.1 .OR. arg2.GT.12 .OR. arg3.LT.0) THEN
	      res(1) = bad_res
	   ELSE
	      res(1) = DAYS_FROM_DAY0( pdays_by_1900, arg1, arg2, arg3, dummy)
	   ENDIF
	   RETURN
	ENDIF
	
	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance arguments
           IF (is_inst_1) arg1 = com1(ifeature)
           IF (is_inst_2) arg2 = com2(ifeature)
           IF (is_inst_3) arg3 = com3(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)
              IF (is_obs_2) arg2 = com2(iobs)
              IF (is_obs_3) arg3 = com3(iobs)


	      IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2 .OR. arg3.EQ.bad3 .OR.
     .            arg2.LT.1 .OR. arg2.GT.12 .OR. arg3.LT.0) THEN
	         res(iobs) = bad_res
	      ELSE
	         res(iobs) = DAYS_FROM_DAY0( pdays_by_1900, arg1,
     .                        arg2, arg3, dummy)
	      ENDIF


           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF

* "RANDU2" - random uniform 
* Call the gfortran RANDOM_SEED and RANDOM_SEED functions.
* See ticket 1886 for discussions of RANDU
*
* iseed = 0   use previous seed - don't reinitialize
* iseed = -1 initialize using system clock
* iseed = integer > 0 initialize using that integer: results will be repeatable

      IF (op .EQ. 41) THEN

	IF (.NOT. is_const_2) THEN
	   RETURN  ! shouldn't get here
	ENDIF

	inseed = INT(arg2)
	IF (inseed.LT.0 .AND. inseed.NE.-1) inseed = -1
	CALL INIT_RANDOM_SEED(inseed)

	IF (is_const_1 .AND. is_const_2) THEN  ! 2 constants may have been sent in
	   IF ( arg1.EQ.bad1 ) THEN
	      res(1) = bad_res
	   ELSE
	      CALL RANDOM_NUMBER( res(1) )
	   ENDIF
	   RETURN
	ENDIF
	
	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance arguments
           IF (is_inst_1) arg1 = com1(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)

	      IF ( arg1.EQ.bad1 ) THEN
	         res(iobs) = bad_res
	      ELSE
	         CALL RANDOM_NUMBER( res(iobs) )
	      ENDIF
	   
           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF


* "RANDN2" - random normal using gfortran RANDOM_SEED and RANDOM_SEED functions.
* seed set as for RANDU2

      IF (op .EQ. 42) THEN

	IF (.NOT. is_const_2) THEN
	   RETURN  ! shouldn't get here
	ENDIF

	inseed = INT(arg2)
	first = .TRUE.

	IF (is_const_1 .AND. is_const_2) THEN  ! 2 constants may have been sent in
	   IF ( arg1.EQ.bad1 ) THEN
	      res(1) = bad_res
	   ELSE
	      res(1) = RANDN2 (inseed )
	   ENDIF
	   RETURN
	ENDIF
	
	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance arguments
           IF (is_inst_1) arg1 = com1(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)

              IF ( arg1 .EQ. bad1 ) THEN
                 res(iobs) = bad_res
              ELSEIF ( first ) THEN
                 res(iobs) = RANDN2( inseed )
                 first = .FALSE.
              ELSE
                 res(iobs) = RANDN2(0)
              ENDIF
           ENDDO

           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF

      IF (op .EQ. 43) THEN
	STOP 'FUNCTION NOT IMPLEMENTED'
      ENDIF

* ----------------------------------------------------------------------------
* ------- LOGIC STRUCTURES -------------
*
      IF (op.EQ.44 .OR. op.EQ.45) THEN

* "THEN" - comes from IF (condition) THEN (choice)   ... no "ELSE" given
* component 1 is the condition.  component 2 is the choice
	base = 0    ! obs index at end of preceding feature

* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance arguments
           IF (is_inst_1) arg1 = com1(ifeature)
           IF (is_inst_2) arg2 = com2(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)
              IF (is_obs_2) arg2 = com2(iobs)

	      IF ( arg1 .NE. bad1
     .       .AND. arg1 .NE. 0.0
     .       .AND. arg2 .NE. bad2 ) THEN
	        res(iobs) = arg2
	      ELSE
	        res(iobs) = bad_res
	      ENDIF
           ENDDO
           base = base + flen
        ENDDO
	RETURN

      ENDIF

* "ELSE" - comes from IF (condition) THEN (choice1) ELSE (choice2)
* component 1 is the condition.  components 2 and 3 are the choices
      IF (op .EQ. 46) THEN

	base = 0    ! obs index at end of preceding feature
* ... loop over the features
        DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance arguments
           IF (is_inst_1) arg1 = com1(ifeature)
           IF (is_inst_2) arg2 = com2(ifeature)
           IF (is_inst_3) arg3 = com3(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)
              IF (is_obs_2) arg2 = com2(iobs)
              IF (is_obs_3) arg3 = com3(iobs)

	      IF ( arg1 .NE. bad1 .AND. arg1 .NE. 0.0 ) THEN
	         IF ( arg2 .NE. bad2 ) THEN
	            res(iobs) = arg2
	         ELSE
	            res(iobs) = bad_res
	         ENDIF
	      ELSE
	         IF ( arg3 .NE. bad3 ) THEN
	            res(iobs) = arg3
	         ELSE
	            res(iobs) = bad_res
	         ENDIF
	      ENDIF
           ENDDO
           base = base + flen
        ENDDO
	RETURN

      ENDIF

* "THEN" - comes from IFV (condition) THEN (choice)   ... no "ELSE" given
* component 1 is the condition.  component 2 is the choice

      IF (op .EQ. 47) THEN

	IF (is_const_1 .AND. is_const_2) THEN  ! all constants?
	   IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2) THEN
	      res(1) = bad_res
	   ELSE
	      res(1) = arg2
	   ENDIF
	   RETURN
	ENDIF
	
	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance arguments
           IF (is_inst_1) arg1 = com1(ifeature)
           IF (is_inst_2) arg2 = com2(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)
              IF (is_obs_2) arg2 = com2(iobs)

	      IF (arg1.EQ.bad1 .OR. arg2.EQ.bad2) THEN
	         res(iobs) = bad_res
	      ELSE
	         res(iobs) = arg2
	      ENDIF


           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF


* "ELSE" - comes from IFV (condition) THEN (choice1) ELSE (choice2)
* component 1 is the condition.  components 2 and 3 are the choices

      IF (op .EQ. 48) THEN

	IF (is_const_1 .AND. is_const_2 .AND. is_const_3) THEN  ! all constants?
	   IF (arg1 .NE. bad1) THEN
	      IF (arg2 .NE. bad2) THEN
	         res(1) = arg2
	      ELSE
	         res(1) = bad_res
	      ENDIF
	   ELSE
	      IF (arg3 .NE. bad3) THEN
	         res(1) = arg3
	      ELSE
	         res(1) = bad_res
	      ENDIF
	   ENDIF
	   RETURN
	ENDIF
	
	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 

           IF (is_obs_res) THEN
	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
           ELSE
              flen = 1  ! inner loop has no meaning for instance result
           ENDIF

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) THEN
              base = base + flen
              CYCLE
           ENDIF

* ... set the instance arguments
           IF (is_inst_1) arg1 = com1(ifeature)
           IF (is_inst_2) arg2 = com2(ifeature)
           IF (is_inst_3) arg3 = com3(ifeature)

           iobs = base

* ... loop over the observations within each feature
	   DO i = 1, flen
	      iobs = iobs + 1   ! index in the contig ragged array

* ... set the obs arguments
              IF (is_obs_1) arg1 = com1(iobs)
              IF (is_obs_2) arg2 = com2(iobs)
              IF (is_obs_3) arg3 = com3(iobs)

              IF (arg1 .NE. bad1) THEN
	         IF (arg2 .NE. bad2) THEN
	            res(iobs) = arg2
	         ELSE
	            res(iobs) = bad_res
	         ENDIF
	      ELSE
	         IF (arg3 .NE. bad3) THEN
	            res(iobs) = arg3
	         ELSE
	            res(iobs) = bad_res
	         ENDIF
	      ENDIF
           ENDDO
           base = base + flen
        ENDDO
	base = 0
	RETURN

      ENDIF

      RETURN
      END
