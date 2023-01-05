	SUBROUTINE MAKE_PSEUDO_VAR ( cx, idim, pvar, mr, lo_ss, hi_ss )

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
* create a "pseudo-variable" - numerical grid of data based solely on the
* underlying grid (not involving external data)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/2/87
* V200:  6/20/89 - 4D-symmetrical version
* V530   1/01 - *sh* - added XBOXLO, XBOXHI and family ...
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
*       *acm*      more 4s that should have been nferdims
*      *acm* 4/12 Add new MODE FOURDEE to allow E,F,M,N,etc as user-vars
* V71 12/16 *acm* For ticket 2158: working with true monthly time axes.
*                 (no actual changes, just modernizing the code a bit.)
* V720 *sh* 5/17 support for DSG coordinates as pseduo-variables
*  1/2023 *acm* Code cleanup: Replace the computed GOTO statement 
*                  "GOTO ( 100, 200, 300..) pv"  with IF blocks

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include	'xalgebra.cmn'
	include	'xdset_info.cmn_text'
	include 'xdyn_linemem.cmn_text'

* calling argument declarations:
	INTEGER	cx, mr, idim, lo_ss, hi_ss
	REAL	pvar( lo_ss:hi_ss )

* internal variable declarations:
	LOGICAL	TM_ITSA_DSG, its_dsg
	INTEGER	TM_DSG_DSET_FROM_GRID, TM_DSG_NFEATURES,
     .		grid, dset, pv, i, iaxis,
     .		dsg_var, dsg_lm, iobs, nobs, ifeature, nfeatures
	REAL	BOX_SIZE, TM_WORLD

* initialize

* defining grid for output data
	grid = cx_grid( cx )
	its_dsg = idim .NE. f_dim .AND. TM_ITSA_DSG(grid)

	IF (its_dsg) THEN
	   dset = TM_DSG_DSET_FROM_GRID(grid)
	   IF (idim .EQ. e_dim) THEN
	      dsg_var = dsg_row_size_var(dset)
	   ELSE
	      dsg_var = dsg_coord_var(idim, dset)
	   ENDIF
	   dsg_lm = dsg_loaded_lm(dsg_var)
	ELSE
	   dset = cx_data_set( cx )
	ENDIF

	pv = (mr_variable(mr)-1) / nferdims  + 1

	IF ( pv .GT. nferdims+1 )  STOP 'MK_PSEUDO_VAR'	! diag


* "I", "J", "K", "L", "M", "N"

      IF (pv .EQ. 1) THEN

	IF (its_dsg) THEN
	   nfeatures = TM_DSG_NFEATURES(grid)
	   IF (lm_size(dsg_lm) .EQ. nfeatures) THEN
* ... DSG instance indices
	      DO ifeature = lo_ss, hi_ss   ! always 1 to nfeatures
	         pvar(ifeature) = ifeature
	      ENDDO
	   ELSE
* ... DSG obs indices - start from 1 on each feature
	      dsg_var = dsg_row_size_var(dset)
	      dsg_lm  = dsg_loaded_lm(dsg_var)
	      i = 0
	      DO ifeature = 1, nfeatures
	         nobs = dsg_linemem(dsg_lm)%ptr(ifeature)
	         DO iobs = 1, nobs
	            i = i + 1
	            pvar( i ) = iobs
	         ENDDO
	      ENDDO
	   ENDIF
	ELSE
	   DO i = lo_ss, hi_ss
	      pvar( i ) = i
	   ENDDO
	ENDIF
	RETURN

      ENDIF

* "X", "Y", "Z", "T", "E", "F"
* TM_WORLD returns days if it is a true-month time axis

      IF (pv .EQ. 2) THEN

	IF (its_dsg) THEN
!DSGTBD: more efficient would be to just retuen the pointer dsg_linemem(dsg_lm)%ptr
!  an example of how to do this is found in FIND_MEM_VAR
!  In IS_PSEUDO would need to trap the X,Y,Z,T pseudo-variable case
	   DO i = lo_ss, hi_ss   ! always full DSG obs or instance dimension
	      pvar(i) = dsg_linemem(dsg_lm)%ptr(i)
	   ENDDO
	ELSE
	   DO i = lo_ss, hi_ss
	      pvar( i ) = TM_WORLD( i,grid,idim,box_middle )
	   ENDDO
	ENDIF
	RETURN

      ENDIF

* "XBOX", "YBOX", "ZBOX", "TBOX", "EBOX", "FBOX"
!DSGTBD: BOX, BOXLO, and BOXHI are useful values to return for the coordinates
! that are observation variable, but not for the coordinates that are instance
! variables.
!       Currently these are trapped as errors in PSEUDO_VAR_CONTEXT

      IF (pv .EQ. 3) THEN

	DO i = lo_ss, hi_ss
	   pvar( i ) = BOX_SIZE( i, grid, idim )
	ENDDO
	RETURN

      ENDIF

* "XBOXLO", "YBOXLO", "ZBOXLO", "TBOXLO", "EBOXLO", "FBOXLO"
!DSGTBD: e.g. the low value of XBOXLO for a trajectory feature can be returned
!    as the midpoint between X-observation and X-previous-observation

      IF (pv .EQ. 4) THEN

	DO i = lo_ss, hi_ss
	   pvar( i ) =  TM_WORLD( i,grid,idim,box_lo_lim )
	ENDDO
	RETURN

      ENDIF

* "XBOXHI", "YBOXHI", "ZBOXHI", "TBOXHI", "EBOXHI", "FBOXHI"

      IF (pv .EQ. 5) THEN

	DO i = lo_ss, hi_ss
	   pvar( i ) =  TM_WORLD( i,grid,idim,box_hi_lim )
	ENDDO
	RETURN

      ENDIF

      END
