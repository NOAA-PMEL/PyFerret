	SUBROUTINE MERGE_CONTEXT( its_gc, iarg, dflt_cx,
     .				  mrg_cx, com_cx, com_str, status )

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
* combine new space/time/data_set/variable/grid context information into a 
* single context checking for consistency 
* (the merged grid is determined in advance by IS_UVAR_GRID)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/20/89 - complete re-write
*	  4/2/90 - beef-up error message for incompatible axis lengths
* Unix/RISC port - 1/91 - cant use "//" with CHAR*(*) variable
*       10/27/92 - "documentation merge" option added
* V450: 1/97	 - changes to accomodate grid changing functions.
*		   MERGE_CONTEXT needs to know the argument number. Use this
*		   to determine the nature of the context merging.
* V533: *sh* 6/01 - merge data types (removed to MERGIFY_CX)
*		  - allow longer char buffer for error message to prevent
*			crash in pathological cases
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* V720 *sh* 5/17 - incorporate merging of a DSG context

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'grid_chg_fcns.parm'
	include	'xcontext.cmn'
	include	'xtext_info.cmn'
	include	'xfr_grid.cmn'
	include 'xrisc.cmn'

* calling argument declarations
	LOGICAL		its_gc
	INTEGER		iarg, dflt_cx, mrg_cx, com_cx, status
	CHARACTER*(*)	com_str

* local variable declarations
	LOGICAL		TM_ITSA_DSG, ss_save, its_dsg, its_cmpnd
	INTEGER		CGRID_AXIS, CX_DIM_LEN, WHICH_GCFCN,
     .			idim, mrg_ax, mrg_len, com_ax, com_len, 
     .			choice(4,5), i, icom, imrg, uvar,
     .			com_str_slen, mrg_str_slen,
     .			com_int_slen, mrg_int_slen, gcfcn,
     .			axis_source(nferdims), lo_off(nferdims),  
     .			hi_off(nferdims), action(nferdims),
     .			dsg_orientation, dsg_obs_dimlen, dsg_obs_line

	CHARACTER	CX_DIM_STR*24, LEFINT*8, com_lims*24, mrg_lims*24,
     .			com_int*8, mrg_int*8

* local parameter declarations:
	LOGICAL         no_decims
	INTEGER		TAX,   ERR,   BUG,   NA
	PARAMETER     ( TAX=1, ERR=2, BUG=3, NA=4,
     .			no_decims = .FALSE. )

* procedure:

* A major question to answer in determining a merged context is: Where does
* the context depend on the default region?  For example, TEMP[K=5] + TEMP[K=1]
* requires no additional Z axis input and no Z axis labelling is needed (since
* the information is part of the definition).  This non-dependence question
* must be resolved AT THE OUTSET or memory resident data cannot be located
* (cannot search memory without a fully specified context to look for).
*
* To address this question each context axis has the logical cx_given.
* This flag tells if the coordinates for that axis were explicitly modified
* in the most recent context level (e.g. TEMP[K=4] explicitly modifies the
* K/Z axis).

* Conformability of component dimensions must also be addressed.  An axis span
* of 1 can be combined with anything (e.g. TEMP[K=1] + TEMP) but lengths
* greater than 1 must match (e.g. TEMP[I=100:105] + TEMP[I=100:106] is illegal).

* When multiple component have differing underlying axis definitions
* another ambiguity must be resolved.
* (e.g. the vars. U and W are on staggered Z axes in many models so what is the
* Z axis of the expression U+W ?  How about U+W[K=1] ? )

* Ambiguities and conflicts are resolved by arbitrarily choosing:
*	1) the region from the first component with region IMPLICITLY defined
*	      ... or failing that ...
*	2) the region fom the first component with explicit region >1
*	      ... or failing that ...
*	3) the region from the first component with any explicit region

* Expressions which are given on the command line present a special case.
* e.g. LIST U[Z=10] should have the title of U and a depth label of 10 but
*      LIST VNEW after DEFINE VAR VNEW=U[Z=10] should have a title of VNEW
*      and no Z depth label.
* Thus on-command-line expressions are sent to this routine only when they
* have multiple component variables.

* The following table summarizes the logic of merging a new component context
* (com_cx) into a partially merged context (mrg_cx):
* "cx_given" on a component axis means that the location was given with the
*	     component (e.g. U[Z=0]) so it doesn't require outside information
* "cx_given" on a the merged context means that all components had cx_given

*	            1           2  COMPONENT    3             4
*		      cx_given		         .NOT.cx_given
*  size:            1          >1	        1	       >1
*             -----------------------------------------------------------------
*1          1 |   no action   | transfer axis | transfer axis | transfer axis |
* cx_given    -----------------------------------------------------------------
*2         >1 |   no action   |  no action    |    error      | transfer axis |
*	      -----------------------------------------------------------------
*3          1 |   no action   |    error      |  no action    | transf ax *** |
* NOT.cx_gvn  -----------------------------------------------------------------
*4         >1 |   no action   |  no action    |  no actn  *** |   no action   |
*             -----------------------------------------------------------------
*5 normal     | transfer axis | transfer axis | transfer axis | transfer axis |
*             -----------------------------------------------------------------

* error : cannot label expressions like yes? CONTOUR/K=1 U[K=1:3] - U
*	   since the first component requires a labeled axis and the
*	   second requires a single point to be named
* "***"	:  situation can occur only in cases like
*		DEFINE VAR TAVE = TEMP[K=@AVE]
*		DEFINE VAR DIFF = TEMP - TAVE
* ... so that both TEMP and TAVE have limits .NOT.given but one has length 1
*	and the other has length possible .GT. 1

	DATA (choice(i,1),i=1,4) / NA,  TAX, TAX, TAX /,
     .	     (choice(i,2),i=1,4) / NA,   NA, ERR, TAX /,
     .	     (choice(i,3),i=1,4) / NA,  ERR,  NA, TAX /,
     .	     (choice(i,4),i=1,4) / NA,   NA,  NA,  NA /
     .	     (choice(i,5),i=1,4) / TAX, TAX, TAX, TAX /

**************************************************************************
* get the DSG-ness -- GET_UVAR_GRID ensures valid DSG grid merging
	its_dsg = TM_ITSA_DSG(cx_grid(mrg_cx))
	IF (its_dsg) THEN
	   CALL TM_DSG_FACTS(cx_grid(com_cx), dsg_orientation, 
     .		dsg_obs_dimlen, dsg_obs_line, its_dsg, its_cmpnd)
	ENDIF

* get the variable under consideration from the merged context
	uvar = cx_variable( mrg_cx )

* merge data types  (removed to MERGIFY_CX -- uvars via uvar_data_type)
!	IF (cx_type(mrg_cx) .EQ. ptype_unknown) THEN
!	   cx_type(mrg_cx) = cx_type(com_cx)
!	ELSEIF( cx_type(mrg_cx) .NE. cx_type(com_cx) ) THEN
!! temporary hack until function return types are implemented
!	   cx_type(mrg_cx) = ptype_float
!	ENDIF

* is this a grid changing variable? (1/97)
	IF (its_gc) THEN
	   gcfcn = WHICH_GCFCN(uvar)
	   CALL GCF_GET_AXIS_SRCS(gcfcn, iarg, axis_source)
	   CALL GCF_GET_ARG_LIMS (gcfcn, iarg, com_cx, cx_grid(mrg_cx),
     .				 lo_off,  hi_off, action, status)
	   IF ( status .NE. ferr_ok ) GOTO 5000
	ENDIF

* initialize fixed length string for RISC
	risc_buff = com_str
	len_rbuff = MIN( size_rbuff, LEN(com_str) )

* loop through the axes
	DO 500 idim = 1,nferdims

* argument of grid-changing function? (1/97)
*
*     If we should ignore the axis:     don't merge
*     Else if we should impose an axis 
*        If (pset_to_unspec):           don't merge
*        Else (ppass_along):            imposed axis matches arg axis --> merge
*     Else (we should merge an axis)
*        If we have offsets:            special merge
*        Else:                          all OK --> merge
*     Endif

           IF (its_gc ) THEN

              IF (axis_source(idim).EQ.pgc_ignore_axis) THEN

                 GOTO 500

              ELSEIF (axis_source(idim).EQ.pgc_impose_axis) THEN

                 IF (action(idim) .EQ. pset_to_unspec) THEN
*     ... IS_DO_GC_FUNC will fill in the result limits on the imposed axes
                    GOTO 500    ! leave it in its initialized (vacant) condition
                 ELSE
*     ... This is the special case of an imposed axis that matches the argument ax
                    CONTINUE
                 ENDIF

              ELSE !(axis_source(idim) .EQ. pgc_merge_axis)

                 IF  (lo_off(idim).NE.0 .OR. hi_off(idim).NE.0) THEN
                    CALL MERGE_AXIS_WITH_OFFSETS(dflt_cx, mrg_cx, 
     .                   com_cx, idim, lo_off, hi_off)
                    GOTO 500
                 ELSE
                    CONTINUE
                 ENDIF

              ENDIF	 
              
           ENDIF


* ignore axes for which region isn't known (e.g. X axis of pseudo-variable "Y")
	   IF ( cx_lo_ww(idim,com_cx) .EQ. unspecified_val8 ) GOTO 500

* get axes and lengths of data along them
	   mrg_ax  = CGRID_AXIS( idim, mrg_cx )
	   com_ax  = CGRID_AXIS( idim, com_cx )
	   mrg_len = CX_DIM_LEN( idim, mrg_cx )
	   com_len = CX_DIM_LEN( idim, com_cx )

* Merging DSG contexts (5/17)
* Only the F axis of a DSG is a "gridded" axis
* On all others the DSG may have world coordinates, but the subscript limits
* are irrelevant.  (Always full limits.  No subsetting is possible.)
	   IF (its_cmpnd) dsg_orientation = z_dim
	   IF (its_dsg .AND. idim.NE.f_dim) THEN
* ... insist that consistent constraints are given on all merged contexts
	      IF (cx_lo_ww(idim,mrg_cx) .EQ. unspecified_val8) THEN
* ... first component: its world constraints are THE merged constraints
	         cx_lo_ww(idim,mrg_cx) = cx_lo_ww(idim,com_cx)
	         cx_hi_ww(idim,mrg_cx) = cx_hi_ww(idim,com_cx)
	      ELSEIF (cx_lo_ww(idim,mrg_cx) .EQ. cx_lo_ww(idim,com_cx)
     .		.AND. cx_hi_ww(idim,mrg_cx) .EQ. cx_hi_ww(idim,com_cx)) THEN
* ... new component matches merged constraints.  Nothing more to do.
	         GOTO 500
	      ELSEIF (idim .EQ. dsg_orientation) THEN
* ... instance vars and obs vars can be combined in IS_DO_OP
	         CONTINUE	         
	      ELSE
* ... inconsistent DSG constraints
*     e.g. 'lon[T="1-MAY-2017"] GT 180 AND rowSize[T="5-OCT-1995"] LT 35'
	         GOTO 5030
	      ENDIF
	      IF (com_ax.EQ.mnormal .OR. mrg_ax.EQ.mnormal) THEN
* ... since a DSG context carries ww constraints on mnormal axes need to bypass
	         GOTO 500
	      ENDIF
	   ENDIF

*********************************************************************
* "documentation merge" option - a patch: 10/27/92
* For special case where both the component and the merged context represent
* idle, redundant documentation  e.g. TEMP[L=1:5@AVE] + TEMP[L=2:3@AVE]
* If the component context is the same as the merged then save it
* if they differ then throw it away
           IF (cx_given(idim,com_cx)
     .           .AND. cx_lo_ss(com_cx,idim) .EQ. unspecified_int4
     .    .AND.cx_given(idim,mrg_cx)
     .           .AND. cx_lo_ss(mrg_cx,idim) .EQ. unspecified_int4) THEN
              IF ( cx_lo_ww(idim,com_cx).NE.cx_lo_ww(idim,mrg_cx)
     .       .OR.  cx_hi_ww(idim,com_cx).NE.cx_hi_ww(idim,mrg_cx) ) THEN
                 cx_lo_ww(idim,com_cx)= unspecified_val8
                 cx_hi_ww(idim,com_cx)= unspecified_val8
              ENDIF
           ENDIF
*********************************************************************

* PRE-TEST FOR ERROR CONDITIONS
* ... non-conformable lengths
	   IF ( mrg_len .NE. com_len
     .    .AND. mrg_len.NE.1 .AND. com_len.NE.1 ) GOTO 5010

* ignore data from non-matching axes (warning was given during grid merge)
	   IF ( com_ax .NE. mrg_ax  ) GOTO 500
	   IF ( mrg_ax .EQ. mnormal ) GOTO 9000

* MERGE NEW COMPONENT WITH PREVIOUS DIMENSIONS
	   IF ( cx_given(idim,com_cx) ) THEN
	      icom = 1
	   ELSE
	      icom = 3
	   ENDIF
	   IF ( cx_given(idim,mrg_cx) ) THEN
	      imrg = 1
	   ELSE
	      imrg = 3
	   ENDIF
	   IF ( com_len .GT. 1 ) icom = icom + 1
	   IF ( mrg_len .GT. 1 ) imrg = imrg + 1
	   IF ( cx_lo_ss(mrg_cx,idim) .EQ. unspecified_int4 ) imrg = 5

* merge new component context with previous stuff
*	          TAX   ERR   BUG   NA
	   GOTO ( 100, 5020, 9000, 400 ) choice(icom,imrg)

* transfer axis (component transforms don't belong to the merged context)
 100	   CALL TRANSFER_AXIS( idim, com_cx, mrg_cx )
	   cx_trans    (idim, mrg_cx ) = trans_no_transform
	   cx_trans_arg(idim, mrg_cx ) = bad_val4

* "given" is false if any component doesn't have an explicit region
 400	   cx_given(idim,mrg_cx) = cx_given(idim,mrg_cx)
     .			     .AND. cx_given(idim,com_cx)

 500	CONTINUE

* successfully merged
	status = ferr_ok
	RETURN

* error exit
 5000	RETURN
* ... message to include "expr has N points (L=lo:hi)"
 5010	ss_save = cx_by_ss(idim,com_cx)
	cx_by_ss(idim,com_cx) = .TRUE.
	com_lims = CX_DIM_STR( idim,com_cx,':',no_decims,com_str_slen )
	cx_by_ss(idim,com_cx) = ss_save
	ss_save = cx_by_ss(idim,mrg_cx)
	cx_by_ss(idim,mrg_cx) = .TRUE.
	mrg_lims = CX_DIM_STR( idim,mrg_cx,':',no_decims,mrg_str_slen )
	cx_by_ss(idim,mrg_cx) = ss_save
	com_int = LEFINT(com_len,com_int_slen)
	mrg_int = LEFINT(mrg_len,mrg_int_slen)
	CALL ERRMSG( ferr_inconsist_plane, status,
     .		ww_dim_name(idim)//' axis'//pCR//
     .		risc_buff(:len_rbuff)//
     .		' has '//com_int(:com_int_slen)//' points ('//
     .					com_lims(:com_str_slen)//')'//pCR//
     .		'expression has '//mrg_int(:mrg_int_slen)//' points ('//
     .					mrg_lims(:mrg_str_slen)//')', *5000 )
 5020	CALL ERRMSG( ferr_inconsist_plane+ghelp_cue, status,
     .		'can''t label '//ww_dim_name(idim)//' axis of '//
     .		risc_buff(:len_rbuff), *5000 )
 5030	CALL ERRMSG( ferr_inconsist_plane+ghelp_cue, status,
     .		'feature constraints on '//ww_dim_name(idim)//
     .          ' axis of '//risc_buff(:len_rbuff), *5000 )
 9000	CALL ERRMSG( ferr_internal, status, 'merge_cx', *5000 )
	END
