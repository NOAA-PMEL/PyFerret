	SUBROUTINE MERGE_CONTEXT( its_gc, iarg,
     .				  mrg_cx, com_cx, com_str, status )

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

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/grid_chg_fcns.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xtext_info.cmn'
	include	'ferret_cmn/xfr_grid.cmn'
	include 'ferret_cmn/xrisc.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XTEXT_INFO.CMN'
	INCLUDE	'FERRET_CMN:XFR_GRID.CMN'
	INCLUDE 'FERRET_CMN:XRISC.CMN'
#endif

* calling argument declarations
	LOGICAL		its_gc
	INTEGER		iarg, mrg_cx, com_cx, status
	CHARACTER*(*)	com_str

* local variable declarations
	LOGICAL		ss_save
	INTEGER		CGRID_AXIS, CX_DIM_LEN, WHICH_GCFCN,
     .			idim, mrg_ax, mrg_len, com_ax, com_len, 
     .			choice(4,5), i, icom, imrg, uvar,
     .			com_str_slen, mrg_str_slen,
     .			com_int_slen, mrg_int_slen, gcfcn,
     .			axis_source(4), lo_off(4),  hi_off(4), action(4)
	CHARACTER	CX_DIM_STR*16, LEFINT*8, com_lims*16, mrg_lims*16,
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
* get the variable under consideration from the merged context
	uvar = cx_variable( mrg_cx )

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
	DO 500 idim = 1,4

* argument of grid-changing function? (1/97)
	IF (its_gc ) THEN
	   IF (    axis_source(idim).EQ.pgc_ignore_axis) THEN
	      GOTO 500
	   ELSEIF (lo_off(idim).NE.0 .OR. hi_off(idim).NE.0) THEN
* ... cannot merge limits from axis with offset because we might be right
*     up to the grid edge where offsets couldn't be applied
	      GOTO 500
	   ELSEIF (axis_source(idim).EQ.pgc_impose_axis) THEN
	      IF (action(idim) .EQ. pset_to_unspec) THEN
* ... IS_DO_GC_FUNC will fill in the result limits on the imposed axes
	         GOTO 500  ! leave it in its initialized (vacant) condition
	      ELSE
* ... This is the special case of an imposed axis that matches the argument ax
	         CONTINUE
	      ENDIF
	   ELSE
* ... it is evidently, a mergible axis ... continue on
	      CONTINUE
	   ENDIF	      
	ENDIF

* ignore axes for which region isn't known (e.g. X axis of pseudo-variable "Y")
	   IF ( cx_lo_ww(idim,com_cx) .EQ. unspecified_val8 ) GOTO 500

*********************************************************************
* "documentation merge" option - a patch: 10/27/92
* For special case where both the component and the merged context represent
* idle, redundant documentation  e.g. TEMP[L=1:5@AVE] + TEMP[L=2:3@AVE]
* If the component context is the same as the merged then save it;
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

* get axes and lengths of data along them
	   mrg_ax  = CGRID_AXIS( idim, mrg_cx )
	   com_ax  = CGRID_AXIS( idim, com_cx )
	   mrg_len = CX_DIM_LEN( idim, mrg_cx )
	   com_len = CX_DIM_LEN( idim, com_cx )

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
 9000	CALL ERRMSG( ferr_internal, status, 'merge_cx', *5000 )
	END
