	SUBROUTINE GRP_CX_DOC( cx_list, ncx, joint_cx, flags )

* set up joint context and document flags to summarize the contexts
* of one or more variables that will be plotted (or listed) simultaneously
* For example, PLOT U[Z=20],U[Z=50] will place data set and X,Y,T locations
* in cx_plot and set up key_doc to display the Z location
* This routine is called at the start of each fresh (non-overlay) plot

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:	  2/2/90
* V230:  7/22/92 - added overlay argument to ADDL_CX_DOC

#ifdef unix
	include	'ferret_cmn/ferret.parm'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations
	LOGICAL	flags(0:10)
	INTEGER	ncx, cx_list(ncx), joint_cx

* internal variable declarations
	INTEGER	i, idim

* internal parameter declarations:
        LOGICAL overlay
        PARAMETER ( overlay = .TRUE. )

* initialize the key documentation flags
	DO 100 i = 0, 10
 100	flags(i) = .FALSE.

* the context of the first variable is the starting point
	CALL TRANSFER_CONTEXT( cx_list(1), joint_cx )

* merge the other contexts with it
	DO 200 i = 2, ncx
 200	CALL ADDL_CX_DOC( joint_cx, cx_list(i), .NOT.overlay, flags )

* regridding is always displayed on the key, not the joint context
!	cx_unstand_grid( cx_plot ) = .FALSE.
	RETURN
	END
