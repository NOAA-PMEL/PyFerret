	SUBROUTINE OV_GRP_CX_DOC( ov_cx, ncx, flags )

* merge additional variable contexts with those already merged into the
* joint context for a plot to produce a list of documentation flags suitable
* to make a key for this new context.  See GRP_CX_DOC

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:	  2/2/90
* V230:  7/22/92 - added overlay argument to ADDL_CX_DOC

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xplot_setup.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XPLOT_SETUP.CMN'
#endif

* calling argument declarations
	INTEGER ncx, ov_cx(ncx)
	LOGICAL	flags(0:10)

* internal variable declarations
	INTEGER	i

* internal parameter declarations:
        LOGICAL overlay
        PARAMETER ( overlay = .TRUE. )
           
* make a fresh copy of cx_plot and key_doc to keep current with the plot labels
	CALL TRANSFER_CONTEXT( cx_plot, cx_buff )
	DO 100 i = 0, 10
 100	flags(i) = key_doc(i)

* merge the additional contexts
	DO 200 i = 1, ncx
 200	CALL ADDL_CX_DOC( cx_buff, ov_cx(i), overlay, flags )

	RETURN
	END
