	CHARACTER*(*) FUNCTION CX_SIZE_STR( cx,slen )

* create a string describing the size of a 4D region
* for example: 160*100*27*12

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V230:  8/28/92

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations
	INTEGER	cx, slen

* local variable declarations:
	INTEGER CX_DIM_LEN, TM_LENSTR, idim, axlen, llen
	CHARACTER LEFINT*8

* initialize with the X axis
        axlen = CX_DIM_LEN ( x_dim, cx )
	CX_SIZE_STR = LEFINT( axlen, slen )

* loop through the other dimensions
        DO 100 idim = 2, 4
           axlen = CX_DIM_LEN ( idim, cx )
           CX_SIZE_STR = CX_SIZE_STR(1:slen)//'*'//LEFINT( axlen, llen )
           slen = TM_LENSTR( CX_SIZE_STR )
 100    CONTINUE

	RETURN
	END
