	SUBROUTINE TM_IRREGAX_MID( ipt1, npts )

* Fill in the midpoint values as the default boundaries between grid
* boxes

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/93 - code lifted from cd_get_1_axis

* argument definitions:
*	ipt1	- subscript of first coordinate for axis in line_memory
*	npt	- length of axis

* argument declarations
	INTEGER	 ipt1, npts

* internal variable declarations:
        INTEGER i

* include files
#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/xio.cmn_text'
	external xio_data
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif

* compute midpoints lying between coordinates
        DO 100 i = ipt1, ipt1+npts-2
 100    line_mem(i+npts+1) = (line_mem(i)+line_mem(i+1)) / 2.0D0 

* lower bound of first box
        line_mem(ipt1+npts)   = line_mem(ipt1)
     .                        - (line_mem(ipt1+1)-line_mem(ipt1))/2.D0
* upper bound of last box
        line_mem(ipt1+npts+npts) = line_mem(ipt1+npts-1)
     .            + (line_mem(ipt1+npts-1)-line_mem(ipt1+npts-2))/2.D0

        RETURN
	END
