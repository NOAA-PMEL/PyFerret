*
* Definition of GQLWK
* (inquire maximum length of workstation state tables)
* set value to the maximum number of each type of object for a window
*

        SUBROUTINE FGD_GQLWK(wintype, stat, maxpens, maxmarks,
     .                 maxfonts, maxbrushes, maxpatterns, maxcolors)

        IMPLICIT NONE
        INTEGER wintype, stat, maxpens, maxmarks, maxfonts,
     .          maxbrushes, maxpatterns, maxcolors

        INCLUDE 'fgrdel.cmn'

*       Last pen reserved for the temporary pen
        maxpens = maxpenobjs - 1
*       Last marker reserved for the temporary marker
        maxmarks = maxmarkers - 1
*       Fonts not used at this time
        maxfonts = maxfontobjs
*       Brushes
        maxbrushes = maxbrushobjs
*       Set to the number of hash styles supported
*       although this might not be what it actually means
        maxpatterns = 7
*       Last color reserved for the temporary color
        maxcolors = maxcolorobjs - 1
        stat = 0

        RETURN
        END

