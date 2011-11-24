*
* Definition of GQLWK
* (inquire maximum length of workstation state tables)
* set value to the maximum number of each type of object for a window
*

        SUBROUTINE FGD_GQLWK(wintype, stat, maxpens, maxsyms,
     .                 maxfonts, maxbrushes, maxpatterns, maxcolors)

        IMPLICIT NONE
        INTEGER wintype, stat, maxpens, maxsyms, maxfonts,
     .          maxbrushes, maxpatterns, maxcolors

        INCLUDE 'fgrdel.cmn'

        maxpens = maxpenobjs
        maxsyms = maxsymbolobjs
        maxfonts = maxfontobjs
        maxbrushes = maxbrushobjs
*       Set to the number of hash styles supported
*       although this might not be what it actually means
        maxpatterns = 7
*       Some issues with trying to define too many colors;
*       try reducing the reported number of available color
*       objects that can be created.
        maxcolors = maxcolorobjs - 5
        stat = 0

        RETURN
        END

