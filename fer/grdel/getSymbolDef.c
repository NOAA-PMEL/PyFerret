#include <Python.h> /* make sure Python.h is first */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "FerMem.h"
#include "grdel.h"

/* Environment variable giving the directories containing symbol definitions */
#define SYMDIRS_ENVVAR "FER_PALETTE"

#define MAXLINELEN 2048

/* 
 * Utility function for reading symbol definitions 
 *
 * Arguments:
 *     ptsxptr: pointer to be assigned with the array of vertex X-coordinates
 *     ptsyptr: pointer to be assigned with the array of vertex Y-coordinates
 *     numptsptr: pointer to be assigned with the number of vertices
 *     fillptr: pointer to be assigned with whether the symbol should be filled
 *     symbolname: name of the symbol; not expected to be null-terminated
 *     namelen: length of the symbol name
 *
 * Returns non-zero if successful.  Memory for the X- and Y-coordinate arrays 
 * was allocated using FerMem_Malloc and will need to be freed using FerMem_Free.
 *
 * On failure, grdelerrmsg is assigned and zero is returned and no memory 
 * remains allocated. (NULL will be assigned to ptsxptr and ptsyptr.)
 */
grdelBool getSymbolDef(float **ptsxptr, float **ptsyptr, int *numptsptr, 
                       grdelBool *fillptr, char symbolname[], int namelen)
{
    const char *envval;
    char   symdirs[MAXLINELEN];
    char  *currdir;
    char   filename[MAXLINELEN];
    FILE  *symfile;
    char   dataline[MAXLINELEN];
    int    numpts;
    float *ptsx;
    float *ptsy;
    int    fill;
    char  *strptr;

    /* initialize for error returns */
    *ptsxptr = NULL;
    *ptsyptr = NULL;
    *numptsptr = 0;
    *fillptr = 0;

    /* get the directories from the environment variable */
    envval = getenv(SYMDIRS_ENVVAR);
    if ( envval == NULL ) {
        sprintf(grdelerrmsg, "Environment variable %s is not defined", SYMDIRS_ENVVAR);
        return 0;
    }

    /* make a copy can be messed with */
    if ( strlen(envval) >= MAXLINELEN ) {
        sprintf(grdelerrmsg, "Value of environment variable %s exceeds %d characters", SYMDIRS_ENVVAR, MAXLINELEN);
        return 0;
    }
    strcpy(symdirs, symdirs);

    /* search for the file in each (space-separated) directory specified */
    symfile = NULL;
    currdir = strtok(symdirs, " \t\v\r\n");
    while ( currdir != NULL ) {
        if ( snprintf(filename, MAXLINELEN, "%s/%.*s.sym", currdir, namelen, symbolname) >= MAXLINELEN ) {
            sprintf(grdelerrmsg, "A full path filename exceeds %d characters", MAXLINELEN);
            return 0;
        }
        symfile = fopen(filename, "r");
        if ( symfile != NULL )
            break;
        currdir = strtok(NULL, " \t\v\r\n");
    }
    if ( symfile == NULL ) {
        sprintf(grdelerrmsg, "Definition file not found for symbol %.*s", namelen, symbolname);
        return 0;
    }

    /* quickly get the maximum number of points for allocating memory */
    numpts = 0;
    while ( fgets(dataline, MAXLINELEN, symfile) != NULL ) {
        /* skip comments and blank lines */
        strptr = dataline;
        while ( isspace(*strptr) )
            strptr++;
        if ( (*strptr == '!') || (*strptr == '\0') )
            continue;
        if ( strncasecmp(strptr, "fill", 4) == 0 ) {
            continue;
        }
        numpts++;
    }
    rewind(symfile);

    /* allocate memory for the coordinates */
    ptsx = (float *) FerMem_Malloc(numpts * sizeof(float), __FILE__, __LINE__);
    if ( ptsx == NULL ) {
        sprintf(grdelerrmsg, "Out of memory for array of %d X-coordinates", numpts);
        fclose(symfile);
        return 0;
    }
    ptsy = (float *) FerMem_Malloc(numpts * sizeof(float), __FILE__, __LINE__);
    if ( ptsy == NULL ) {
        sprintf(grdelerrmsg, "Out of memory for array of %d X-coordinates", numpts);
        FerMem_Free(ptsx, __FILE__, __LINE__);
        fclose(symfile);
        return 0;
    }

    /* now carefully read the coordinates */
    numpts = 0;
    fill = 0;
    while ( fgets(dataline, MAXLINELEN, symfile) != NULL ) {
        /* skip comments and blank lines */
        strptr = dataline;
        while ( isspace(*strptr) )
            strptr++;
        if ( (*strptr == '!') || (*strptr == '\0') )
            continue;
        if ( strncasecmp(strptr, "fill", 4) == 0 ) {
            fill = 1;
            continue;
        }
        if ( sscanf(strptr, "%f %f", &(ptsx[numpts]), &(ptsy[numpts])) != 2 ) {
            /* trim off end blank spaces (completely blank lines were already skipped) */
            strptr = &(dataline[strlen(dataline)-1]);
            while ( isspace(*strptr) )
                strptr--;
            *strptr = '\0';
            /* truncate long data lines */
            if ( strlen(dataline) > 65 )
                strcpy(&(dataline[62]), "...");
            /* and assign the error message */
            sprintf(grdelerrmsg, "Invalid coordinates data in %s: %s", filename, dataline);
            FerMem_Free(ptsy, __FILE__, __LINE__);
            FerMem_Free(ptsx, __FILE__, __LINE__);
            fclose(symfile);
            return 0;
        }
        numpts++;
    }
    fclose(symfile);

    /* successful return */
    *ptsxptr = ptsx;
    *ptsyptr = ptsy;
    *numptsptr = numpts;
    *fillptr = fill;
    return 1;
}
