#include <Python.h> /* make sure Python.h is first */
#include <dirent.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ferret.h"
#include "FerMem.h"
#include "grdel.h"

/* Environment variable giving the directories containing symbol definitions */
#define SYMDIRS_ENVVAR "FER_PALETTE"

#define SYMFILEEXT ".sym"
#define SYMFILEEXTLEN 4

#define MAXLINELEN 2048

typedef struct SymbolInfo_ {
    struct SymbolInfo_ *next;
    char     *name;
    float    *ptsx;
    float    *ptsy;
    int       namelen;
    int       numpts;
    grdelBool fill;
} SymbolInfo;

/* Could use a LIST if needed, but current code keeps alphabetically ordered */
static SymbolInfo *SymbolInfoList = NULL;

/* Static function prototypes */
static SymbolInfo *readSymbolDefFile(char *filename, char symname[], int symnamelen);
static int         symbolNameFilter(const struct dirent *direntry);
static SymbolInfo *readSymbolDef(char symname[], int symnamelen);

/* 
 * Reads and returns the symbol definition from the specified file. 
 * Memory allocated in and for the returned symbol definition 
 * structure should be freed when no longer needed.
 */
static SymbolInfo *readSymbolDefFile(char *filename, char symname[], int symnamelen) 
{
    SymbolInfo *infoptr;
    FILE  *symfile;
    char   dataline[MAXLINELEN];
    int    numpts;
    char  *strptr;

    symfile = fopen(filename, "r");
    if ( symfile == NULL ) {
        return NULL;
    }

    /* read through just to get the maximum number of points for allocating memory */
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

    /* allocate memory for the symbol information */
    infoptr = (SymbolInfo *) FerMem_Malloc(sizeof(SymbolInfo), __FILE__, __LINE__);
    if ( infoptr == NULL ) {
        fclose(symfile);
        return NULL;
    }
    infoptr->name = (char *) FerMem_Malloc((symnamelen+1) * sizeof(char), __FILE__, __LINE__);
    if ( infoptr->name == NULL ) {
        FerMem_Free(infoptr, __FILE__, __LINE__);
        fclose(symfile);
        return NULL;
    }
    infoptr->ptsx = (float *) FerMem_Malloc(2 * numpts * sizeof(float), __FILE__, __LINE__);
    if ( infoptr->ptsx == NULL ) {
        FerMem_Free(infoptr->name, __FILE__, __LINE__);
        FerMem_Free(infoptr, __FILE__, __LINE__);
        fclose(symfile);
        return NULL;
    }
    infoptr->ptsy = &(infoptr->ptsx[numpts]);

    strncpy(infoptr->name, symname, symnamelen);
    infoptr->name[symnamelen] = '\0';
    infoptr->namelen = symnamelen;

    /* now carefully read the coordinates */
    infoptr->numpts = 0;
    infoptr->fill = 0;
    while ( fgets(dataline, MAXLINELEN, symfile) != NULL ) {
        /* skip comments and blank lines */
        strptr = dataline;
        while ( isspace(*strptr) )
            strptr++;
        if ( (*strptr == '!') || (*strptr == '\0') )
            continue;
        if ( strncasecmp(strptr, "fill", 4) == 0 ) {
            infoptr->fill = 1;
            continue;
        }
        if ( sscanf(strptr, "%f %f", &(infoptr->ptsx[infoptr->numpts]), &(infoptr->ptsy[infoptr->numpts])) != 2 ) {
            FerMem_Free(infoptr->ptsx, __FILE__, __LINE__);
            FerMem_Free(infoptr->name, __FILE__, __LINE__);
            FerMem_Free(infoptr, __FILE__, __LINE__);
            fclose(symfile);
            return NULL;
        }
        (infoptr->numpts)++;
    }
    fclose(symfile);

    return infoptr;
}

/* Filter function for getting only symbol files from a directory */
static int symbolNameFilter(const struct dirent *direntry) 
{
     int namelen = strlen(direntry->d_name);
     /* Need a name before the filename extension */
     if ( namelen <= SYMFILEEXTLEN )
         return 0;
     if ( strcmp(&(direntry->d_name[namelen-SYMFILEEXTLEN]), SYMFILEEXT) != 0 )
         return 0;
     return 1;
}

/*
 * Free all memory in SymbolInfoList, the static list of information 
 * for all PyFerret named symbols.
 */
void FORTRAN(fgd_delete_all_symboldefs)(void) 
{
    SymbolInfo *nextptr;
    SymbolInfo *infoptr;

    nextptr = SymbolInfoList;
    while ( nextptr != NULL ) {
        infoptr = nextptr;
        nextptr = infoptr->next;
        if ( infoptr->ptsx != NULL )
            FerMem_Free(infoptr->ptsx, __FILE__, __LINE__);
        FerMem_Free(infoptr->name, __FILE__, __LINE__);
        FerMem_Free(infoptr, __FILE__, __LINE__);
    }
    SymbolInfoList = NULL;
}

/*
 * Reads all PyFerret named symbols in the directories given by the environment 
 * variables SYMDIRS_ENVVAR.  The symbols definitions are stored SymbolInfoList.
 * If an error occurs, grdelerrmsg is assigned and zero is returned in status.
 * If successful, non-zero (FERR_OK) is returned in status.
 */
void FORTRAN(fgd_read_all_symboldefs)(int *status)
{
    const char     *envval;
    char            symdirs[MAXLINELEN];
    char           *currdir;
    struct dirent **namelist;
    int             numfiles;
    char            filename[MAXLINELEN];
    SymbolInfo     *infoptr;
    SymbolInfo     *nextptr;

    /* Free any previous list */
    FORTRAN(fgd_delete_all_symboldefs)();

    /* get the directories to search from the environment variable */
    envval = getenv(SYMDIRS_ENVVAR);
    if ( envval == NULL ) {
        sprintf(grdelerrmsg, "Environment variable for markers %s is not defined", SYMDIRS_ENVVAR);
        *status = 0;
        return;
    }

    /* make a copy that can be messed with */
    if ( strlen(envval) >= MAXLINELEN ) {
        sprintf(grdelerrmsg, "Value of environment variable for markers %s exceeds %d characters", SYMDIRS_ENVVAR, MAXLINELEN);
        *status = 0;
        return;
    }
    strcpy(symdirs, envval);

    /* read the symbol files in each (space-separated) directory specified */
    currdir = strtok(symdirs, " \t\v\r\n");
    while ( currdir != NULL ) {
        namelist = NULL;
        numfiles = scandir(currdir, &namelist, symbolNameFilter, alphasort);
        /* ignore any errors; eg, directory named does not exist */
        while ( numfiles > 0 ) {
            numfiles--;
            if ( snprintf(filename, MAXLINELEN, "%s/%s", currdir, namelist[numfiles]->d_name) >= MAXLINELEN ) {
                /* filename too long - skip */
                free(namelist[numfiles]);
                continue;
            }
            infoptr = readSymbolDefFile(filename, namelist[numfiles]->d_name, strlen(namelist[numfiles]->d_name) - SYMFILEEXTLEN);
            if ( infoptr == NULL ) {
                /* problems reading the definition - skip */
                free(namelist[numfiles]);
                continue;
            }
            /* put into the list in alphabetical order */
            if ( (SymbolInfoList == NULL) || (strcmp(infoptr->name, SymbolInfoList->name) < 0) ) {
                /* first entry */
                infoptr->next = SymbolInfoList;
                SymbolInfoList = infoptr;
            }
            else {
                nextptr = SymbolInfoList;
                while ( (nextptr->next != NULL) && (strcmp(infoptr->name, nextptr->next->name) >= 0) )
                    nextptr = nextptr->next;
                infoptr->next = nextptr->next;
                nextptr->next = infoptr;
            }
            free(namelist[numfiles]);
        }
        if ( namelist != NULL )
            free(namelist);
        currdir = strtok(NULL, " \t\v\r\n");
    }

    *status = FERR_OK;
    return;
}

/*
 * Search the directories given by the environment variables SYMDIRS_ENVVAR 
 * for the definition of the given symbol name.  If found, the symbol 
 * information is added to SymbolInfoList and returned.
 */
static SymbolInfo *readSymbolDef(char symname[], int symnamelen) 
{
    const char     *envval;
    char            symdirs[MAXLINELEN];
    char           *currdir;
    SymbolInfo     *infoptr;
    char            filename[MAXLINELEN];
    SymbolInfo     *nextptr;

    /* get the directories to search from the environment variable */
    envval = getenv(SYMDIRS_ENVVAR);
    if ( envval == NULL ) {
        return NULL;
    }

    /* make a copy that can be messed with */
    if ( strlen(envval) >= MAXLINELEN ) {
        return NULL;
    }
    strcpy(symdirs, envval);

    /* search for the symbol file in each (space-separated) directory specified */
    currdir = strtok(symdirs, " \t\v\r\n");
    infoptr = NULL;
    while ( currdir != NULL ) {
        if ( snprintf(filename, MAXLINELEN, "%s/%.*s%s", currdir, symnamelen, symname, SYMFILEEXT) < MAXLINELEN ) {
            infoptr = readSymbolDefFile(filename, symname, symnamelen);
            if ( infoptr != NULL ) {
                break;
            }
        }
        currdir = strtok(NULL, " \t\v\r\n");
    }
    if ( infoptr == NULL ) {
        return NULL;
    }

    /* add to the list in alphabetical order */
    if ( (SymbolInfoList == NULL) || (strcmp(infoptr->name, SymbolInfoList->name) < 0) ) {
        /* first entry */
        infoptr->next = SymbolInfoList;
        SymbolInfoList = infoptr;
    }
    else {
        nextptr = SymbolInfoList;
        while ( (nextptr->next != NULL) && (strcmp(infoptr->name, nextptr->next->name) >= 0) )
            nextptr = nextptr->next;
        infoptr->next = nextptr->next;
        nextptr->next = infoptr;
    }

    return infoptr;
}

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
    SymbolInfo *infoptr;
    int minlen;

    /* Check if the symbol definition has already been read */
    infoptr = SymbolInfoList;
    while ( infoptr != NULL ) {
        if ( (infoptr->namelen == namelen) && (strncmp(infoptr->name, symbolname, namelen) == 0) )
            break;
        /* list is in alphabetical order - check if already passed any possible match */
        minlen = (infoptr->namelen < namelen) ? infoptr->namelen : namelen;
        if ( strncmp(infoptr->name, symbolname, minlen) > 0 )
            infoptr = NULL;
        else
            infoptr = infoptr->next;
    }

    /* If not found, check is there is a new definition file */
    if ( infoptr == NULL )
        infoptr = readSymbolDef(symbolname, namelen);

    if ( infoptr == NULL ) {
        sprintf(grdelerrmsg, "unknown symbol %.*s", namelen, symbolname);
        *ptsxptr = NULL;
        *ptsyptr = NULL;
        *numptsptr = 0;
        *fillptr = 0;
        return 0;
    }

    /* successful return */
    *ptsxptr = infoptr->ptsx;
    *ptsyptr = infoptr->ptsy;
    *numptsptr = infoptr->numpts;
    *fillptr = infoptr->fill;
    return 1;
}

