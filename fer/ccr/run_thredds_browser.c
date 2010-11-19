/*
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
 */

#include <wchar.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "run_thredds_browser.h"

/*
 * Interface to the FORTRAN function:
 * INTEGER FUNCTION RUN_THREDDS_BROWSER(DATASET_NAME, ERR_WARN_MSG)
 *         CHARACTER*(*) DATASET_NAME, ERR_WARN_MSG
 * where the return value is the actual length of the name in DATASET_NAME
 * if successful, 0 if canceled, -1 if error.  (See runThreddsBrowser)
 * Assumes DATASET_NAME and ERR_WARN_MSG are blank-filled when passed
 * to this function.  Does not do bounds checking.
 *
 * Assumes Holerith strings are passed as a char array with a max string 
 * length appended to the end of the argument list.  Also assumes the C 
 * name is the single underscore appended to the lower-cased FORTRAN name.
 */
int run_thredds_browser_(char dataset_name[], char err_warn_msg[], int max_len_dataset_name, int max_len_err_warn_msg) {
    int len_dataset_name;

    /* Run the browser */
    len_dataset_name = runThreddsBrowser(dataset_name, err_warn_msg);

    /* Replace the null terminators with blanks */
    dataset_name[strlen(dataset_name)] = ' ';
    err_warn_msg[strlen(err_warn_msg)] = ' ';

    /* Return the return value from runThreddsBrowser */
    return len_dataset_name;
}

/*
 * Verifies access to Java 6 or later (version 1.6+) and runs the ThreddsBrowser Java 
 * application.  Writes the dataset name output from ThreddsBrowser to datasetName and 
 * any warning messages are written to errWarn.  The length of datasetName is returned.
 * If the user cancels out of the ThreddsBrowser, datasetName is empty and zero is 
 * returned.  If an error occurs, datasetName is empty, -1 is returned and an error 
 * message is written to errWarn.
 *
 * Requires the environment variable FER_LIBS and the jar files
 * ${FER_LIBS}/threddsBrowser.jar (from the ThreddsBrowser Java project) and
 * ${FER_LIBS}/toolsUI.jar (from http://www.unidata.ucar.edu/software/netcdf-java/)
 */
int runThreddsBrowser(char datasetName[], char errWarn[]) {
    char *envVal;
    char  javaExeName[FILENAME_MAX];
    int   version;
    char  errMsg[256];
    char  classPath[2*FILENAME_MAX];
    char *argvStack[5];
    FILE *pipeFile;
    pid_t childPid;
    char  output[2*FILENAME_MAX];
    char *strptr;

    datasetName[0] = '\0';
    errWarn[0] = '\0';

    /* Check if JAVA_HOME is defined and reasonable */
    envVal = getenv("JAVA_HOME");
    if ( envVal != NULL ) {
        snprintf(javaExeName, FILENAME_MAX, "%s/bin/java", envVal);
        version = getJavaVersion(javaExeName, errMsg);
        if ( version == -1 ) {
            strcat(errWarn, errMsg);
            strcat(errWarn, "WARNING: Ignoring environment variable JAVA_HOME (invalid path)\n");
            envVal = NULL;
       }
        else if ( version < 6 ) {
            strcat(errWarn, "WARNING: Ignoring environment variable JAVA_HOME (java version too old)\n");
            envVal = NULL;
        }
    }

    /* JAVA_HOME either undefined or unacceptable, so try java without a path */
    if ( envVal == NULL ) {
        strcpy(javaExeName, "java");
        version = getJavaVersion(javaExeName, errMsg);
        if ( version == -1 ) {
            strcat(errWarn, errMsg);
        }
        if ( version < 6 ) {
            strcat(errWarn, "ERROR: unable to find version 6 (or later) of Java\n");
            return -1;
        }
    }

    /* Create the class path for the jar files needed */
    envVal = getenv("FER_LIBS");
    if ( envVal == NULL ) {
        strcat(errWarn, "ERROR: environment variable FER_LIBS is not defined\n");
        return -1;
    }
    snprintf(classPath, 2*FILENAME_MAX, "%s/threddsBrowser.jar:%s/toolsUI.jar", envVal, envVal);

    /* Run the ThreddsBrowser application */
    argvStack[0] = javaExeName;
    argvStack[1] = "-classpath";
    argvStack[2] = classPath;
    argvStack[3] = "gov.noaa.pmel.ferret.threddsBrowser.ThreddsBrowser";
    argvStack[4] = NULL;

    pipeFile = executableOutput(argvStack, &childPid, errMsg);
    if ( pipeFile == NULL ) {
        strcat(errWarn, errMsg);
        return -1;
    }

    /* Get the dataset name */
    while ( fgets(output, 2*FILENAME_MAX, pipeFile) != NULL ) {
        if ( strncmp(output, "USE \"", 5) == 0 ) {
            /* get the contents between the double quotes */
            strptr = strrchr(output, '"');
            if ( strptr > output + 4 ) {
                *strptr = '\0';
                strcpy(datasetName, output + 5);
            }
        }
        else {
           strcat(errWarn, output);
        }
    }

    /* Close the read end of the pipe and reap the child process */
    fclose(pipeFile);
    waitpid(childPid, NULL, 0);

    return strlen(datasetName);
}

/*
 * Return the Java version of the given java executable.  The version is obtained
 * from parsing the output from running the java executable with the "-version"
 * flag.  So a version string of "1.5.0_05" will return 5 and "1.6.0_20" will return 6.
 * If an error occurs, -1 is returned and an error message is written to errMsg.
 */
int getJavaVersion(char javaExeName[], char errMsg[]) {
    char *argvStack[3];
    FILE *pipeFile;
    pid_t childPid;
    char  output[FILENAME_MAX];
    int   versionNum;
    int   major, minor, revision, build;

    /* Run this java executable with the -version flag */
    argvStack[0] = javaExeName;
    argvStack[1] = "-version";
    argvStack[2] = NULL;
    pipeFile = executableOutput(argvStack, &childPid, errMsg);
    if ( pipeFile == NULL ) {
        return -1;
    }

    /* Get the version number string - note that this is printed to stderr */
    versionNum = -1;
    while ( fgets(output, FILENAME_MAX, pipeFile) != NULL ) {
        if ( sscanf(output, "java version \"%d.%d.%d_%d", &major, &minor, &revision, &build) == 4 ) {
            /* if 1.x, the version is the x */
            if ( major == 1 ) {
                versionNum = minor;
            }
        }
    }

    /* Close the read end of the pipe and reap the child process */
    fclose(pipeFile);
    waitpid(childPid, NULL, 0);

    if ( versionNum == -1 )
        strcpy(errMsg, "Unable to interpret the Java version\n");
    return versionNum;
}

/*
 * Forks off a child process that becomes the executable given by exeArgv,
 * which matches the argv array given as the second argument to execvp.
 * The first argument to execvp will be exeArgv[0].  On success, the PID of
 * the child process will be returned in childPidPtr, and a FILE is returned
 * from which the combined stdout and stderr output of the execuable can be
 * read.  On error, NULL is returned, the value in childPidPtr is unchanged,
 * an error message is written to errMsg, and any child process which might
 * have been created has been killed using SIGTERM and reaped.
 */
FILE *executableOutput(char *exeArgv[], pid_t *childPidPtr, char errMsg[]) {
    int   fildes[2];
    pid_t childPid;
    FILE *outFile;

    if ( pipe(fildes) != 0 ) {
        sprintf(errMsg, "Unable to create a pipe: %s\n", strerror(errno));
        return NULL;
    }

    childPid = fork();
    if ( childPid < 0 ) {
        /* error - this is the parent and there is no child */
        sprintf(errMsg, "Unable to fork off a child process: %s\n", strerror(errno));
        close(fildes[0]);
        close(fildes[1]);
        return NULL;
    }

    if ( childPid == 0 ) {
        /* child which never returns - close the read end of the pipe */
        close(fildes[0]);

        /* redirect stdout and stderr to the writing end of the pipe */
        if ( (dup2(fildes[1], 1) < 0) || (dup2(fildes[1], 2) < 0) )
            exit(1);

        /* become the program given in argsStack */
        execvp(exeArgv[0], exeArgv);

        /* if execvp returns, an error has occurred */
        exit(1);
    }

    /* parent - close the write end of the pipe */
    close(fildes[1]);

    /* create a FILE wrapping the reading end of the pipe */
    outFile = fdopen(fildes[0], "r");
    if ( outFile == NULL ) {
        sprintf(errMsg, "Unable to create a FILE from a pipe file descriptor: %s\n", strerror(errno));
        kill(childPid, SIGTERM);
        waitpid(childPid, NULL, 0);
        close(fildes[0]);
        return NULL;
    }

    /* parent records the child pid and returns a FILE wrapping the reading end of the pipe to the child */
    *childPidPtr = childPid;
    return outFile;
}

