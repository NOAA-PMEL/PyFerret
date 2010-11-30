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
 *  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY
 *  SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
 *  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
 *  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
 *  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include "ferret_lib.h"
#include <wchar.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include "ferret.h"
#include "ferret_shared_buffer.h"

int its_script;
char script_args[2048];
int arg_pos;
static void command_line_run(float **mymemory);

void help_text()
{
  printf("\n"
         "Usage:  ferret [-memsize Mwords] [-batch [outfile]] [-server] [-secure] [-gif] [-help] [-nojnl] [-noverify] [-script [args]]\n"
         "-memsize:  specify the memory cache size in megawords (default 3.2)\n"
         "-batch:  output directly to metafile \"outfile\" w/out X windows\n"
         "-gif:  output to GIF file w/o X windows only w/ FRAME command\n"
         "-secure:  run securely -- don't allow system commands\n"
         "-server:  run in server mode -- don't stop on message commands\n"
         "-help:  obtain this listing\n"
         "-nojnl:  on startup don't open a journal file (can be turned on later with SET MODE JOURNAL\n"
         "-noverify:  on startup turn off verify mode (can be turned on later with SET MODE VERIFY\n"
         "-script scriptname [arguments]: execute the specified script and exit: SPECIFY THIS LAST\n"
         "-version lists the version number of Ferret and stops. \n"
         "\n");
  exit(0);
}

/*
 * Signal handler for SIGILL, SIGFPE, and SIGSEGV
 * (ie, just program-crashing signals, not user-generated signals)
 * for generating a stderr message for LAS and exiting gracefully.
 */
static void fer_signal_handler(int signal_num)
{
  fprintf(stderr, "**ERROR Ferret crash; signal = %d\n", signal_num);
  fflush(stderr);
  exit(-1);
}


static int ttout_lun=TTOUT_LUN,
           max_mem_blks=PMAX_MEM_BLKS,
           mem_blk_size,
           old_mem_blk_size,
           pmemsize,
           fermem_size = PMEM_BLK_SIZE * PMAX_MEM_BLKS;

int main(int oargc, char *oargv[])
{
  int status;
  float *fermem;
  float *pplmem;
  int argc = oargc;
  char **argv = oargv;

  int i=1;
  int j=1;
  float rmem_size;
  int pplmem_size;

  int journalfile = 1;
  int verify_flag = 1;
  int len_str;
  int uvar_dset;

  its_script = 0;
  arg_pos = 0;

#ifdef MIXING_NAG_F90_AND_C
  f90_io_init();
#endif

#ifdef __CYGWIN__
  for_rtl_init_(&argc, argv);
#endif

  /* Catch SIGILL, SIGFPE, and SIGSEGV */
  if ( (signal(SIGILL, fer_signal_handler) == SIG_ERR) ||
       (signal(SIGFPE, fer_signal_handler) == SIG_ERR) ||
       (signal(SIGSEGV, fer_signal_handler) == SIG_ERR) ) {
     perror("**ERROR Unable to catch SIGILL, SIGFPE, or SIGSEGV");
     exit(1);
  }


  /* decode the command line options: "-memsize", and "-unmapped" */
  rmem_size = fermem_size/1.E6;
  while (i<argc) {
    if (strcmp(argv[i],"-version")==0){
      version_only_();
      exit(0);
    } else if (strcmp(argv[i],"-memsize")==0){
      if (++i==argc) help_text();
      if ( sscanf(argv[i++],"%f",&rmem_size) != 1 ) help_text();
      if ( rmem_size <= 0.0 ) help_text();
      fermem_size = rmem_size * 1.E6;
    } else if (strcmp(argv[i],"-gif")==0) {
      char *meta_name = ".gif";        /* Unused dummy name */
      set_batch_graphics_(meta_name);  /* inhibit X output altogether */
      ++i;
    } else if (strcmp(argv[i],"-secure")==0) {
      set_secure();
      ++i;
    } else if (strcmp(argv[i],"-server")==0) {
      set_server();
      ++i;
    } else if (strcmp(argv[i],"-nojnl")==0) {
      journalfile = 0;
      ++i;

    } else if (strcmp(argv[i],"-batch")==0) {
      char *meta_name = "metafile.plt";
      if (++i < argc && argv[i][0] != '-'){
          meta_name = argv[i++];
      }
      set_batch_graphics_(meta_name);  /* inhibit X output altogether*/

    } else if (strcmp(argv[i],"-noverify")==0) {
      verify_flag = 0;
          ++i;
    /* -script mode implies -server and -nojnl */
    } else if (strcmp(argv[i],"-script")==0)  {

      char *script_name = "noscript";
      set_server();
      journalfile = 0;
      verify_flag = 0;

      /*//<YWEI> this is added because sometime string arguments are not
      //parsed correctly by the system*/
      if (++i < argc){
        script_name = argv[i];

        its_script = 1;
        len_str = strlen(script_name);
        for (j = 0; j < len_str; j++) {
           if (argv[i][j] == ' ') {
             argv[i][j] = '\0';
             j++;
             break;
           }
        }

        arg_pos = 0;
        if (j < len_str) {
          while (j < len_str) {
            script_args[arg_pos++] = argv[i][j++];
          }
          if (i+1 < argc) {
            script_args[arg_pos++] = ' ';
          }
        }

        len_str = strlen(script_name);
        save_scriptfile_name_(script_name, &len_str, &its_script);
        if ( its_script != 1 || strcmp(script_name,"noscript") == 0 ) {
          help_text();
        }

        i++;
        while ( i < argc ) {
          len_str = strlen(argv[i]);
          j = 0;
          while ( j < len_str ) {
            script_args[arg_pos++] = argv[i][j++];
          }

          if (i+1 < argc) {
            script_args[arg_pos++] = ' ';
          }
          i++;
        }

        /* //</YWEI> */
        script_args[arg_pos]='\0';

      }

    } else { /* -help also comes here */
      help_text();
    }
  }

  /* initial allocation of memory space */
  mem_blk_size =  fermem_size / max_mem_blks;
  fermem = (float *) malloc(fermem_size*sizeof(float));
  if ( fermem == (float *)0 ) {
    printf("Unable to allocate the requested %f Mwords of memory.\n",fermem_size/1.E6);
    exit(0);
  }
  if (mem_blk_size < 0)
  { printf("internal overflow expressing %g Mwords as words %d \n",rmem_size,fermem_size);
    printf("Unable to allocate the requested %g Mwords of memory.\n",rmem_size);
    exit(0);
  }
  /* initialize size and shape of memory and linked lists */
  set_fer_memory(fermem, fermem_size);


  /* initial allocation of PPLUS memory size pointer*/
  pplmem_size = 0.5* 1.E6;
  pplmem = (float *) malloc(sizeof(float) * pplmem_size );
  if ( pplmem == (float *)0 ) {
    printf("Unable to allocate the initial %d words of PLOT memory.\n",pplmem_size);
    exit(0);
  }
  set_ppl_memory(pplmem, pplmem_size);

  /* initialize the shared buffer */
  set_shared_buffer();

  /* initialize stuff: keyboard, todays date, grids, GFDL terms, PPL brain */
  initialize_();

  /*  prepare appropriate console input state and open the output journal file */

  if ( journalfile ) {
    init_journal_( &status );
  } else {
    no_journal_();
  }

  if ( verify_flag ==0) {
    turnoff_verify_( &status );
  }

  command_line_run(&fermem);
  /*
   *kob* 5/97 - need to close f90 files and flush buffers.....
   */

#ifdef MIXING_NAG_F90_AND_C
  f90_io_finish();
#endif
#ifdef __CYGWIN__
  for_rtl_finish_(&argc, argv);
#endif

  return 0;
}

static void command_line_run(float **mymemory){
  FILE *fp = 0;
  char init_command[2176], script_file[2048], *home = getenv("HOME");
  int ipath = 0;
  int len_str = 0;
  int j = 0;
  int script_resetmem = 0;

  /* turn on ^C interrupts  */
  set_ctrl_c_( ctrlc_ast_ );

  /* program name and revision number */
  if (its_script==0) proclaim_c_( &ttout_lun, "\t" );

  /* set up to execute $HOME/.ferret if it exists: '\GO "$HOME/.ferret"' */
  /* --> need to see if it exists!!! */
  if (home != NULL ) {
    strcpy( init_command, home );
    strcat( init_command, "/.ferret" );
    fp = fopen( init_command, "r" );
    if ( fp == NULL )
      strcpy( init_command, " " );
    else {
      strcpy( init_command, "GO \"$HOME/.ferret\"" );
      fclose( fp ); }     /* moved close inside brackets - can't close a null fp *kob* */
  } else {
    strcpy( init_command, " " );
  }

  /* If Ferret was started with the -script switch, execute the script with its args. */

    if (its_script) {
        get_scriptfile_name_(script_file, &ipath);
      if ( ipath ) {
          strcat( init_command, "; GO \"" );
          strcat( init_command, script_file );
          strcat( init_command, "\"" );
          strcat( init_command, " ");
      } else {
          strcat( init_command, "; GO " );
          strcat( init_command, script_file );
          strcat( init_command, " ");
      }
          if (arg_pos !=0) {
               len_str = strlen(init_command);
                   strcat( init_command, script_args );
          }
          strcat( init_command, "; EXIT/PROGRAM");
    }


  /* run the initialization file
     * Note 1: do not pass a fixed string as the command - FERRET needs to blank it
     * Note 2: if not under GUI control we will not normally exit FERRET_DISPATCH
     *          until we are ready to exit FERRET
     * Note 3: in -script mode, a SET MEM command executes commands in this
              routine to reset memory.  Once this has been done, call ferret_dispatch_c
                  with blank second argument to continue executing commands in the script.
                  Reset the flag script_resetmem so that further SET MEM commands are executed
                  as well.*/
  while ( 1) {
    if ( script_resetmem != 0 ) {
      ferret_dispatch_c(*mymemory, " ", sBuffer);
      script_resetmem = 0;
    }
    else {
      ferret_dispatch_c(*mymemory, init_command, sBuffer);
    }

    /* ***** REALLOCATE MEMORY ***** */
    if (sBuffer->flags[FRTN_ACTION] == FACTN_MEM_RECONFIGURE) {
      old_mem_blk_size = mem_blk_size;
      fermem_size = sBuffer->flags[FRTN_IDATA1];
      mem_blk_size = fermem_size / max_mem_blks;
      free ( (void *) *mymemory );
      *mymemory = (float *) malloc(fermem_size*sizeof(float));
      if ( *mymemory == 0 ) {
        printf("Unable to allocate %f Mwords of memory.\n",fermem_size/1.E6 );
        mem_blk_size = old_mem_blk_size;
        fermem_size = mem_blk_size * max_mem_blks;
        *mymemory = (float *) malloc(fermem_size*sizeof(float));
        if ( *mymemory == (float *)0 ) {
          printf("Unable to reallocate previous memory of %f Mwords.\n",fermem_size/1.E6 );
          exit(0);
        } else {
          printf("Restoring previous memory of %f Mwords.\n",fermem_size/1.E6 );
        }
      }
      set_fer_memory(*mymemory, fermem_size);
      script_resetmem = 1;  /* Likewise if the SET MEM is in the init script */
    }

    /* ***** EXIT ***** */
    else if  (sBuffer->flags[FRTN_ACTION] == FACTN_EXIT ) {
      finalize_();
      exit(0);
    }

    /* ***** TEMPORARY RETURN IN CASE MAIN NEEDS TO DISPLAY FERRET MSG ***** */
    else if ( sBuffer->flags[FRTN_ACTION] != FACTN_NO_ACTION ) {

      /*
        check the sBuffer->flags[FRTN_STATUS] to see if you need
        to display any messages from FERRET to the user
      */
      if (sBuffer->flags[FRTN_STATUS] != 3 ) {
        printf("error buffer from FERRET: %d\n", sBuffer->numStrings);
        printf("error lines:\n%s\n",sBuffer->text);
      }
      else
        printf("no action - returning from GUI to FERRET\n");

    }
  }
}
