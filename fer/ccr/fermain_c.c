/*
*
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
*
*/



/*	PROGRAM FERRET - C version of MAIN program with reconfigurable memory */

/* compile this with:
   cc -c [-g] -Iferret_cmn fermain_c.c
         (and use -D_NO_PROTO for non-ANSI compilers)
*/

/*
* TMAP interactive data analysis program

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*/

/*
* FERRET program history:
* initially tailored to output format and content of the Philander/Seigel 
* model from GFDL
* revision 0.0  - 4/3/86
* revision 1.0 - 11/17/86 - first "official" release
* revision 2.0 - 10/23/87 - "official" release
* revision 2.01(temporary) - 10/23/87 - smaller memory size, bug fixes,
*			2 typos in XVARIABLES_DATA, ^C added, ZT planes added
* revision 2.02 - ?????
* revision 2.10 - 5/6/88 - "final" release - /NODEBUG version
* FERRET 1.00     - 6/10/88 - rename of GFDL 2.10
* FERRET 1.10     -  8/2/88 - numerous bug fixes and enhancements
* FERRET 1.20     - 2/17/89 - numerous bug fixes and enhancements
* FERRET 1.21     - 4/19/89 - minor bug fixes
* FERRET 2.00	  - 5/??/89 - internal re-write: 4D grids and "object oriented"
*			      transformations
* FERRET 3.00     - 1/29/93 - revision 2.2-->2.3 changes became so extensive
*                             and prolonged it made sense to rename to V3.0
* FERRET 3.10     - 4/94 - official release using XGKS
* FERRET 3.11     - 5/94 - added FILE/ORDER=/FORMAT=STREAM
* FERRET 3.12     - 5/94 - restructured to be "dynamic memory" (C main routine)
*			   former MAIN became FERRET_DISPATCH routine
* FERRET 3.13     - 7/94 - relink of Solaris version
*                          (using IBM-portable TMAP libs)
*|*|*|*|*|*|*|*|*|*|*|*|
*
*/

/*
   revision history for MAIN program unit:
      11/16/94 - changed to a c version of ferret_dispatch_c
* FERRET 4.0     - 7/94 - using a C main program with dynamic memory
*                - 6/95 - *kob* had to add ifdef checks for 
*		  	  NO_ENTRY_NAME_UNDERSCORES for hp port, as hp
*			  doesnt need trailing underscores for c/fortran 
*			  compatibility
*    3/5/97 *sh* changes to incorporate "-batch" qualifier
* Linux Port 5/97 *kob*   Using NAG f90 for linux, we first have to 
*                         call f90_io_init() to set up lun's etc, and then
*                         call f90_io_finish() after we are done 
*                         to flush buffers,etc
*    7/25/97 *js* changes to incorporate output file for -batch
*    8/97 *kob* - had to add another ifdef check for entry_name_underscores
*              around call to curv_coord_sub
*
*    10/16/97 *kob* - Combining non-gui main program w/ gui main program so that
*                     there needs only be one main program.  added an ifdef 
*                     LINK_GUI_AS_MAIN around the gui-exclusive code
*    10.28.98 *js* Added -secure option
*     3.12.99 *js* Moved GUI code into new module. Also, cleaned up
*                   numerous LINK_GUI_AS_MAIN ifdefs. Also added new gui_get_memory
*                   function since the GUI uses a global memory variable ptr and
*                   the non-GUI uses a stack ptr that was #ifdef'd.
*     9/18/01 *acm* Add ppl memory buffer ppl_memory, defined in ferret_shared_buffer.h
*                  along with Ferret *memory.  Initialize its size, and call
*                  save_ppl_memory_size so the size is available via common to 
*                  Fortran routines.  New declaration of save_ppl_memory_size 
*                  in ferret.h
*    10/19/01 *kob* fix output formatting bug which was printing memory size
*                   (in Mwords) divided by float 1.E6 as a decimal, rather than
*                   a float value - changed in three places
*     8/22/03 *acm* New -script command-line switch: run the named script and exit.
*                   This mode also sets -nojnl and -server, as well as the new
*                   -noverify switch.  It also supresses the banner lines.
*/

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "ferret.h"
#include "ferret_shared_buffer.h"

void gui_run(int *, char **);
int gui_init();
int its_script;
char script_args[2048];
int arg_pos;
float **gui_get_memory();
static void command_line_run(float **memory);

void help_text()
{
  printf(
	 "Usage:  ferret [-memsize Mwords] [-batch [outfile]] [-server] [-secure] [-gif] [-gui] [-unmapped] [-help] [-nojnl] [-noverify] [-script [args]]\n\
-memsize:  specify the memory cache size in megawords (default 3.2)\n\
-batch:  output directly to metafile \"outfile\" w/out X windows\n\
-unmapped:  use invisible output windows (superceded by -batch)\n\
-gif:  output to GIF file w/o X windows only w/ FRAME command\n\
-gui:  to start Ferret in point and click mode (not available on all platforms)\n\
-secure:  run securely -- don't allow system commands\n\
-server:  run in server mode -- don't stop on message commands\n\
-help:  obtain this listing\n\
-nojnl:  on startup don't open a journal file (can be turned on later with SET MODE JOURNAL\n\
-noverify:  on startup turn off verify mode (can be turned on later with SET MODE VERIFY\n\
-script scriptname [arguments]: execute the specified script and exit: SPECIFY THIS LAST\n");
  exit(0);
}


/*
 * Eliminated _NO_PROTO ifdef (are there still non-ANSI C compilers around?)
 * JS
 */

static int ttout_lun=TTOUT_LUN,
  max_mem_blks=PMAX_MEM_BLKS,
  mem_blk_size,
  old_mem_blk_size,
  pmemsize,
  mem_size = PMEM_BLK_SIZE * PMAX_MEM_BLKS;


main (int oargc, char *oargv[])
{
  int status;
  smPtr sBuffer;
  float **memory = gui_get_memory(); 
  int argc = oargc;
  char **argv = oargv;

  int i=1;
  int j=1;
  float rmem_size;
  int using_gui = 0;
  int pplmem_size;

  int gui_enabled = gui_init();
  int journalfile = 1;
  int verify_flag = 1;
  int len_str;

  its_script = 0;
  arg_pos = 0;

#ifdef MIXING_NAG_F90_AND_C
  f90_io_init();
#endif

#ifdef __CYGWIN__
  for_rtl_init_(&argc, argv);
#endif

  /* decode the command line options: "-memsize", and "-unmapped" */
  while (i<argc) {
    if (strcmp(argv[i],"-memsize")==0){
      if (++i==argc) help_text();
      if ( sscanf(argv[i++],"%f",&rmem_size) != 1 ) help_text();
      if ( rmem_size <= 0.0 ) help_text();
      mem_size = rmem_size * 1.E6;
    } else if (strcmp(argv[i],"-unmapped")==0) {
      WindowMapping(0);  /* new routine added to xopws.c */
      i++;    /* advance to next argument */
    } else if (strcmp(argv[i],"-gif")==0) {
      char *meta_name = ".gif";	/* Unused dummy name */
      FORTRAN(set_batch_graphics)(meta_name);  /* inhibit X output altogether */
      ++i;
    } else if (strcmp(argv[i],"-secure")==0) {
      set_secure();
      ++i;
    } else if (strcmp(argv[i],"-server")==0) {
      set_server();
      ++i;
    } else if (strcmp(argv[i],"-nojnl")==0) {
      journalfile = 0;
	  /*FORTRAN(set_start_jnl_file)(journalfile);*/ 
      ++i;
	  
    } else if (strcmp(argv[i],"-batch")==0) {
      char *meta_name = "metafile.plt";
      if (++i < argc && argv[i][0] != '-'){
	  meta_name = argv[i++];
      }
      FORTRAN(set_batch_graphics)(meta_name);  /* inhibit X output altogether*/
	  
    } else if (strcmp(argv[i],"-noverify")==0) {
      verify_flag = 0;    
	  ++i;
      
    } else if (strcmp(argv[i],"-gui")==0) {
      i++;
      if (gui_enabled){
	using_gui = 1;  /* new routine added to xopws.c */
      } else {
	fprintf(stderr,
		"Warning: the -gui flag has no effect on this platform\n");
      }
    /* -script mode implies -server and -nojnl */
    } else if (strcmp(argv[i],"-script")==0)  {
      char *script_name = "noscript";
	  set_server();
	  journalfile = 0;
	  verify_flag = 0;
	  strcpy( script_args, " " );
	  if (++i < argc && argv[i][0] != '-'){
		script_name = argv[i++];
		its_script = 1;
		len_str = strlen(script_name);
		FORTRAN(save_scriptfile_name)(script_name, &len_str, &its_script);
		if (its_script != 1) {
			help_text();
	     }
		 arg_pos = 0;
		 while (i < argc && argv[i][0] != '-'){
			len_str = strlen(argv[i]);
			j = 0;
			while (j<len_str) {
				script_args[arg_pos++] = argv[i][j];
				j++;
			}
			if (i+1 < argc && argv[i+1][0] != '-') {
				strcat(script_args, ",");
				arg_pos = arg_pos+1;
		   }
		   i = i+1;
		 }
		 if (strcmp(script_name,"noscript")==0 ) help_text();
	  } 

	  } else  /* -help also comes here */
      help_text();
  }

  /* initial allocation of memory space */
  mem_blk_size =  mem_size / max_mem_blks;
  *memory = (float *) malloc(mem_size*sizeof(float));
  if ( *memory == (float *)0 ) {
    printf("Unable to allocate the requested %f Mwords of memory.\n",mem_size/1.E6);
    exit(0);
  }
 
  /* initial allocation of PPLUS memory size pointer*/
  pplmem_size = 0.5* 1.E6;  
  FORTRAN(save_ppl_memory_size)( &pplmem_size ); 
  ppl_memory = (float *) malloc(sizeof(float) * pplmem_size );

  if ( ppl_memory == (float *)0 ) {
    printf("Unable to allocate the initial %f words of PLOT memory.\n",pplmem_size);
    exit(0);
 }
  /* initialize stuff: keyboard, todays date, grids, GFDL terms, PPL brain */
  FORTRAN(initialize)();

  /*  prepare appropriate console input state and open the output journal file */

  if ( journalfile ) {
    FORTRAN(init_journal)( &status );
  } else {
    FORTRAN(no_journal)();
  }

  if ( verify_flag ==0) {
    FORTRAN(turnoff_verify)( &status );
  }


  /* initialize size and shape of memory and linked lists */
  FORTRAN(init_memory)( &mem_blk_size, &max_mem_blks );

  if ( using_gui ) {
    gui_run(&argc, argv);
  } else {
    command_line_run(memory);
  }
  /* 
   *kob* 5/97 - need to close f90 files and flush buffers.....
   */
#ifdef MIXING_NAG_F90_AND_C
  f90_io_finish();
#endif
#ifdef __CYGWIN__
  for_rtl_finish_(&argc, argv);
#endif
}

static void command_line_run(float **memory){
  FILE *fp = 0;
  char init_command[2176], script_file[2048], *home = getenv("HOME");
  int ipath = 0;
  int len_str = 0;
  int j = 0;


  /* turn on ^C interrupts  */
  /* 10/97 *kob* add check for gui now that there is only one main program */
  FORTRAN(set_ctrl_c)( FORTRAN(ctrlc_ast) );

  /* program name and revision number */
  /* 10/97 *kob* add check for gui now that there is only one main program */
  if (its_script==0) FORTRAN(proclaim_c)( &ttout_lun, "\t" );

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
  
    if (its_script)
    {
	FORTRAN(get_scriptfile_name)(script_file, &ipath);
      if ( ipath ) {
	  strcat( init_command, "; GO \"" ); 
	  strcat( init_command, script_file );
	  strcat( init_command, "\ "); 
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

  /* allocate the shared buffer */
  sBuffer = (sharedMem *)malloc(sizeof(sharedMem));

  /* run the initialization file
     * Note 1: do not pass a fixed string as the command - FERRET needs to blank it
     * Note 2: if not under GUI control we will not normally exit FERRET_DISPATCH
     *	  until we are ready to exit FERRET */
  while ( 1) {
    /*    ferret_dispatch_( memory, init_command, rtn_buff );  FORTRAN version */
    ferret_dispatch_c( *memory, init_command, sBuffer );

    /* debugging flow control checks */
    /*
      printf("       --> control returned to C MAIN:%d %d %d\n",
      sBuffer->flags[FRTN_CONTROL],
      sBuffer->flags[FRTN_STATUS],
      sBuffer->flags[FRTN_ACTION]);
    */

    /* ***** REALLOCATE MEMORY ***** */
    if (sBuffer->flags[FRTN_ACTION] == FACTN_MEM_RECONFIGURE) {
      old_mem_blk_size = mem_blk_size;
      mem_size = sBuffer->flags[FRTN_IDATA1];
      mem_blk_size = mem_size / max_mem_blks;
      /*
	printf("memory reconfiguration requested: %d\n",mem_size);
	printf("new mem_blk_size = %d\n",mem_blk_size);
      */
      free ( (void *) *memory );
      *memory = (float *) malloc(mem_size*sizeof(float));
      if ( *memory == 0 ) {
	printf("Unable to allocate %f Mwords of memory.\n",mem_size/1.E6 );
	mem_blk_size = old_mem_blk_size;
	mem_size = mem_blk_size * max_mem_blks;
	*memory = (float *) malloc(mem_size*sizeof(float));
	if ( *memory == (float *)0 ) {
	  printf("Unable to reallocate previous memory of %f Mwords.\n",mem_size/1.E6 );
	  exit(0);
	} else {
	  printf("Restoring previous memory of %f Mwords.\n",mem_size/1.E6 );
	}
      }
      FORTRAN(init_memory)( &mem_blk_size, &max_mem_blks );
    }

    /* ***** EXIT ***** */
    else if  (sBuffer->flags[FRTN_ACTION] == FACTN_EXIT ) {
      /*      printf("exit from FERRET requested\n"); */
      exit(0);
    }

    /* ***** TEMPORARY RETURN IN CASE MAIN NEEDS TO DISPLAY FERRET MSG ***** */
    else {

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
