/*
 * JC_CommandGen.h
 *
 * Jonathan Callahan
 * Dec 11th 1995
 *
 * This file contains function declarations for the methods which create
 * command string for shipment to Ferret.
 *
 * I have endeavored to stay very close to true Object-Oriented-Programming
 * principles and have in general followed the C++ guidelines as I learned
 * them from "Weiskamp & Flamig:  The complete C++ primer".
 *
 */


/* .................... JC_PlotCommand methods .................... */

#ifndef _JC_COMMANDGEN_H
#define _JC_COMMANDGEN_H

#include "ferret_structures.h"

extern struct customSaveOptions;

extern void JC_Clone_Print( JC_Object *this, FILE *File_ptr );


void JC_SolidLandCommand_Create(char *command, JC_Object *O_ptr, int resolution);

void JC_ListCommand_Create(char *command, JC_Object *O_ptr );

/*
 * Input     command:      text string which will contain the created command
 *           O_ptr:        pointer to a JC_Object containing the current context
 *    
 * Output    command:      this string will contain a Ferret command to create a list for
 *                         the variable and region in the JC_Object 
 */

void JC_ListFileCommand_Create( char *command, JC_Object *O_ptr, struct customSaveOptions *CSO_ptr );

/*
 * Input     command:      text string which will contain the created command
 *           O_ptr:        pointer to a JC_Object containing the current context
 *           CSO_ptr:      pointer to a customSaveOptions which has various options
 *    
 * Output    command:      this string will contain a Ferret command to create a list for
 *                         the variable and region in the JC_Object and write it to a file
 */

void JC_LetCommand_Create(char *command, JC_DefinedVariable *DV_ptr );

/*
 * Input     command:      text string which will contain the created command
 *           DV_ptr:       pointer to a JC_DefinedVariable
 *    
 * Output    command:      this string will contain a Ferret command to create a "LET"
 *                         command which defines a new variable as described by the
 *                         current state of the "JC_DefineVariable" interface. 
 */

void JC_PlotCommand_Create(char *command, JC_Object *O_ptr, JC_PlotOptions *PO_ptr);

/*
 * Input     command:      text string which will contain the created command
 *           O_ptr:        pointer to a JC_Object containing the current context
 *           PO_ptr:       pointer to the JC_PlotOptions containing the current plot options
 *    
 * Output    command:      this string will contain a Ferret command to create a plot for
 *                         the JC_Object and JC_PlotOptions currently chosen in the GUI
 */

#endif /* _JC_COMMANDGEN_H */

/* ~~~~~~~~~~~~~~~~~~~~ END OF JC_CommandGen.h ~~~~~~~~~~~~~~~~~~~~ */
