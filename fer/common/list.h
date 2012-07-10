/* list.h -- data structures and such for generic list package
 * 
 * Last edited: Tue Jul 28 15:29:56 1992 by bcs (Bradley C. Spatz) on wasp
 *
 * Copyright (C) 1992, Bradley C. Spatz, bcs@ufl.edu
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 */

/* Define a structure to describe the list. */
struct list_t {
   int size;
   struct list_element_t *front;
   struct list_element_t *rear;
   struct list_element_t *curr;
};

/* Define a structure to describe each element in the list. */
struct list_element_t {
   struct list_element_t *prev;
   struct list_element_t *next;
   char *data;
};

/* Structs are ugly, so... */
typedef struct list_t LIST;
typedef struct list_element_t LIST_ELEMENT;

/* Define some constants for controlling list traversals.  We
 * bit-code the attributes so they can be OR'd together.
 */
#define LIST_FORW	0
#define LIST_BACK	2
#define LIST_FRNT	4
#define LIST_CURR	8
#define LIST_REAR	18
/* 16 + 2, since REAR implies BACKwards. */
#define LIST_SAVE	32
#define LIST_ALTR	64

/* Define some constants for return codes and such. */
#ifndef TRUE
#define TRUE  1
#endif
#ifndef FALSE
#define FALSE 0
#endif
/* LIST_DEALLOC and LIST_NODEALLOC are passed as pointer-to-function "standard" values */
#define LIST_DEALLOC   ((void (*)(char *))(-1))
#define LIST_NODEALLOC ((void (*)(char *))(-2))
#define LIST_EMPTY      0
#define LIST_OK         1
#define LIST_EXTENT     2

/* prototypes */
LIST *list_init(void);
LIST *list_mvprev(LIST *list);
LIST *list_mvnext(LIST *list);
char *list_insert_before(LIST *list, char *data, int bytes);
char *list_insert_after(LIST *list, char *data, int bytes);
char *list_remove_front(LIST *list);
char *list_remove_rear(LIST *list);
char *list_remove_curr(LIST *list);
int   list_traverse(LIST *list, char *data, int (*func)(char *, char *), int opts);
void  list_free(LIST *list, void (*dealloc)(char *));

/* Yet more prototypes - previously were possibly macros */
LIST *list_mvfront(LIST *list);
LIST *list_mvrear(LIST *list);
int   list_empty(LIST *list);
char *list_front(LIST *list);
char *list_curr(LIST *list);
char *list_rear(LIST *list);
int   list_size(LIST *list);



