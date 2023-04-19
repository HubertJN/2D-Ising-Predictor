/* This file written by C Brady, University of Warwick (2019)
 * The contents of this file are placed into the public domain
 * under the Creative Commons CC0 license
 * https://creativecommons.org/publicdomain/zero/1.0/legalcode
 * As per this license this work is provided as-is with no warranty
 * made of any kind*/

#ifndef KVP_H
#define KVP_H

#define KVP_ERROR_TYPE unsigned char

#define KVP_OK 0
#define KVP_BAD_KEY 1
#define KVP_BAD_VALUE 2
#define KVP_BAD_TYPE 4
#define KVP_BAD_RANGE 8
#define KVP_ERR 255

#ifndef KVP_MAX_KEY
  #define KVP_MAX_KEY 128
#endif

#define KVP_ISWHITESPACE(x) (x==' ' || x=='\t' || x=='\r')
#define KVP_ISSEPARATOR(x) (x=='=')
#define KVP_ISQUOTE(x) (x=='\"')

#include <stddef.h>

/*Struct for registered key*/
typedef struct {
  char name[KVP_MAX_KEY];
	KVP_ERROR_TYPE (*action)(char *key, char *value, void *data);
	void *data;
	unsigned char free;
	size_t size;
} kvp_item;

typedef struct{
  char *string;
	size_t size;
} kvp_held_string;

extern kvp_item *kvp_items;
extern int kvp_itemcount;
extern int kvp_itemslots;

/*Free used memory*/
void kvp_free();

/*Registration functions*/
void kvp_register_function(char *key, KVP_ERROR_TYPE (*action)(char *key, char *value, void *data));
void kvp_register_function_with_data(char *key, KVP_ERROR_TYPE (*action)(char *key, char *value, void *data), void *data);
void kvp_register_c(char *key, char *dest);
void kvp_register_uc(char *key, unsigned char *dest);
void kvp_register_s(char *key, short *dest);
void kvp_register_us(char *key, unsigned short *dest);
void kvp_register_i(char *key, int *dest);
void kvp_register_ui(char *key, unsigned int *dest);
void kvp_register_l(char *key, long *dest);
void kvp_register_ul(char *key, unsigned long *dest);
void kvp_register_ll(char *key, long long *dest);
void kvp_register_ull(char *key, unsigned long long *dest);
void kvp_register_f(char *key, float *dest);
void kvp_register_d(char *key, double *dest);
void kvp_register_string(char *key, char *dest, size_t maxlen);

/*Read key value pairs*/
void kvp_from_text(char *text);
void kvp_from_const_text(const char* text);
void kvp_from_file(const char* filename);

#ifdef KVP_HEADER_ONLY
#include "../src/kvp.c"
#endif

#endif
