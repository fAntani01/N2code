/*
 *
 * This file is part of 3MTSLib Software Development Kit.
 * All Rights Reserved.
 *
 * Contact information:
 * www.senis.ch
 * www.matesy.de
 * www.innovent-jena.de
 *
 */

#ifndef A3MTSLib_h
#define A3MTSLib_h

#ifndef EXPORT
#define EXPORT
#endif

/******************************************************************************/

typedef struct {
    int dimSize;
    int elt[1];
    } TD2;
typedef TD2 **TD2Hdl;

typedef struct {
    int dimSize;
    char elt[1];
    } TD3;
typedef TD3 **TD3Hdl;

/******************************************************************************
                              Function declaration
******************************************************************************/

EXPORT  int clear_buffer(int* device_number);
EXPORT  int count_devices(unsigned short* number_of_devices);
EXPORT  int open_device(int* device_number);
EXPORT  int get_sensor_count(int* device_number,int *sensor_count);
EXPORT  int get_sensor_values (int* device_number,unsigned long* timestamp, TD2Hdl values );
EXPORT  int get_sensor_values_fl (int *device_number, unsigned long *timestamp, float *sensorx, float *sensory, float *sensorz );
EXPORT  int set_range(int* device_number,unsigned short range);
EXPORT  int get_range(int* device_number,unsigned short* range);
EXPORT  int set_trigger(int* device_number,unsigned short range);
EXPORT  int get_trigger(int* device_number,unsigned short* range);
EXPORT  int set_speed(int* device_number,unsigned short range);
EXPORT  int get_speed(int* device_number,unsigned short* range);
EXPORT  int get_firmware_version(int* device_number,TD3Hdl values );
EXPORT  int get_firmware_version_ch(int* device_number,char *values );
EXPORT  int get_device_name(int* device_number,TD3Hdl values );
EXPORT  int get_device_name_ch(int *device_number, char *name );
EXPORT  int close_device(int* device_number);

#endif

/******************************************************************************/
