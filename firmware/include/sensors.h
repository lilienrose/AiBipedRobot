#ifndef SENSORS_H
#define SENSORS_H

#include <Arduino.h>

struct MPUData {
  float pitch;
  float roll;
};

void setupSensors();
float readDistanceCM();
MPUData readMPUData();

#endif // SENSORS_H