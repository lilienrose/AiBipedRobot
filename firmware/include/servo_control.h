#ifndef SERVO_CONTROL_H
#define SERVO_CONTROL_H

#include <Arduino.h>
#include "config.h"

void setupServos();
void updateServosSmooth();
void setTargetAngles(const int newAngles[NUM_SERVOS]);
void setHomePosition();
bool areServosAtTarget();
#endif // SERVO_CONTROL_H