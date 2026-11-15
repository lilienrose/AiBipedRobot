#ifndef ROBOT_H
#define ROBOT_H

#include <Arduino.h>


void setupWiFi();
void setupServos();
void setupSensors();


void updateServosSmooth();
float readDistanceCM();
void handleUDP();

#endif // ROBOT_H