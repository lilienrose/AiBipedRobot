#include "servo_control.h"
#include <ESP32Servo.h>

Servo servos[NUM_SERVOS];
int currentAngles[NUM_SERVOS];
int targetAngles[NUM_SERVOS];

unsigned long lastServoUpdate = 0;
const int STAGGER_INTERVAL = 3;
int activeServoIndex = 0;

void setupServos() {
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  for (int i = 0; i < NUM_SERVOS; i++) {
    servos[i].setPeriodHertz(50);
    servos[i].attach(SERVO_PINS[i], 500, 2400);

    currentAngles[i] = HOME_ANGLES[i];
    targetAngles[i] = HOME_ANGLES[i];
    
    int rawAngle = constrain(currentAngles[i] + SERVO_OFFSETS[i], 0, 180);
    servos[i].write(rawAngle);
  }
  Serial.println("Všech 6 serv nastaveno do výchozí pozice s offsety.");
}

void updateServosSmooth() {
  if (millis() - lastServoUpdate < STAGGER_INTERVAL) {
    return;
  }
  lastServoUpdate = millis();

  int i = activeServoIndex;

  if (currentAngles[i] < targetAngles[i]) {
    currentAngles[i]++;
    int rawAngle = constrain(currentAngles[i] + SERVO_OFFSETS[i], 0, 180);
    servos[i].write(rawAngle);
  } else if (currentAngles[i] > targetAngles[i]) {
    currentAngles[i]--;
    int rawAngle = constrain(currentAngles[i] + SERVO_OFFSETS[i], 0, 180);
    servos[i].write(rawAngle);
  }

  activeServoIndex = (activeServoIndex + 1) % NUM_SERVOS;
}

void setTargetAngles(const int newAngles[NUM_SERVOS]) {
  for (int i = 0; i < NUM_SERVOS; i++) {
    targetAngles[i] = constrain(newAngles[i], 0, 180);
  }
}

void setHomePosition() {
  for (int i = 0; i < NUM_SERVOS; i++) {
    targetAngles[i] = HOME_ANGLES[i];
  }
}

bool areServosAtTarget() {
  for (int i = 0; i < NUM_SERVOS; i++) {
    if (currentAngles[i] != targetAngles[i]) return false;
  }
  return true;
}