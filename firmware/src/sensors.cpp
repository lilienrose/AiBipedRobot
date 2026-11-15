#include "sensors.h"
#include "config.h"
#include <Wire.h>
#include <math.h>

const uint8_t MPU6050_ADDR = 0x68;

void setupSensors() {
  pinMode(PIN_HC_TRIG, OUTPUT);
  pinMode(PIN_HC_ECHO, INPUT);
  digitalWrite(PIN_HC_TRIG, LOW);

  Wire.begin(PIN_MPU_SDA, PIN_MPU_SCL);

  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x6B); // Power Management 1 register
  Wire.write(0);    // Probuzení čipu z reźimu spánku
  Wire.endTransmission(true);

  Serial.println("Senzory HC-SR04 a MPU6050 inicializovány.");
}

float readDistanceCM() {
  digitalWrite(PIN_HC_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_HC_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_HC_TRIG, LOW);

  long duration = pulseIn(PIN_HC_ECHO, HIGH, 30000);
  if (duration == 0) return -1.0;

  return (duration * 0.0343) / 2.0;
}

MPUData readMPUData() {
  MPUData data = {0.0f, 0.0f};

  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x3B); // Registr ACCEL_XOUT_H
  if (Wire.endTransmission(false) != 0) {
    return data;
  }

  Wire.requestFrom(MPU6050_ADDR, (uint8_t)6, (uint8_t)true);
  if (Wire.available() >= 6) {
    int16_t accX = (Wire.read() << 8) | Wire.read();
    int16_t accY = (Wire.read() << 8) | Wire.read();
    int16_t accZ = (Wire.read() << 8) | Wire.read();

    data.pitch = atan2((float)accY, sqrt((float)accX * accX + (float)accZ * accZ)) * 180.0 / M_PI;
    data.roll  = atan2(-(float)accX, (float)accZ) * 180.0 / M_PI;
  }

  return data;
}