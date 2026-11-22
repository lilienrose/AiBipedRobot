#include <Arduino.h>
#include "config.h"
#include "servo_control.h"
#include "robot_wifi.h"
#include "sensors.h"
#include "gait_sequencer.h"

unsigned long lastSensorRead = 0;
const int SENSOR_READ_INTERVAL = 200;
const float MAX_TILT_ANGLE = 45.0; // Maximální bezpečný náklon ve stupních (ochrana proti pádu)

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n--- Spouštění robota ---");

  setupWiFi();
  setupServos();
  setupSensors();

  Serial.println("Robot je připraven.");
}

void loop() {
  handleUDP();
  updateGait();          // Aktualizace stavového stroje chůze
  updateServosSmooth();  // Plynulý posun serv po 3ms

  if (millis() - lastSensorRead >= SENSOR_READ_INTERVAL) {
    lastSensorRead = millis();
    
    // 1. Kontrola překážek ultrazvukem (HC-SR04)
    float distance = readDistanceCM();
    if (distance > 0 && distance < 15.0) {
      Serial.printf("Překážka: %.1f cm -> Zastavuji chůzi\n", distance);
      stopGait();
    }

    // 2. Kontrola stability a pádu gyroskopem/akcelerometrem (MPU6050)
    MPUData tilt = readMPUData();
    if (abs(tilt.pitch) > MAX_TILT_ANGLE || abs(tilt.roll) > MAX_TILT_ANGLE) {
      Serial.printf("Kritický náklon (Pitch: %.1f, Roll: %.1f) -> Nouzové zastavení!\n", tilt.pitch, tilt.roll);
      stopGait();
    }
  }
}